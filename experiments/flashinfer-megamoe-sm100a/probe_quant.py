"""Time and check formulations of the NVFP4 quantize tile on real hardware.

`probe_lower.py` could only ask "does it build"; with a B200 the question becomes "is it
right, and how fast". The shipped version expands the per-block scale into a full-width
divisor with 16 one-hot masked accumulations, which the benchmark says costs more than
both GEMMs' worth of time. Everything here is an attempt to do that expansion cheaply.
"""
import functools
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace  # noqa: E402

from nvfp4 import FP4_DTYPE, FP4_MAX, SF_DTYPE, SF_MAX, SF_VEC_SIZE, quantize_nvfp4  # noqa: E402

M, TK = 128, 256
NB = TK // SF_VEC_SIZE
L, ROWS, K = 8, 512, 2048


def block_scales(vals, gs):
    """The part every variant shares: a per-16-column max, then the e4m3 block scale."""
    sf_cols = []
    for b in range(NB):
        block = vals[:, b * SF_VEC_SIZE:(b + 1) * SF_VEC_SIZE]
        amax = plgpu.layout_cast(jnp.max(jnp.abs(block), axis=-1),
                                 plgpu.Layout.WGMMA.reduce(1))
        sf = jnp.clip(amax / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE).astype(jnp.float32)
        sf_cols.append(sf)
    return sf_cols


def v_onehot(vals, gs):
    """What ships today: accumulate each block through a constant one-hot mask."""
    inv_full = jnp.zeros((M, TK), jnp.float32)
    sf_full = jnp.zeros((M, NB), jnp.float32)
    for b, sf in enumerate(block_scales(vals, gs)):
        cols = (jnp.arange(TK) // SF_VEC_SIZE == b).astype(jnp.float32)
        slot = (jnp.arange(NB) == b).astype(jnp.float32)
        step = sf / gs
        inv = jnp.where(step > 0.0, 1.0, 0.0) / jnp.where(step == 0.0, 1.0, step)
        inv_full += inv[:, None] * cols[None, :]
        sf_full += sf[:, None] * slot[None, :]
    return jnp.clip(vals * inv_full, -FP4_MAX, FP4_MAX).astype(FP4_DTYPE), sf_full.astype(SF_DTYPE)


def _stack_sf(sf_cols):
    return jnp.concatenate([s[:, None] for s in sf_cols], axis=-1)  # (M, NB)


def v_repeat(vals, gs):
    """Stack the scales, then widen with jnp.repeat instead of masked adds."""
    sf_full = _stack_sf(block_scales(vals, gs))
    step = sf_full / gs
    inv = jnp.where(step > 0.0, 1.0, 0.0) / jnp.where(step == 0.0, 1.0, step)
    inv_full = jnp.repeat(inv, SF_VEC_SIZE, axis=-1)
    return jnp.clip(vals * inv_full, -FP4_MAX, FP4_MAX).astype(FP4_DTYPE), sf_full.astype(SF_DTYPE)


def v_broadcast(vals, gs):
    """Widen by broadcasting into (M, NB, 16) and reshaping back."""
    sf_full = _stack_sf(block_scales(vals, gs))
    step = sf_full / gs
    inv = jnp.where(step > 0.0, 1.0, 0.0) / jnp.where(step == 0.0, 1.0, step)
    inv_full = jnp.broadcast_to(inv[:, :, None], (M, NB, SF_VEC_SIZE)).reshape(M, TK)
    return jnp.clip(vals * inv_full, -FP4_MAX, FP4_MAX).astype(FP4_DTYPE), sf_full.astype(SF_DTYPE)


def v_perblock_mul(vals, gs):
    """No full-width divisor at all: scale each 16-column slice and concatenate."""
    outs = []
    sf_cols = block_scales(vals, gs)
    for b, sf in enumerate(sf_cols):
        block = vals[:, b * SF_VEC_SIZE:(b + 1) * SF_VEC_SIZE]
        step = sf / gs
        inv = jnp.where(step > 0.0, 1.0, 0.0) / jnp.where(step == 0.0, 1.0, step)
        outs.append(jnp.clip(block * inv[:, None], -FP4_MAX, FP4_MAX))
    q = jnp.concatenate(outs, axis=-1).astype(FP4_DTYPE)
    return q, _stack_sf(sf_cols).astype(SF_DTYPE)


def v_floor(vals, gs):
    """Ablation, wrong on purpose: no per-block work at all. The memory-bound floor."""
    sf_full = jnp.full((M, NB), 1.0, jnp.float32)
    return jnp.clip(vals, -FP4_MAX, FP4_MAX).astype(FP4_DTYPE), sf_full.astype(SF_DTYPE)


def build(compute):
    def body(x_gmem, gs_gmem, q_gmem, sf_gmem):
        e, mi, kb = (lax.axis_index(a) for a in ("l", "mi", "kb"))
        ms = pl.ds(mi * M, M)

        @functools.partial(
            pl.run_scoped,
            x_smem=plgpu.SMEM((M, TK), jnp.bfloat16),
            q_smem=plgpu.SMEM((M, TK), FP4_DTYPE),
            sf_smem=plgpu.SMEM((M, NB), SF_DTYPE),
            barrier=plgpu.Barrier(),
        )
        def scoped(x_smem, q_smem, sf_smem, barrier):
            cols = pl.ds(kb * TK, TK)
            plgpu.copy_gmem_to_smem(x_gmem.at[e, ms, cols], x_smem, barrier)
            plgpu.barrier_wait(barrier)
            vals = plgpu.layout_cast(x_smem[...].astype(jnp.float32), plgpu.Layout.WGMMA)
            q_smem[...], sf_smem[...] = compute(vals, gs_gmem[e])
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(q_smem, q_gmem.at[e, ms, cols])
            plgpu.copy_smem_to_gmem(sf_smem, sf_gmem.at[e, ms, pl.ds(kb * NB, NB)])
            plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        body,
        out_shape=(jax.ShapeDtypeStruct((L, ROWS, K), FP4_DTYPE),
                   jax.ShapeDtypeStruct((L, ROWS, K // SF_VEC_SIZE), SF_DTYPE)),
        grid=(L, ROWS // M, K // TK), grid_names=("l", "mi", "kb"),
    )


x = jax.random.normal(jax.random.key(0), (L, ROWS, K), jnp.float32).astype(jnp.bfloat16)
gs = jnp.full((L,), 64.0, jnp.float32)
want_q, want_sf = quantize_nvfp4(x.astype(jnp.float32), gs)

for name, fn in [("one-hot masked adds (shipped)", v_onehot), ("jnp.repeat", v_repeat),
                 ("broadcast + reshape", v_broadcast), ("per-block mul + concat", v_perblock_mul),
                 ("ablation: no per-block work", v_floor)]:
    try:
        f = jax.jit(build(fn))
        q, sf = jax.block_until_ready(f(x, gs))
        exact = (np.array_equal(np.asarray(q, np.float32), np.asarray(want_q, np.float32))
                 and np.array_equal(np.asarray(sf, np.float32), np.asarray(want_sf, np.float32)))
        with KernelTrace() as tr:
            for _ in range(20):
                jax.block_until_ready(f(x, gs))
        us = sum(r.duration_us for r in tr.records) / 20
        tag = "exact" if exact else ("WRONG (expected)" if fn is v_floor else "MISMATCH")
        print(f"{us:7.1f} us  {tag:<17} {name}")
    except Exception as e:
        print(f"{'  --':>7}     {'no lowering':<17} {name}: {' '.join(str(e).split())[:70]}")
