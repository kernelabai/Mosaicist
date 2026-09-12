"""Cheaper ways to widen a per-16-column scale back across the tile, in TCGEN05 layout.

The one-hot expansion costs 9.5 us of the fused kernel's 34.2. Every cheaper form was
rejected earlier by layout inference -- but that was from the WGMMA layout, reading smem.
The fused epilogue works in TCGEN05, having come from TMEM, so the question is open again.
"""
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

from nvfp4 import FP4_DTYPE, FP4_MAX, SF_DTYPE, SF_MAX, SF_VEC_SIZE  # noqa: E402

M, TN = 128, 256
NB = TN // SF_VEC_SIZE
L, ROWS = 8, 512
RED = plgpu.Layout.TCGEN05.reduce(1)


def _scales(vals, gs):
    out = []
    for b in range(NB):
        blk = vals[:, b * SF_VEC_SIZE:(b + 1) * SF_VEC_SIZE]
        amax = plgpu.layout_cast(jnp.max(jnp.abs(blk), axis=-1), RED)
        out.append(jnp.clip(amax / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE).astype(jnp.float32))
    return out


def _inv(sf, gs):
    step = sf / gs
    return jnp.where(step > 0.0, 1.0, 0.0) / jnp.where(step == 0.0, 1.0, step)


def v_onehot(vals, gs):
    """What the fused kernel does now."""
    inv_full = jnp.zeros((M, TN), jnp.float32)
    sf_full = jnp.zeros((M, NB), jnp.float32)
    for b, sf in enumerate(_scales(vals, gs)):
        inv_full += _inv(sf, gs)[:, None] * (
            jnp.arange(TN) // SF_VEC_SIZE == b).astype(jnp.float32)[None, :]
        sf_full += sf[:, None] * (jnp.arange(NB) == b).astype(jnp.float32)[None, :]
    return jnp.clip(vals * inv_full, -FP4_MAX, FP4_MAX).astype(FP4_DTYPE), sf_full


def _stack(cols):
    return jnp.concatenate([c[:, None] for c in cols], axis=-1)


def v_repeat(vals, gs):
    sf_full = _stack(_scales(vals, gs))
    inv_full = jnp.repeat(_inv(sf_full, gs), SF_VEC_SIZE, axis=-1)
    return jnp.clip(vals * inv_full, -FP4_MAX, FP4_MAX).astype(FP4_DTYPE), sf_full


def v_broadcast(vals, gs):
    sf_full = _stack(_scales(vals, gs))
    inv_full = jnp.broadcast_to(_inv(sf_full, gs)[:, :, None],
                                (M, NB, SF_VEC_SIZE)).reshape(M, TN)
    return jnp.clip(vals * inv_full, -FP4_MAX, FP4_MAX).astype(FP4_DTYPE), sf_full


def v_perblock(vals, gs):
    """Scale each 16-column slice on its own and concatenate the quantized pieces."""
    cols = _scales(vals, gs)
    outs = [jnp.clip(vals[:, b * SF_VEC_SIZE:(b + 1) * SF_VEC_SIZE] * _inv(sf, gs)[:, None],
                     -FP4_MAX, FP4_MAX) for b, sf in enumerate(cols)]
    return jnp.concatenate(outs, axis=-1).astype(FP4_DTYPE), _stack(cols)


def build(compute):
    def body(a_gmem, q_gmem, sf_gmem, acc_tmem, buf_smem, q_smem, sf_smem, bar):
        e, mi = lax.axis_index("l"), lax.axis_index("mi")
        ms = pl.ds(mi * M, M)
        plgpu.copy_gmem_to_smem(a_gmem.at[e, ms], buf_smem, bar)
        plgpu.barrier_wait(bar)
        plgpu.async_store_tmem(acc_tmem, buf_smem[...].astype(jnp.float32))
        plgpu.commit_tmem()
        vals = plgpu.layout_cast(plgpu.async_load_tmem(acc_tmem), plgpu.Layout.TCGEN05)
        plgpu.wait_load_tmem()
        q, sf = compute(vals, 32.0)
        q_smem[...], sf_smem[...] = q, sf.astype(SF_DTYPE)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(q_smem, q_gmem.at[e, ms])
        plgpu.copy_smem_to_gmem(sf_smem, sf_gmem.at[e, ms])
        plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        body, out_shape=(jax.ShapeDtypeStruct((L, ROWS, TN), FP4_DTYPE),
                         jax.ShapeDtypeStruct((L, ROWS, NB), SF_DTYPE)),
        grid=(L, ROWS // M), grid_names=("l", "mi"),
        compiler_params=plgpu.CompilerParams(approx_math=True),
        scratch_shapes=[plgpu.TMEM((M, TN), jnp.float32),
                        plgpu.SMEM((M, TN), jnp.bfloat16),
                        plgpu.SMEM((M, TN), FP4_DTYPE),
                        plgpu.SMEM((M, NB), SF_DTYPE),
                        plgpu.Barrier()])


x = (jax.random.normal(jax.random.key(0), (L, ROWS, TN), jnp.float32) * 4).astype(jnp.bfloat16)
want = None
for name, fn in [("one-hot (current)", v_onehot), ("jnp.repeat", v_repeat),
                 ("broadcast + reshape", v_broadcast), ("per-block concat", v_perblock)]:
    try:
        f = jax.jit(build(fn))
        q, sf = jax.block_until_ready(f(x))
        got = (np.asarray(q, np.float32), np.asarray(sf, np.float32))
        if want is None:
            want, tag = got, "reference"
        else:
            tag = "same" if (np.array_equal(got[0], want[0])
                             and np.array_equal(got[1], want[1])) else "DIFFERS"
        with KernelTrace() as tr:
            for _ in range(20):
                jax.block_until_ready(f(x))
        us = sum(r.duration_us for r in tr.records if "mosaic" in r.name) / 20
        print(f"{us:7.1f} us  {tag:<10} {name}")
    except Exception as ex:
        print(f"{'  --':>7}     {'no lowering':<10} {name}: {' '.join(str(ex).split())[:60]}")
