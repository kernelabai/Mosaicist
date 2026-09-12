"""Shrink the span of the one-hot expansion.

The masked expansion costs one full-tile multiply-add per 16-element block, so a tile of
width T pays T/16 masks over T columns: O(T^2/16). Splitting the tile into sub-tiles of
width W and storing each separately makes it O(T*W/16) -- 8x less at W=32 than W=256.

The catch is the store. A column slice of the swizzled fp4 output cannot be written, but
a leading index of a (nsub, M, W) buffer can, at the price of one TMA per sub-tile. TMA
needs >=16 bytes per row, and e2m1 is half a byte, so W >= 32.
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


def _scale_of(block, gs):
    amax = plgpu.layout_cast(jnp.max(jnp.abs(block), axis=-1), plgpu.Layout.WGMMA.reduce(1))
    return jnp.clip(amax / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE).astype(jnp.float32)


def _inv(sf, gs):
    step = sf / gs
    return jnp.where(step > 0.0, 1.0, 0.0) / jnp.where(step == 0.0, 1.0, step)


def build(w):
    """w == TK reproduces the shipped kernel (one store, masks over the whole tile)."""
    nsub, nb_sub = TK // w, w // SF_VEC_SIZE

    def body(x_gmem, gs_gmem, q_gmem, sf_gmem):
        e, mi, kb = (lax.axis_index(a) for a in ("l", "mi", "kb"))
        ms = pl.ds(mi * M, M)

        @functools.partial(
            pl.run_scoped,
            x_smem=plgpu.SMEM((M, TK), jnp.bfloat16),
            q_smem=plgpu.SMEM((nsub, M, w), FP4_DTYPE),
            sf_smem=plgpu.SMEM((M, NB), SF_DTYPE),
            barrier=plgpu.Barrier())
        def scoped(x_smem, q_smem, sf_smem, barrier):
            cols = pl.ds(kb * TK, TK)
            plgpu.copy_gmem_to_smem(x_gmem.at[e, ms, cols], x_smem, barrier)
            plgpu.barrier_wait(barrier)
            vals = plgpu.layout_cast(x_smem[...].astype(jnp.float32), plgpu.Layout.WGMMA)
            gs = gs_gmem[e]

            sf_full = jnp.zeros((M, NB), jnp.float32)
            for s in range(nsub):
                sub = vals[:, s * w:(s + 1) * w]
                inv_sub = jnp.zeros((M, w), jnp.float32)
                for j in range(nb_sub):
                    sf = _scale_of(sub[:, j * SF_VEC_SIZE:(j + 1) * SF_VEC_SIZE], gs)
                    inv_sub += _inv(sf, gs)[:, None] * (
                        jnp.arange(w) // SF_VEC_SIZE == j).astype(jnp.float32)[None, :]
                    sf_full += sf[:, None] * (
                        jnp.arange(NB) == s * nb_sub + j).astype(jnp.float32)[None, :]
                q_smem[s] = jnp.clip(sub * inv_sub, -FP4_MAX, FP4_MAX).astype(FP4_DTYPE)
            sf_smem[...] = sf_full.astype(SF_DTYPE)

            plgpu.commit_smem()
            for s in range(nsub):
                plgpu.copy_smem_to_gmem(
                    q_smem.at[s], q_gmem.at[e, ms, pl.ds(kb * TK + s * w, w)])
            plgpu.copy_smem_to_gmem(sf_smem, sf_gmem.at[e, ms, pl.ds(kb * NB, NB)])
            plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        body, out_shape=(jax.ShapeDtypeStruct((L, ROWS, K), FP4_DTYPE),
                         jax.ShapeDtypeStruct((L, ROWS, K // SF_VEC_SIZE), SF_DTYPE)),
        grid=(L, ROWS // M, K // TK), grid_names=("l", "mi", "kb"))


x = jax.random.normal(jax.random.key(0), (L, ROWS, K), jnp.float32).astype(jnp.bfloat16)
gs = jnp.full((L,), 64.0, jnp.float32)
want_q, want_sf = quantize_nvfp4(x.astype(jnp.float32), gs)
wq, wsf = np.asarray(want_q, np.float32), np.asarray(want_sf, np.float32)

for w in (256, 128, 64, 32):
    try:
        f = jax.jit(build(w))
        q, sf = jax.block_until_ready(f(x, gs))
        exact = (np.array_equal(np.asarray(q, np.float32), wq)
                 and np.array_equal(np.asarray(sf, np.float32), wsf))
        with KernelTrace() as tr:
            for _ in range(20):
                jax.block_until_ready(f(x, gs))
        us = sum(r.duration_us for r in tr.records if "mosaic" in r.name) / 20
        print(f"{us:7.1f} us  {'exact' if exact else 'MISMATCH':<10} sub-tile width {w:<4}"
              f"({TK // w} stores/tile)")
    except Exception as e:
        print(f"{'  --':>7}     {'':<10} sub-tile width {w:<4}: {' '.join(str(e).split())[:70]}")
