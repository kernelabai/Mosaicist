"""Pallas Mosaic GPU quantization kernels for the Hopper MoE analog.

Two kernels, matching the reference's fused steps:

  `quantize_blockwise_pallas`  ~ FlashInfer's `scaled_fp4_grouped_quantize`
  `silu_mul_quantize_pallas`   ~ FlashInfer's `silu_and_mul_scaled_nvfp4_experts_quantize`

Both produce e4m3 values plus e4m3 per-block scales in the K-major layout the GEMM
wants: (l, blocks, rows). A block is BLOCK_K elements along the reduced axis; the
per-expert global scale keeps the block scales inside e4m3's range.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from quant import BLOCK_K, FP8_DTYPE, FP8_MAX, SF_MAX

TILE_M = 128


def _quantize_tile(vals: jax.Array, gs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(tile_m, BLOCK_K) fp32 -> (e4m3 values, e4m3 per-row block scale), as quant.py does.

    Two concessions to Mosaic GPU's layout inference, both load-bearing:

      * the layouts are annotated explicitly -- inference cannot solve a row reduction
        plus a broadcast back on its own (it fails with "Layout inference failed to
        find a solution"). WGMMA layout for the tile, its reduced variant for the
        per-row scale.
      * `quant.py`'s `where(step > 0, ..., 0)` guard, which zeroes a row whose block
        scale flushed to zero in e4m3, is expressed as a multiply by a 0/1 mask instead
        of a select. A select whose *condition* is broadcast from the reduced layout to
        the full tile has no layout either; a broadcast multiply does. The result is
        identical: x * 1.0 is exact and x * 0.0 is zero.
    """
    vals = plgpu.layout_cast(vals, plgpu.Layout.WGMMA)
    amax = plgpu.layout_cast(jnp.max(jnp.abs(vals), axis=-1), plgpu.Layout.WGMMA.reduce(1))  # (tile_m,)
    sf = jnp.clip(amax / FP8_MAX * gs, 0.0, SF_MAX).astype(FP8_DTYPE)
    step = sf.astype(jnp.float32) / gs
    safe = jnp.where(step == 0.0, 1.0, step)  # avoid a divide by zero in the dead rows
    keep = jnp.where(step > 0.0, 1.0, 0.0)  # ... then mask them out
    q = jnp.clip(vals / safe[:, None] * keep[:, None], -FP8_MAX, FP8_MAX)
    return q.astype(FP8_DTYPE), sf


def quantize_blockwise_pallas(x: jax.Array, global_scale: jax.Array, masked_m: jax.Array):
    """(l, m, k) bf16 -> ((l, m, k) e4m3, (l, k // BLOCK_K, m) e4m3)."""
    l, m, k = x.shape
    if m % TILE_M or k % BLOCK_K:
        raise ValueError(f"({m}, {k}) must be tiled by ({TILE_M}, {BLOCK_K})")

    def body(x_gmem, gs_gmem, mask_gmem, q_gmem, sf_gmem):
        e, mi, kb = (lax.axis_index(ax) for ax in ("l", "mi", "kb"))
        m_slice = pl.ds(mi * TILE_M, TILE_M)

        @pl.when(mi * TILE_M < mask_gmem[e])
        def _():
            @functools.partial(
                pl.run_scoped,
                x_smem=plgpu.SMEM((TILE_M, BLOCK_K), x.dtype),
                q_smem=plgpu.SMEM((TILE_M, BLOCK_K), FP8_DTYPE),
                sf_smem=plgpu.SMEM((TILE_M,), FP8_DTYPE),
                barrier=plgpu.Barrier(),
            )
            def scoped(x_smem, q_smem, sf_smem, barrier):
                plgpu.copy_gmem_to_smem(x_gmem.at[e, m_slice, pl.ds(kb * BLOCK_K, BLOCK_K)], x_smem, barrier)
                plgpu.barrier_wait(barrier)
                q, sf = _quantize_tile(x_smem[...].astype(jnp.float32), gs_gmem[e])
                q_smem[...] = q
                sf_smem[...] = sf
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(q_smem, q_gmem.at[e, m_slice, pl.ds(kb * BLOCK_K, BLOCK_K)])
                plgpu.copy_smem_to_gmem(sf_smem, sf_gmem.at[e, kb, m_slice])
                plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        body,
        out_shape=(jax.ShapeDtypeStruct((l, m, k), FP8_DTYPE),
                   jax.ShapeDtypeStruct((l, k // BLOCK_K, m), FP8_DTYPE)),
        grid=(l, m // TILE_M, k // BLOCK_K),
        grid_names=("l", "mi", "kb"),
    )(x, global_scale, masked_m)


def silu_mul_quantize_pallas(gateup: jax.Array, global_scale: jax.Array, masked_m: jax.Array):
    """(l, m, 2n) bf16 -> ((l, m, n) e4m3, (l, n // BLOCK_K, m) e4m3).

    silu(gate) * up over the two halves, then the same blockwise quantization; this is
    GEMM2's A operand, so the scale blocks run along n (GEMM2's K).
    """
    l, m, two_n = gateup.shape
    n = two_n // 2
    if two_n % 2 or m % TILE_M or n % BLOCK_K:
        raise ValueError(f"({m}, {two_n}) must be tiled by ({TILE_M}, 2 * {BLOCK_K})")

    def body(x_gmem, gs_gmem, mask_gmem, q_gmem, sf_gmem):
        e, mi, kb = (lax.axis_index(ax) for ax in ("l", "mi", "kb"))
        m_slice = pl.ds(mi * TILE_M, TILE_M)

        @pl.when(mi * TILE_M < mask_gmem[e])
        def _():
            @functools.partial(
                pl.run_scoped,
                gate_smem=plgpu.SMEM((TILE_M, BLOCK_K), gateup.dtype),
                up_smem=plgpu.SMEM((TILE_M, BLOCK_K), gateup.dtype),
                q_smem=plgpu.SMEM((TILE_M, BLOCK_K), FP8_DTYPE),
                sf_smem=plgpu.SMEM((TILE_M,), FP8_DTYPE),
                barrier=plgpu.Barrier(num_arrivals=2),
            )
            def scoped(gate_smem, up_smem, q_smem, sf_smem, barrier):
                cols = pl.ds(kb * BLOCK_K, BLOCK_K)
                up_cols = pl.ds(n + kb * BLOCK_K, BLOCK_K)
                plgpu.copy_gmem_to_smem(x_gmem.at[e, m_slice, cols], gate_smem, barrier)
                plgpu.copy_gmem_to_smem(x_gmem.at[e, m_slice, up_cols], up_smem, barrier)
                plgpu.barrier_wait(barrier)
                gate = gate_smem[...].astype(jnp.float32)
                act = jax.nn.sigmoid(gate) * gate * up_smem[...].astype(jnp.float32)
                q, sf = _quantize_tile(act, gs_gmem[e])
                q_smem[...] = q
                sf_smem[...] = sf
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(q_smem, q_gmem.at[e, m_slice, cols])
                plgpu.copy_smem_to_gmem(sf_smem, sf_gmem.at[e, kb, m_slice])
                plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        body,
        out_shape=(jax.ShapeDtypeStruct((l, m, n), FP8_DTYPE),
                   jax.ShapeDtypeStruct((l, n // BLOCK_K, m), FP8_DTYPE)),
        grid=(l, m // TILE_M, n // BLOCK_K),
        grid_names=("l", "mi", "kb"),
    )(gateup, global_scale, masked_m)
