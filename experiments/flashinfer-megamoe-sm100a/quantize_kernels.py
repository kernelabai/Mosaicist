"""NVFP4 quantization kernels for the Blackwell port.

  `quantize_nvfp4_pallas`           ~ FlashInfer's `scaled_fp4_grouped_quantize`
  `silu_mul_quantize_nvfp4_pallas`  ~ `silu_and_mul_scaled_nvfp4_experts_quantize`

UNVERIFIED on hardware; see `masked_gemm.py`. Both lower for sm_100a
(`check_lowering.py`) and both follow the arithmetic in `nvfp4.quantize_nvfp4`.

Four shapes here are forced rather than chosen. Each was found by `probe_lower.py`,
which bisects formulations by whether they survive Mosaic's lowering:

  * `TILE_K` is 256, not 128, because the scale tile is stored with TMA and TMA needs
    at least 128 bits along the last dimension. 256 elements of K give 16 e4m3 scales
    per row, which is exactly 128 bits; 128 elements give 8 and are rejected.
  * the per-16-element block maxima come from a static Python loop over 2D column
    slices. Reshaping the tile to (rows, blocks, 16) and reducing the last axis has no
    layout solution, with or without annotations, under WGMMA or WG_STRIDED.
  * the per-block results are accumulated into full-width arrays through constant
    one-hot masks, rather than concatenated or written to column slices of smem.
    A concatenate has no layout; a strided store into a swizzled ref is rejected
    outright ("We cannot apply swizzle to non-contiguous refs").
  * the "scale flushed to zero" guard is a multiply by a 0/1 mask rather than a
    `jnp.where` whose condition broadcasts from the reduced layout to the full tile --
    that select has no layout either. `x * 1.0` is exact and `x * 0.0` is zero, so the
    arithmetic is identical.

Scales come out row-major, `(l, m, k // 16)`, which is the layout FlashInfer documents
for these kernels. The MMA wants them tiled; `moe.py` applies `to_mma_scale_layout`
between the stages. A production version would have the kernel write the tiled layout
directly, but that store is a scatter within the tile (row r, block c lands at
[c // 4, r % 32, 4 * ((r % 128) // 32) + c % 4]) and is not expressible as a vector
store, so it would need a different epilogue rather than a tweak to this one.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from nvfp4 import FP4_DTYPE, FP4_MAX, SF_DTYPE, SF_MAX, SF_VEC_SIZE

LOG2E = 1.4426950408889634


def _silu(g: jax.Array) -> jax.Array:
    """silu(g) = g * sigmoid(g), written so it lowers to the hardware exponential.

    `jax.nn.sigmoid` and `lax.logistic` both cost 42.1 us on the fused kernel where this
    costs 27.1 (the rest of that kernel is 11.8), and all three agree bit-for-bit. The
    win is `exp2`: `exp` does not reach `ex2.approx` through this lowering. A fast
    reciprocal would likely take another few us, but `pl.reciprocal` is not implemented
    in Mosaic GPU's warpgroup lowering, so the divide stays.
    """
    return g / (1.0 + jnp.exp2(-g * LOG2E))

APPROX_MATH = True
"""Lower divisions and transcendentals with the `afn` fast-math flag.

The fused silu+quantize kernel's PTX carries 448 `div.full.f32` -- the multi-instruction
full-precision divide -- against 256 `ex2.approx.f32`. The exponential was never the
problem; the divides are. This flag turns them into single-instruction approximate
divides. Both outputs are e2m1 values and e4m3 scales, 3 and 4 mantissa bits, so the
extra error is nowhere near a representable step, and `probe_approx.py` checks that the
quantized results stay bit-identical to `nvfp4.py` rather than assuming it."""

TILE_M = 128
SUB_K = 64
"""Columns spanned by one round of the one-hot expansion, and by one output store.

The expansion costs a full-width multiply-add per 16-element block, so a span of W
columns pays W/16 masks over W columns. Doing it per sub-tile instead of per tile makes
the total O(TILE_K * SUB_K / 16) rather than O(TILE_K^2 / 16): at SUB_K=64 the plain
quantize kernel drops from 14.2 us to 10.3 (its memory-bound floor is 5.5). Each
sub-tile is stored separately, which is why it is a leading index of `q_smem` -- a
column slice of the swizzled output cannot be written. 64 is the floor: TMA wants at
least 16 bytes per row, e2m1 is half a byte, and SUB_K=32 sits exactly on that minimum
and produced wrong values."""

TILE_K = 256  # elements of K per tile; gives TILE_K // SF_VEC_SIZE = 16 scales,
              # the narrowest scale row TMA will store (128 bits)


def _quantize_tile(vals: jax.Array, gs: jax.Array, q_smem, sf_smem) -> None:
    """(TILE_M, TILE_K) fp32 -> e2m1 values in `q_smem`, e4m3 block scales in `sf_smem`.

    Follows `nvfp4.quantize_nvfp4` exactly; only the shape of the computation differs,
    for the lowering reasons in the module docstring. `q_smem` is (TILE_K // SUB_K,
    TILE_M, SUB_K) and each sub-tile is written whole.
    """
    nsub, nb_sub = TILE_K // SUB_K, SUB_K // SF_VEC_SIZE
    nb = TILE_K // SF_VEC_SIZE
    vals = plgpu.layout_cast(vals, plgpu.Layout.WGMMA)
    sf_full = jnp.zeros((TILE_M, nb), jnp.float32)
    for s in range(nsub):
        sub = vals[:, s * SUB_K:(s + 1) * SUB_K]
        inv_sub = jnp.zeros((TILE_M, SUB_K), jnp.float32)
        for j in range(nb_sub):
            block = sub[:, j * SF_VEC_SIZE:(j + 1) * SF_VEC_SIZE]
            amax = plgpu.layout_cast(jnp.max(jnp.abs(block), axis=-1),
                                     plgpu.Layout.WGMMA.reduce(1))  # (TILE_M,)
            # round-tripping through e4m3 here is what the format stores, and casting
            # the accumulated result back at the end is exact because it is idempotent
            sf = jnp.clip(amax / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE).astype(jnp.float32)
            step = sf / gs
            inv = jnp.where(step > 0.0, 1.0, 0.0) / jnp.where(step == 0.0, 1.0, step)
            inv_sub += inv[:, None] * (
                jnp.arange(SUB_K) // SF_VEC_SIZE == j).astype(jnp.float32)[None, :]
            sf_full += sf[:, None] * (
                jnp.arange(nb) == s * nb_sub + j).astype(jnp.float32)[None, :]
        q_smem[s] = jnp.clip(sub * inv_sub, -FP4_MAX, FP4_MAX).astype(FP4_DTYPE)
    sf_smem[...] = sf_full.astype(SF_DTYPE)


def quantize_nvfp4_pallas(x: jax.Array, global_scale: jax.Array, masked_m: jax.Array):
    """(l, m, k) bf16 -> ((l, m, k) e2m1, (l, m, k // 16) e4m3)."""
    l, m, k = x.shape
    if m % TILE_M or k % TILE_K:
        raise ValueError(f"({m}, {k}) must be tiled by ({TILE_M}, {TILE_K})")
    ks = TILE_K // SF_VEC_SIZE

    def body(x_gmem, gs_gmem, mask_gmem, q_gmem, sf_gmem):
        e, mi, kb = (lax.axis_index(ax) for ax in ("l", "mi", "kb"))
        m_slice = pl.ds(mi * TILE_M, TILE_M)

        @pl.when(mi * TILE_M < mask_gmem[e])
        def _():
            @functools.partial(
                pl.run_scoped,
                x_smem=plgpu.SMEM((TILE_M, TILE_K), x.dtype),
                q_smem=plgpu.SMEM((TILE_K // SUB_K, TILE_M, SUB_K), FP4_DTYPE),
                sf_smem=plgpu.SMEM((TILE_M, ks), SF_DTYPE),
                barrier=plgpu.Barrier(),
            )
            def scoped(x_smem, q_smem, sf_smem, barrier):
                cols = pl.ds(kb * TILE_K, TILE_K)
                plgpu.copy_gmem_to_smem(x_gmem.at[e, m_slice, cols], x_smem, barrier)
                plgpu.barrier_wait(barrier)
                _quantize_tile(x_smem[...].astype(jnp.float32), gs_gmem[e], q_smem, sf_smem)
                plgpu.commit_smem()
                for sub in range(TILE_K // SUB_K):
                    plgpu.copy_smem_to_gmem(
                        q_smem.at[sub],
                        q_gmem.at[e, m_slice, pl.ds(kb * TILE_K + sub * SUB_K, SUB_K)])
                plgpu.copy_smem_to_gmem(sf_smem, sf_gmem.at[e, m_slice, pl.ds(kb * ks, ks)])
                plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        body,
        out_shape=(jax.ShapeDtypeStruct((l, m, k), FP4_DTYPE),
                   jax.ShapeDtypeStruct((l, m, k // SF_VEC_SIZE), SF_DTYPE)),
        grid=(l, m // TILE_M, k // TILE_K),
        grid_names=("l", "mi", "kb"),
        compiler_params=plgpu.CompilerParams(approx_math=APPROX_MATH),
    )(x, global_scale, masked_m)


def silu_mul_quantize_nvfp4_pallas(gateup: jax.Array, global_scale: jax.Array,
                                   masked_m: jax.Array):
    """(l, m, 2n) bf16 -> ((l, m, n) e2m1, (l, m, n // 16) e4m3).

    silu(gate) * up over the two halves, then the same quantization. The activation is
    never rounded to bf16 in between -- it is fused, exactly as in the reference.
    """
    l, m, two_n = gateup.shape
    n = two_n // 2
    if two_n % 2 or m % TILE_M or n % TILE_K:
        raise ValueError(f"({m}, {two_n}) must be tiled by ({TILE_M}, 2 * {TILE_K})")
    ks = TILE_K // SF_VEC_SIZE

    def body(x_gmem, gs_gmem, mask_gmem, q_gmem, sf_gmem):
        e, mi, kb = (lax.axis_index(ax) for ax in ("l", "mi", "kb"))
        m_slice = pl.ds(mi * TILE_M, TILE_M)

        @pl.when(mi * TILE_M < mask_gmem[e])
        def _():
            @functools.partial(
                pl.run_scoped,
                gate_smem=plgpu.SMEM((TILE_M, TILE_K), gateup.dtype),
                up_smem=plgpu.SMEM((TILE_M, TILE_K), gateup.dtype),
                q_smem=plgpu.SMEM((TILE_K // SUB_K, TILE_M, SUB_K), FP4_DTYPE),
                sf_smem=plgpu.SMEM((TILE_M, ks), SF_DTYPE),
                barrier=plgpu.Barrier(num_arrivals=2),
            )
            def scoped(gate_smem, up_smem, q_smem, sf_smem, barrier):
                cols = pl.ds(kb * TILE_K, TILE_K)
                up_cols = pl.ds(n + kb * TILE_K, TILE_K)
                plgpu.copy_gmem_to_smem(x_gmem.at[e, m_slice, cols], gate_smem, barrier)
                plgpu.copy_gmem_to_smem(x_gmem.at[e, m_slice, up_cols], up_smem, barrier)
                plgpu.barrier_wait(barrier)
                gate = gate_smem[...].astype(jnp.float32)
                act = _silu(gate) * up_smem[...].astype(jnp.float32)
                _quantize_tile(act, gs_gmem[e], q_smem, sf_smem)
                plgpu.commit_smem()
                for sub in range(TILE_K // SUB_K):
                    plgpu.copy_smem_to_gmem(
                        q_smem.at[sub],
                        q_gmem.at[e, m_slice, pl.ds(kb * TILE_K + sub * SUB_K, SUB_K)])
                plgpu.copy_smem_to_gmem(sf_smem, sf_gmem.at[e, m_slice, pl.ds(kb * ks, ks)])
                plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        body,
        out_shape=(jax.ShapeDtypeStruct((l, m, n), FP4_DTYPE),
                   jax.ShapeDtypeStruct((l, m, n // SF_VEC_SIZE), SF_DTYPE)),
        grid=(l, m // TILE_M, n // TILE_K),
        grid_names=("l", "mi", "kb"),
        compiler_params=plgpu.CompilerParams(approx_math=APPROX_MATH),
    )(gateup, global_scale, masked_m)
