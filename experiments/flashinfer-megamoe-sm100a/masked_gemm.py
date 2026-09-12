"""Masked, block-scaled grouped GEMM for Blackwell (sm_100a) in Pallas Mosaic GPU.

UNVERIFIED: written against the Mosaic GPU / PTX contracts, never run -- no sm_100a
device was available. Every assumption that a real run would settle is called out in a
comment marked `ASSUMPTION`. The pieces that could be checked without the hardware
(the NVFP4 arithmetic and the MMA scale tiling) are checked, in `test_nvfp4.py`.

This is the direct port of FlashInfer's `grouped_gemm_nt_masked`, as opposed to the
Hopper analog in `../flashinfer-megamoe-sm90a`. The structural difference is the whole
point of Blackwell: `tcgen05.mma.kind.block_scale` applies the per-16-element scales
*inside* the MMA, so the accumulator stays in TMEM across all of K and is read exactly
once. The Hopper analog has to read and rescale the accumulator every K block, which
measurement there showed costs more than everything else in the kernel combined.

Operand layouts:
  * `a_q` (l, m, k) and `b_q` (l, n, k) e2m1, both K-minor. B keeps the reference's NT
    layout and is handed to the MMA as a transposed view, leaving it K-minor in smem.
    ASSUMPTION: block-scaled tcgen05 requires K-minor operands, as fp8 wgmma does on
    Hopper.
  * scales arrive already in the MMA's `.scale_vec::1X` tiling, shaped
    (l, mn // 128, k // 64, 32, 16) -- see `nvfp4.to_mma_scale_layout`. A K block's
    scales are then a contiguous slab of `BLOCK_K // 64` tiles, so one TMA per stage
    lands them in exactly the smem layout `async_copy_scales_to_tmem` wants.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from nvfp4 import FP4_DTYPE, SF_DTYPE, SF_VEC_SIZE

TMEM_ROWS = 128
"""The MMA scale path is defined in terms of 128-row TMEM tiles, so `tile_m` and
`tile_n` are both fixed at 128: `async_copy_scales_to_tmem` derives the smem shape as
(mn // 128, k_scales // 4, 32, 16) and rejects anything above 2 tiles."""

SF_TILE_K = 4 * SF_VEC_SIZE  # 64 elements of K per scale tile in the MMA layout


@dataclasses.dataclass(frozen=True)
class GemmConfig:
    block_k: int = 128  # K elements per pipeline stage; must be a multiple of SF_TILE_K
    stages: int = 4


def _fp4_transforms(minor_elems: int):
    """Swizzle/tiling for a K-minor e2m1 tile. e2m1 is 4 bits, so a 128-byte swizzle
    spans 256 elements."""
    swizzle = plgpu.find_swizzle(min(minor_elems, 256) * 4)  # 4 bits per element
    return (plgpu.TilingTransform((8, swizzle * 2)), plgpu.SwizzleTransform(swizzle))


def masked_grouped_gemm(
    a_q: jax.Array,  # (l, m, k) e2m1
    a_sf: jax.Array,  # (l, m // 128, k // 64, 32, 16) e4m3, MMA scale layout
    b_q: jax.Array,  # (l, n, k) e2m1  (the reference's NT layout)
    b_sf: jax.Array,  # (l, n // 128, k // 64, 32, 16) e4m3
    alpha: jax.Array,  # (l,) f32
    masked_m: jax.Array,  # (l,) int32
    config: GemmConfig = GemmConfig(),
) -> jax.Array:
    l, m, k = a_q.shape
    _, n, _ = b_q.shape
    tm = tn = TMEM_ROWS
    bk, stages = config.block_k, config.stages
    if m % tm or n % tn or k % bk:
        raise ValueError(f"({m}, {n}, {k}) must be tiled by ({tm}, {tn}, {bk})")
    if bk % SF_TILE_K:
        raise ValueError(f"block_k={bk} must be a multiple of {SF_TILE_K}")
    num_kb = k // bk
    sf_tiles = bk // SF_TILE_K  # scale tiles per K block
    k_scales = bk // SF_VEC_SIZE  # TMEM scale columns per K block
    ab_transforms = _fp4_transforms(bk)

    def body(a_gmem, asf_gmem, b_gmem, bsf_gmem, alpha_gmem, mask_gmem, out_gmem):
        e = lax.axis_index("l")
        mi = lax.axis_index("mi")
        ni = lax.axis_index("ni")
        m_slice = pl.ds(mi * tm, tm)
        n_slice = pl.ds(ni * tn, tn)

        @pl.when(mi * tm < mask_gmem[e])  # experts skip the tiles beyond their rows
        def _():
            @functools.partial(
                pl.run_scoped,
                a_smem=plgpu.SMEM((stages, tm, bk), FP4_DTYPE, transforms=ab_transforms),
                b_smem=plgpu.SMEM((stages, tn, bk), FP4_DTYPE, transforms=ab_transforms),
                # exactly the shape async_copy_scales_to_tmem expects, per stage
                asf_smem=plgpu.SMEM((stages, 1, sf_tiles, 32, 16), SF_DTYPE),
                bsf_smem=plgpu.SMEM((stages, 1, sf_tiles, 32, 16), SF_DTYPE),
                out_smem=plgpu.SMEM((tm, tn), jnp.bfloat16),
                acc_tmem=plgpu.TMEM((tm, tn), jnp.float32),
                asf_tmem=plgpu.TMEM((TMEM_ROWS, k_scales), SF_DTYPE,
                                    layout=plgpu.TMEMLayout.SCALES_LAYOUT),
                bsf_tmem=plgpu.TMEM((TMEM_ROWS, k_scales), SF_DTYPE,
                                    layout=plgpu.TMEMLayout.SCALES_LAYOUT),
                ab_barrier=plgpu.Barrier(num_arrivals=4, num_barriers=stages),
                consumed=plgpu.Barrier(num_barriers=stages, orders_tensor_core=True),
                mma_done=plgpu.Barrier(orders_tensor_core=True),
            )
            def compute(a_smem, b_smem, asf_smem, bsf_smem, out_smem, acc_tmem,
                        asf_tmem, bsf_tmem, ab_barrier, consumed, mma_done):
                def fetch(kb, slot):
                    k_slice = pl.ds(kb * bk, bk)
                    sf_slice = pl.ds(kb * sf_tiles, sf_tiles)
                    plgpu.copy_gmem_to_smem(a_gmem.at[e, m_slice, k_slice], a_smem.at[slot],
                                            ab_barrier.at[slot])
                    plgpu.copy_gmem_to_smem(b_gmem.at[e, n_slice, k_slice], b_smem.at[slot],
                                            ab_barrier.at[slot])
                    plgpu.copy_gmem_to_smem(asf_gmem.at[e, mi, sf_slice], asf_smem.at[slot, 0],
                                            ab_barrier.at[slot])
                    plgpu.copy_gmem_to_smem(bsf_gmem.at[e, ni, sf_slice], bsf_smem.at[slot, 0],
                                            ab_barrier.at[slot])

                for slot in range(stages):  # prologue
                    @pl.when(slot < num_kb)
                    def _(slot=slot):
                        fetch(slot, slot)

                def k_block(kb, _):
                    slot = lax.rem(kb, stages)
                    plgpu.barrier_wait(ab_barrier.at[slot])
                    # ASSUMPTION: a single scale buffer per operand is safe to reuse
                    # across K blocks. The copy and the MMA are both issued from this
                    # thread into the tensor core's queue, and that queue is ordered, so
                    # block kb+1's copy cannot overtake block kb's MMA. If a real run
                    # shows corruption here, give these a `stages` leading dimension.
                    plgpu.async_copy_scales_to_tmem(asf_smem.at[slot], asf_tmem)
                    plgpu.async_copy_scales_to_tmem(bsf_smem.at[slot], bsf_tmem)
                    plgpu.tcgen05_mma(
                        acc_tmem,
                        a_smem.at[slot],
                        # logical (bk, tn); physically K-minor, so no data movement
                        plgpu.transpose_ref(b_smem.at[slot], (1, 0)),
                        consumed.at[slot],
                        a_scale=asf_tmem,
                        b_scale=bsf_tmem,
                        accumulate=kb > 0,  # first block overwrites, rest accumulate
                    )

                    @pl.when(kb + stages < num_kb)
                    def _():
                        plgpu.barrier_wait(consumed.at[slot])  # operands read; slot free
                        fetch(kb + stages, slot)

                    return 0

                lax.fori_loop(0, num_kb, k_block, 0)

                plgpu.tcgen05_commit_arrive(mma_done)
                plgpu.barrier_wait(mma_done)
                acc = plgpu.async_load_tmem(acc_tmem)
                plgpu.wait_load_tmem()
                out_smem[...] = (acc * alpha_gmem[e]).astype(jnp.bfloat16)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[e, m_slice, n_slice])
                plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((l, m, n), jnp.bfloat16),
        grid=(l, m // tm, n // tn),
        grid_names=("l", "mi", "ni"),
    )(a_q, a_sf, b_q, b_sf, alpha, masked_m)
