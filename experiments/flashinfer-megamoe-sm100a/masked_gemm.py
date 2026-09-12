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
"""The MMA scale path is defined in terms of 128-row TMEM tiles.

`tile_m` is fixed at 128 by it. `tile_n` may be 128 or 256: `async_copy_scales_to_tmem`
derives the smem scale shape as (mn // 128, k_scales // 4, 32, 16) and rejects more than
two mn-tiles, so 256 is the ceiling."""

SF_TILE_K = 4 * SF_VEC_SIZE  # 64 elements of K per scale tile in the MMA layout

SMEM_PER_SM = 227 * 1024
"""B200 shared memory per SM. Half of it is the number that matters -- see GemmConfig."""


@dataclasses.dataclass(frozen=True)
class GemmConfig:
    """Defaults are the measured best on a B200 at l=8, m=512, k=n=2048 (gemm1's shape).

    Shared memory here buys occupancy, not depth. A block costs
    `stages * 144 * block_k + tile_m * tile_n * 2` bytes, and at more than half the SM's
    232 KB it runs alone, which is what actually decides throughput:

        block_k=512 stages=1   107 KB   2 blocks/SM   1829 TFLOP/s  (needs k % 512)
        block_k=256 stages=2   107 KB   2 blocks/SM   1745          <- default
        block_k=128 stages=4   107 KB   2 blocks/SM   1388
        block_k=128 stages=8   180 KB   1 block/SM     880
        tile_n=256  stages=2   176 KB   1 block/SM    1502

    At equal shared memory a wider K step beats more stages, and both beat a deeper
    pipeline that costs the second resident block. Leaving `block_k` and `stages` as
    None applies exactly that rule to whatever k the call has -- take the widest K step
    k allows, then as many stages as fit in half an SM -- which is why the two GEMMs in
    a MoE layer, whose k differs, do not need to be configured by hand.
    """
    tile_n: int = 128  # 128 or 256; see TMEM_ROWS
    block_k: int | None = None  # None: the largest of 512/256/128 that divides k
    stages: int | None = None  # None: the most that still fit two blocks on an SM


def _resolve(config: GemmConfig, k: int, tm: int, tn: int) -> tuple[int, int]:
    """Fill in `block_k` / `stages`, following the table in GemmConfig.

    Widest K step that divides k, then the deepest pipeline that still leaves the block
    inside half an SM's shared memory so a second one can be resident alongside it.
    """
    bk = config.block_k
    if bk is None:
        bk = next((c for c in (512, 256, 128) if k % c == 0), None)
        if bk is None:
            raise ValueError(f"k={k} is not a multiple of 128; pass block_k explicitly")
    stages = config.stages
    if stages is None:
        per_stage = tm * bk // 2 + tn * bk // 2 + 2 * (bk // SF_TILE_K) * 512
        stages = max(1, (SMEM_PER_SM // 2 - tm * tn * 2) // per_stage)
    return bk, stages


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
    tm, tn = TMEM_ROWS, config.tile_n
    if tn not in (128, 256):
        raise ValueError(f"tile_n must be 128 or 256, got {tn}")
    n_tiles = tn // TMEM_ROWS  # mn-tiles of B scales this tile spans
    bk, stages = _resolve(config, k, tm, tn)
    if m % tm or n % tn or k % bk:
        raise ValueError(f"({m}, {n}, {k}) must be tiled by ({tm}, {tn}, {bk})")
    if bk % SF_TILE_K:
        raise ValueError(f"block_k={bk} must be a multiple of {SF_TILE_K}")
    smem = stages * (tm * bk // 2 + tn * bk // 2 + 2 * (bk // SF_TILE_K) * 512) + tm * tn * 2
    if smem > SMEM_PER_SM:
        raise ValueError(
            f"{smem} bytes of smem exceeds {SMEM_PER_SM}: lower `stages` or `block_k`")
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
                bsf_smem=plgpu.SMEM((stages, n_tiles, sf_tiles, 32, 16), SF_DTYPE),
                out_smem=plgpu.SMEM((tm, tn), jnp.bfloat16),
                acc_tmem=plgpu.TMEM((tm, tn), jnp.float32),
                asf_tmem=plgpu.TMEM((TMEM_ROWS, k_scales), SF_DTYPE,
                                    layout=plgpu.TMEMLayout.SCALES_LAYOUT),
                bsf_tmem=plgpu.TMEM((tn, k_scales), SF_DTYPE,
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
                    plgpu.copy_gmem_to_smem(
                        bsf_gmem.at[e, pl.ds(ni * n_tiles, n_tiles), sf_slice],
                        bsf_smem.at[slot], ab_barrier.at[slot])

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
