"""GEMM1 fused with silu_and_mul and the NVFP4 quantize, for Blackwell.

Replaces two launches -- `masked_grouped_gemm` producing an (l, m, 2n) bf16 tensor and
`silu_mul_quantize_nvfp4_pallas` reading it straight back -- with one that never writes
the intermediate. At the benchmark shape that round trip is 16.8 MB out and 16.8 MB in,
and ablating every arithmetic operation out of the standalone activation kernel still
leaves 11.9 us of it.

Two shapes are forced, and they are what makes this kernel look the way it does:

  * **The activation tile is 256 columns wide, not 128.** Its e4m3 block scales are one
    byte per 16 elements, so a 128-wide tile stores 8 of them per row -- 64 bits, and TMA
    requires at least 128 along the last dimension. 256 columns give 16.
  * **Those 256 columns are computed in two passes of 128.** silu needs `gate[j]` and
    `up[j]`, which live `n` apart in w1's output, so a 128-wide activation tile already
    needs two accumulators. Four of them (two passes at once) would be 512 TMEM columns,
    the entire file, leaving nothing for the MMA's own scale operands. The cost of two
    passes is that the A tile is fetched twice; it is still fetched fewer times overall
    than the unfused GEMM1 fetches it, because each pass covers a gate and an up tile.

The accumulator is rounded to bf16 before the activation, which is not an accident:
the unfused path writes a bf16 gateup tensor, and skipping that rounding here would make
this kernel disagree with `nvfp4.moe_reference` for no good reason.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from masked_gemm import SF_TILE_K, SMEM_PER_SM, TMEM_ROWS, _fp4_transforms
from nvfp4 import FP4_DTYPE, FP4_MAX, SF_DTYPE, SF_MAX, SF_VEC_SIZE
from quantize_kernels import APPROX_MATH, LOG2E, SUB_K

TILE_N = 256  # activation columns per block; see the docstring
HALF = TILE_N // 2  # one MMA pass
TMA_WARP, MMA_WARP = 0, 1


@dataclasses.dataclass(frozen=True)
class FusedConfig:
    block_k: int = 128
    stages: int = 2


def _quantize_half(vals, gs, q_smem, pass_idx, sf_full, nb_total):
    """Quantize one 128-column half into `q_smem[pass_idx]`, accumulating its scales.

    Same construction as `quantize_kernels._quantize_tile` -- sub-tiles of SUB_K columns
    so the one-hot expansion stays cheap, and each sub-tile stored whole because a column
    slice of the swizzled fp4 output cannot be written.
    """
    nsub, nb_sub = HALF // SUB_K, SUB_K // SF_VEC_SIZE
    # TCGEN05, not WGMMA: these values came out of TMEM via async_load_tmem, and the
    # WGMMA layout has no inference solution from there (probe_tmem_quant.py)
    vals = plgpu.layout_cast(vals, plgpu.Layout.TCGEN05)
    for s in range(nsub):
        sub = vals[:, s * SUB_K:(s + 1) * SUB_K]
        inv_sub = jnp.zeros((TMEM_ROWS, SUB_K), jnp.float32)
        for j in range(nb_sub):
            block = sub[:, j * SF_VEC_SIZE:(j + 1) * SF_VEC_SIZE]
            amax = plgpu.layout_cast(jnp.max(jnp.abs(block), axis=-1),
                                     plgpu.Layout.TCGEN05.reduce(1))
            sf = jnp.clip(amax / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE).astype(jnp.float32)
            step = sf / gs
            inv = jnp.where(step > 0.0, 1.0, 0.0) / jnp.where(step == 0.0, 1.0, step)
            inv_sub += inv[:, None] * (
                jnp.arange(SUB_K) // SF_VEC_SIZE == j).astype(jnp.float32)[None, :]
            # this half owns scale columns [pass_idx * 8, +8) of the 256-wide tile
            col = pass_idx * (HALF // SF_VEC_SIZE) + s * nb_sub + j
            sf_full += sf[:, None] * (
                jnp.arange(nb_total) == col).astype(jnp.float32)[None, :]
        q_smem[pass_idx, s] = jnp.clip(sub * inv_sub, -FP4_MAX, FP4_MAX).astype(FP4_DTYPE)
    return sf_full


def gemm1_silu_quantize(
    a_q: jax.Array,  # (l, m, k) e2m1
    a_sf: jax.Array,  # (l, m // 128, k // 64, 32, 16) e4m3, MMA scale layout
    w1_q: jax.Array,  # (l, 2n, k) e2m1
    w1_sf: jax.Array,  # (l, 2n // 128, k // 64, 32, 16) e4m3
    alpha: jax.Array,  # (l,) f32 -- GEMM1's alpha, global scales already folded in
    a2_global_scale: jax.Array,  # (l,) f32
    masked_m: jax.Array,  # (l,) int32
    config: FusedConfig = FusedConfig(),
):
    """-> ((l, m, n) e2m1, (l, m, n // 16) e4m3), the same pair the unfused pair returns."""
    l, m, k = a_q.shape
    two_n = w1_q.shape[1]
    n = two_n // 2
    tm, bk, stages = TMEM_ROWS, config.block_k, config.stages
    if m % tm or n % TILE_N or k % bk or bk % SF_TILE_K:
        raise ValueError(f"({m}, {n}, {k}) must be tiled by ({tm}, {TILE_N}, {bk})")
    num_kb = k // bk
    sf_tiles, k_scales = bk // SF_TILE_K, bk // SF_VEC_SIZE
    nb_total = TILE_N // SF_VEC_SIZE  # 16 scale columns -> a legal TMA store
    nsub = HALF // SUB_K
    ab_transforms = _fp4_transforms(bk)

    smem = (stages * (3 * tm * bk // 2 + 3 * sf_tiles * 512)
            + 2 * nsub * tm * SUB_K // 2 + tm * nb_total)
    if smem > SMEM_PER_SM:
        raise ValueError(f"{smem} bytes of smem exceeds {SMEM_PER_SM}")

    def body(a_gmem, asf_gmem, w_gmem, wsf_gmem, alpha_gmem, a2gs_gmem, mask_gmem,
             q_gmem, sf_gmem,
             a_smem, bg_smem, bu_smem, asf_smem, bgsf_smem, busf_smem, q_smem, sf_smem,
             acc_g, acc_u, asf_tmem, bgsf_tmem, busf_tmem, ab_barrier, consumed, mma_done):
        e, mi, ni = (lax.axis_index(x) for x in ("l", "mi", "ni"))
        m_slice = pl.ds(mi * tm, tm)

        @pl.when(mi * tm < mask_gmem[e])
        def _():
            sf_full = jnp.zeros((tm, nb_total), jnp.float32)
            for p in range(2):  # two 128-column passes; see the module docstring
                c0 = ni * TILE_N + p * HALF
                gate_rows, up_rows = pl.ds(c0, HALF), pl.ds(n + c0, HALF)
                # scale slabs are indexed in 128-row tiles of w1's 2n rows
                g_tile, u_tile = (ni * 2 + p), (n // TMEM_ROWS + ni * 2 + p)

                @plgpu.warp_map
                def _per_warp(warp_id, gate_rows=gate_rows, up_rows=up_rows,
                              g_tile=g_tile, u_tile=u_tile):
                    @pl.when(warp_id == TMA_WARP)
                    def _memory():
                        def _fetch(kb, _):
                            slot = lax.rem(kb, stages)
                            ks = pl.ds(kb * bk, bk)
                            sfs = pl.ds(kb * sf_tiles, sf_tiles)

                            @pl.when(kb >= stages)
                            def _():
                                plgpu.barrier_wait(consumed.at[slot])

                            plgpu.copy_gmem_to_smem(a_gmem.at[e, m_slice, ks],
                                                    a_smem.at[slot], ab_barrier.at[slot])
                            plgpu.copy_gmem_to_smem(w_gmem.at[e, gate_rows, ks],
                                                    bg_smem.at[slot], ab_barrier.at[slot])
                            plgpu.copy_gmem_to_smem(w_gmem.at[e, up_rows, ks],
                                                    bu_smem.at[slot], ab_barrier.at[slot])
                            plgpu.copy_gmem_to_smem(asf_gmem.at[e, mi, sfs],
                                                    asf_smem.at[slot, 0], ab_barrier.at[slot])
                            plgpu.copy_gmem_to_smem(wsf_gmem.at[e, g_tile, sfs],
                                                    bgsf_smem.at[slot, 0], ab_barrier.at[slot])
                            plgpu.copy_gmem_to_smem(wsf_gmem.at[e, u_tile, sfs],
                                                    busf_smem.at[slot, 0], ab_barrier.at[slot])
                            return 0

                        lax.fori_loop(0, num_kb, _fetch, 0)
                        for i in range(min(stages, num_kb)):  # leave the pipeline balanced
                            plgpu.barrier_wait(
                                consumed.at[(num_kb - min(stages, num_kb) + i) % stages])

                    @pl.when(warp_id == MMA_WARP)
                    def _compute():
                        def _mma(kb, _):
                            slot = lax.rem(kb, stages)
                            plgpu.barrier_wait(ab_barrier.at[slot])
                            plgpu.async_copy_scales_to_tmem(asf_smem.at[slot], asf_tmem)
                            plgpu.async_copy_scales_to_tmem(bgsf_smem.at[slot], bgsf_tmem)
                            plgpu.async_copy_scales_to_tmem(busf_smem.at[slot], busf_tmem)
                            plgpu.tcgen05_mma(
                                acc_g, a_smem.at[slot],
                                plgpu.transpose_ref(bg_smem.at[slot], (1, 0)),
                                a_scale=asf_tmem, b_scale=bgsf_tmem, accumulate=kb > 0)
                            plgpu.tcgen05_mma(
                                acc_u, a_smem.at[slot],
                                plgpu.transpose_ref(bu_smem.at[slot], (1, 0)),
                                consumed.at[slot],
                                a_scale=asf_tmem, b_scale=busf_tmem, accumulate=kb > 0)
                            return 0

                        lax.fori_loop(0, num_kb, _mma, 0)
                        plgpu.tcgen05_commit_arrive(mma_done)

                plgpu.barrier_wait(mma_done)
                al = alpha_gmem[e]
                # bf16 first: the unfused path stores a bf16 gateup, and this has to agree
                gate = (plgpu.async_load_tmem(acc_g) * al).astype(jnp.bfloat16).astype(jnp.float32)
                up = (plgpu.async_load_tmem(acc_u) * al).astype(jnp.bfloat16).astype(jnp.float32)
                plgpu.wait_load_tmem()
                act = (gate / (1.0 + jnp.exp2(-gate * LOG2E))) * up
                sf_full = _quantize_half(act, a2gs_gmem[e], q_smem, p, sf_full, nb_total)

            sf_smem[...] = sf_full.astype(SF_DTYPE)
            plgpu.commit_smem()
            for p in range(2):
                for sub in range(nsub):
                    plgpu.copy_smem_to_gmem(
                        q_smem.at[p, sub],
                        q_gmem.at[e, m_slice,
                                  pl.ds(ni * TILE_N + p * HALF + sub * SUB_K, SUB_K)])
            # 16 e4m3 columns = 128 bits, the narrowest store TMA accepts
            plgpu.copy_smem_to_gmem(
                sf_smem, sf_gmem.at[e, m_slice, pl.ds(ni * nb_total, nb_total)])
            plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        body,
        out_shape=(jax.ShapeDtypeStruct((l, m, n), FP4_DTYPE),
                   jax.ShapeDtypeStruct((l, m, n // SF_VEC_SIZE), SF_DTYPE)),
        grid=(l, m // tm, n // TILE_N),
        grid_names=("l", "mi", "ni"),
        compiler_params=plgpu.CompilerParams(approx_math=APPROX_MATH),
        scratch_shapes=[
            plgpu.SMEM((stages, tm, bk), FP4_DTYPE, transforms=ab_transforms),
            plgpu.SMEM((stages, HALF, bk), FP4_DTYPE, transforms=ab_transforms),
            plgpu.SMEM((stages, HALF, bk), FP4_DTYPE, transforms=ab_transforms),
            plgpu.SMEM((stages, 1, sf_tiles, 32, 16), SF_DTYPE),
            plgpu.SMEM((stages, 1, sf_tiles, 32, 16), SF_DTYPE),
            plgpu.SMEM((stages, 1, sf_tiles, 32, 16), SF_DTYPE),
            plgpu.SMEM((2, nsub, tm, SUB_K), FP4_DTYPE),
            plgpu.SMEM((tm, nb_total), SF_DTYPE),
            plgpu.TMEM((tm, HALF), jnp.float32),
            plgpu.TMEM((tm, HALF), jnp.float32),
            plgpu.TMEM((TMEM_ROWS, k_scales), SF_DTYPE,
                       layout=plgpu.TMEMLayout.SCALES_LAYOUT),
            plgpu.TMEM((HALF, k_scales), SF_DTYPE, layout=plgpu.TMEMLayout.SCALES_LAYOUT),
            plgpu.TMEM((HALF, k_scales), SF_DTYPE, layout=plgpu.TMEMLayout.SCALES_LAYOUT),
            plgpu.Barrier(num_arrivals=6, num_barriers=stages),
            plgpu.Barrier(num_barriers=stages, orders_tensor_core=True),
            plgpu.Barrier(orders_tensor_core=True),
        ],
    )(a_q, a_sf, w1_q, w1_sf, alpha, a2_global_scale, masked_m)
