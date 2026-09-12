"""Warp-specialized variant of the Blackwell masked block-scaled GEMM.

Same maths as `masked_gemm.py`; the difference is who does what. There, one warpgroup
issues the TMAs, the MMA and the epilogue in sequence, so the TMA for block kb+1 cannot
be issued until the MMA for kb has been. Here the work is split the way FlashInfer's
kernel splits it, following JAX's own `blackwell_matmul_mgpu.py`:

    warpgroup 0, warp 0   issues every TMA, running ahead of the MMA and gated only by
                          the `consumed` barrier the MMA raises when it frees a stage
    warpgroup 0, warp 1   issues the scale copies and the MMA
    warpgroup 1           waits for the MMA, drains TMEM and stores the tile

The split only pays if there is a *next* tile for the store warpgroup to overlap with,
so the kernel is also persistent: `dynamic_scheduling_loop` walks work items and the
accumulator is double-buffered in TMEM, letting warpgroup 1 drain tile i while
warpgroup 0 starts tile i+1. Measured without persistence, this same split is slower
than doing everything in one warpgroup.

Masked tiles make the barrier protocol delicate: a tile whose rows all sit past
`masked_m` does no work, but it must still take part in every arrive/wait or the pairing
across the double-buffered accumulator desyncs and the kernel hangs. Every iteration
therefore arrives on `mma_done`, `store_done` and `tile_done` whether or not it computes
anything; only the actual loads, MMAs and stores are skipped.

Shared memory is declared through `scratch_shapes` rather than `pl.run_scoped` so that
one copy is shared by both warpgroups rather than allocated per warpgroup.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from masked_gemm import SF_TILE_K, SMEM_PER_SM, TMEM_ROWS, GemmConfig, _fp4_transforms, _resolve
from nvfp4 import FP4_DTYPE, SF_DTYPE, SF_VEC_SIZE

TMA_WARP, MMA_WARP = 0, 1
COMPUTE_WG, STORE_WG = 0, 1


def masked_grouped_gemm_ws(
    a_q: jax.Array,  # (l, m, k) e2m1
    a_sf: jax.Array,  # (l, m // 128, k // 64, 32, 16) e4m3, MMA scale layout
    b_q: jax.Array,  # (l, n, k) e2m1
    b_sf: jax.Array,  # (l, n // 128, k // 64, 32, 16) e4m3
    alpha: jax.Array,  # (l,) f32
    masked_m: jax.Array,  # (l,) int32
    config: GemmConfig = GemmConfig(),
) -> jax.Array:
    l, m, k = a_q.shape
    _, n, _ = b_q.shape
    tm, tn = TMEM_ROWS, config.tile_n
    if config.collective:
        raise NotImplementedError("collective + warp specialization not wired up yet")
    bk, stages = _resolve(config, k, tm, tn)
    if m % tm or n % tn or k % bk:
        raise ValueError(f"({m}, {n}, {k}) must be tiled by ({tm}, {tn}, {bk})")
    num_kb = k // bk
    sf_tiles = bk // SF_TILE_K
    k_scales = bk // SF_VEC_SIZE
    n_tiles = tn // TMEM_ROWS
    ab_transforms = _fp4_transforms(bk)

    smem = stages * (tm * bk // 2 + tn * bk // 2 + 2 * sf_tiles * 512) + tm * tn * 2
    if smem > SMEM_PER_SM:
        raise ValueError(f"{smem} bytes of smem exceeds {SMEM_PER_SM}")

    def body(a_gmem, asf_gmem, b_gmem, bsf_gmem, alpha_gmem, mask_gmem, out_gmem,
             a_smem, b_smem, asf_smem, bsf_smem, out_smem,
             acc_tmem, asf_tmem, bsf_tmem, ab_barrier, consumed, mma_done,
             store_done):
        wg = lax.axis_index("wg")

        @plgpu.dynamic_scheduling_loop(grid_names=("l", "mi", "ni"), thread_axis="wg")
        def _tile(info):
            e, mi, ni = info.index
            li = info.local_index
            acc_slot = lax.rem(li, 2)
            acc = acc_tmem.at[:, pl.ds(acc_slot * tn, tn)]
            m_slice = pl.ds(mi * tm, tm)
            n_slice = pl.ds(ni * tn, tn)
            active = mi * tm < mask_gmem[e]  # experts skip tiles beyond their rows

            @pl.when(wg == COMPUTE_WG)
            def _():
                @plgpu.warp_map
                def _per_warp(warp_id):
                    @pl.when(warp_id == TMA_WARP)
                    def _memory():
                        @pl.when(active)
                        def _():
                            def _fetch(kb, _):
                                slot = lax.rem(kb, stages)
                                k_slice = pl.ds(kb * bk, bk)
                                sf_slice = pl.ds(kb * sf_tiles, sf_tiles)

                                @pl.when(kb >= stages)
                                def _():
                                    plgpu.barrier_wait(consumed.at[slot])

                                plgpu.copy_gmem_to_smem(a_gmem.at[e, m_slice, k_slice],
                                                        a_smem.at[slot], ab_barrier.at[slot])
                                plgpu.copy_gmem_to_smem(b_gmem.at[e, n_slice, k_slice],
                                                        b_smem.at[slot], ab_barrier.at[slot])
                                plgpu.copy_gmem_to_smem(asf_gmem.at[e, mi, sf_slice],
                                                        asf_smem.at[slot, 0],
                                                        ab_barrier.at[slot])
                                plgpu.copy_gmem_to_smem(
                                    bsf_gmem.at[e, pl.ds(ni * n_tiles, n_tiles), sf_slice],
                                    bsf_smem.at[slot], ab_barrier.at[slot])
                                return 0

                            lax.fori_loop(0, num_kb, _fetch, 0)

                            # Drain the stages the mainloop never waited on. The guard
                            # above only waits for a stage that a *later* fetch reuses,
                            # so the last `stages` K blocks each leave one arrival
                            # standing. In a persistent kernel those accumulate tile
                            # after tile until the barrier's phase runs ahead of its
                            # waiters and the kernel faults -- which is why this only
                            # breaks once num_kb exceeds stages. Draining here leaves
                            # every tile balanced, which also makes a skipped tile safe
                            # and means the next tile cannot overwrite a buffer the MMA
                            # is still reading.
                            for i in range(min(stages, num_kb)):
                                plgpu.barrier_wait(
                                    consumed.at[(num_kb - min(stages, num_kb) + i) % stages])

                    @pl.when(warp_id == MMA_WARP)
                    def _compute():
                        # do not start filling an accumulator the store warpgroup has
                        # not finished draining
                        @pl.when(li > 1)
                        def _():
                            plgpu.barrier_wait(store_done.at[acc_slot])

                        @pl.when(active)
                        def _():
                            def _mma(kb, _):
                                slot = lax.rem(kb, stages)
                                plgpu.barrier_wait(ab_barrier.at[slot])
                                # issued from this thread, so ordered ahead of the MMA
                                plgpu.async_copy_scales_to_tmem(asf_smem.at[slot], asf_tmem)
                                plgpu.async_copy_scales_to_tmem(bsf_smem.at[slot], bsf_tmem)
                                plgpu.tcgen05_mma(
                                    acc,
                                    a_smem.at[slot],
                                    plgpu.transpose_ref(b_smem.at[slot], (1, 0)),
                                    consumed.at[slot],
                                    a_scale=asf_tmem,
                                    b_scale=bsf_tmem,
                                    accumulate=kb > 0,
                                )
                                return 0

                            lax.fori_loop(0, num_kb, _mma, 0)

                        # arrive unconditionally: a skipped tile still has to keep the
                        # double-buffer protocol paired
                        plgpu.tcgen05_commit_arrive(mma_done.at[acc_slot])

            @pl.when(wg == STORE_WG)
            def _():
                plgpu.barrier_wait(mma_done.at[acc_slot])

                @pl.when(active)
                def _():
                    vals = plgpu.async_load_tmem(acc)
                    plgpu.wait_load_tmem()
                    out_smem[...] = (vals * alpha_gmem[e]).astype(jnp.bfloat16)
                    plgpu.commit_smem()
                    plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[e, m_slice, n_slice])
                    plgpu.wait_smem_to_gmem(0)

                plgpu.barrier_arrive(store_done.at[acc_slot])

    return plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((l, m, n), jnp.bfloat16),
        grid=(l, m // tm, n // tn),
        grid_names=("l", "mi", "ni"),
        num_threads=2,
        thread_name="wg",
        scratch_shapes=[
            plgpu.SMEM((stages, tm, bk), FP4_DTYPE, transforms=ab_transforms),
            plgpu.SMEM((stages, tn, bk), FP4_DTYPE, transforms=ab_transforms),
            plgpu.SMEM((stages, 1, sf_tiles, 32, 16), SF_DTYPE),
            plgpu.SMEM((stages, n_tiles, sf_tiles, 32, 16), SF_DTYPE),
            plgpu.SMEM((tm, tn), jnp.bfloat16),
            plgpu.TMEM((tm, tn * 2), jnp.float32),  # double buffered across tiles
            plgpu.TMEM((TMEM_ROWS, k_scales), SF_DTYPE,
                       layout=plgpu.TMEMLayout.SCALES_LAYOUT),
            plgpu.TMEM((tn, k_scales), SF_DTYPE,
                       layout=plgpu.TMEMLayout.SCALES_LAYOUT),
            plgpu.Barrier(num_arrivals=4, num_barriers=stages),
            plgpu.Barrier(num_barriers=stages, orders_tensor_core=True),
            plgpu.Barrier(num_barriers=2, orders_tensor_core=True),  # mma_done
            plgpu.Barrier(num_barriers=2, orders_tensor_core=True),  # store_done
        ],
    )(a_q, a_sf, b_q, b_sf, alpha, masked_m)
