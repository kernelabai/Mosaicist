"""Masked, block-scaled grouped GEMM for Hopper in Pallas Mosaic GPU.

The Hopper analog of FlashInfer's `grouped_gemm_nt_masked` (see quant.py for the
mapping). Per expert l:

    out[l, m, n] = alpha[l] * sum_kb  sfa[l, kb, m] * sfb[l, kb, n]
                                     * sum_{k in kb} a[l, m, k] * b[l, k, n]

with fp8 (e4m3) operands, e4m3 block scales, and only the first `masked_m[l]` rows
computed. `alpha` is expected to already carry the two global scales (see moe.py).

Layout notes (both forced by the hardware, and both documented in the README):
  * scale factors are K-major, (l, k // BLOCK_K, rows), so a tile's scale vector is
    contiguous -- TMA cannot load the 1-byte inner dimension of the natural layout;
  * B keeps the reference's NT layout (l, n, k). fp8 wgmma cannot transpose its
    operands (only 16-bit types can), so both operands must be K-minor in smem: the B
    tile is stored (bn, BLOCK_K) and handed to wgmma as a transposed *view*, which
    leaves the physical layout K-minor.

Hopper has no block-scaled MMA, so the scales are applied at accumulator granularity:
each K block is accumulated by wgmma in fp32, read out, scaled by the outer product of
the block's two scale vectors, and added to a register total. That readout is the
expensive part -- reading an accumulator drains every wgmma in flight -- and two things
here hide it:

  * the mainloop runs two accumulators and two K blocks at a time, draining the first
    while the second is still executing on the tensor cores;
  * `stages` is kept small enough that two blocks fit on an SM at once, so one block's
    rescale (CUDA cores) overlaps another's wgmma (tensor cores).

That second one is worth explaining, because the obvious way to get the same overlap --
a second consumer warpgroup splitting the tile's N -- was tried and is slower. It is
still here behind `consumers`, and it is correct, but it loses twice over:

  * two warpgroups sharing one ring of smem buffers have to hand off through a
    `consumed` barrier before any stage can be refetched, and that handshake pulls them
    back into lockstep -- exactly the interleaving it was meant to create. Releasing
    per K block costs 251 TFLOP/s against 318 for one warpgroup; batching the releases
    once per iteration starves the pipeline instead and drops to 135.
  * it doubles the threads per block at an unchanged 255 registers each, so only one
    block fits per SM. One warpgroup at 230 registers and 92 KB of smem fits two, which
    is where the overlap actually comes from -- and that is 376 TFLOP/s.

Measured on an H100 PCIe at l=8, m=512, k=n=2048 (gemm1's shape):

    consumers=1  stages=3  92 KB   2 blocks/SM    376 TFLOP/s   <- default
    consumers=1  stages=8  215 KB  1 block/SM     318
    consumers=2  stages=4  166 KB  1 block/SM     251
    consumers=2  stages=2  100 KB  1 block/SM     237  (register-bound, not smem-bound)

The lesson generalises past this kernel: on Hopper, "add a warpgroup" and "fit another
block" are two ways to buy the same overlap, and the second is free if the register and
smem budget allows it.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from quant import BLOCK_K, FP8_DTYPE

SF_ALIGN = 128
"""Bytes per stage in the scale-factor smem buffers.

TMA faults with a misaligned address if a destination slice starts on a 64-byte
boundary, which is what a (stages, 64) e4m3 buffer gives for odd slots. Padding each
stage's slice to 128 bytes keeps every destination 128-byte aligned, and is what makes
per-warpgroup N tiles below 128 usable at all."""


SMEM_PER_SM = 227 * 1024
"""H100 shared memory per SM. Half of it is the number that matters: a block using more
than `SMEM_PER_SM // 2` is alone on its SM, and the mainloop's rescale then has no other
block's wgmma to hide behind."""


@dataclasses.dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 64  # per warpgroup; a 128x64 fp32 accumulator is 64 registers/thread,
    #                   leaving room for the second one the two-block mainloop needs
    stages: int = 3  # small on purpose: 92 KB, so two blocks fit per SM (see above)
    consumers: int = 1  # warpgroups splitting the tile's N; >1 is correct but slower
    smem_limit: int = SMEM_PER_SM


def _fp8_transforms(minor_elems: int):
    swizzle = plgpu.find_swizzle(min(minor_elems, 128) * 8)  # fp8: 1 byte per element
    return (plgpu.TilingTransform((8, swizzle)), plgpu.SwizzleTransform(swizzle))


def masked_grouped_gemm(
    a_q: jax.Array,  # (l, m, k) e4m3
    a_sf: jax.Array,  # (l, k // BLOCK_K, m) e4m3
    b_q: jax.Array,  # (l, n, k) e4m3  (the reference's NT layout)
    b_sf: jax.Array,  # (l, k // BLOCK_K, n) e4m3
    alpha: jax.Array,  # (l,) f32
    masked_m: jax.Array,  # (l,) int32
    config: GemmConfig = GemmConfig(),
) -> jax.Array:
    l, m, k = a_q.shape
    _, n, _ = b_q.shape
    tm, tn, stages = config.tile_m, config.tile_n, config.stages
    nwg = config.consumers
    bn = tn * nwg  # the block's N tile, split one slice per consumer warpgroup
    if m % tm or n % bn or k % BLOCK_K:
        raise ValueError(f"({m}, {n}, {k}) must be tiled by ({tm}, {bn}, {BLOCK_K})")
    num_kb = k // BLOCK_K
    ab_transforms = _fp8_transforms(BLOCK_K)  # both operands are K-minor
    msf, nsf = max(tm, SF_ALIGN), max(bn, SF_ALIGN)

    smem = stages * (tm * BLOCK_K + bn * BLOCK_K + msf + nsf) + tm * bn * 2
    if smem > config.smem_limit:
        raise ValueError(f"{smem} bytes of smem exceeds {config.smem_limit}; lower `stages`")

    def body(a_gmem, asf_gmem, b_gmem, bsf_gmem, alpha_gmem, mask_gmem, out_gmem,
             a_smem, b_smem, asf_smem, bsf_smem, out_smem, ab_barrier, consumed):
        e = lax.axis_index("l")
        mi = lax.axis_index("mi")
        ni = lax.axis_index("ni")
        wg = lax.axis_index("wg")
        m_slice = pl.ds(mi * tm, tm)
        block_n = pl.ds(ni * bn, bn)  # the whole block's columns, loaded once
        my_n = pl.ds(wg * tn, tn)  # this warpgroup's slice of them

        @pl.when(mi * tm < mask_gmem[e])  # experts skip the tiles beyond their rows
        def _():
            @functools.partial(
                pl.run_scoped,  # registers, so these are per warpgroup
                acc0=plgpu.ACC((tm, tn), jnp.float32),
                acc1=plgpu.ACC((tm, tn), jnp.float32),
            )
            def compute(acc0, acc1):
                def fetch(kb, slot):
                    """Issued by warpgroup 0 only -- one set of tiles serves them all."""
                    k_slice = pl.ds(kb * BLOCK_K, BLOCK_K)
                    plgpu.copy_gmem_to_smem(a_gmem.at[e, m_slice, k_slice], a_smem.at[slot],
                                            ab_barrier.at[slot])
                    plgpu.copy_gmem_to_smem(b_gmem.at[e, block_n, k_slice], b_smem.at[slot],
                                            ab_barrier.at[slot])
                    plgpu.copy_gmem_to_smem(asf_gmem.at[e, kb, m_slice], asf_smem.at[slot, :tm],
                                            ab_barrier.at[slot])
                    plgpu.copy_gmem_to_smem(bsf_gmem.at[e, kb, block_n], bsf_smem.at[slot, :bn],
                                            ab_barrier.at[slot])

                def issue(slot, acc):
                    plgpu.barrier_wait(ab_barrier.at[slot])
                    plgpu.wgmma(acc, a_smem.at[slot],
                                # this warpgroup's N columns; logically (BLOCK_K, tn),
                                # physically K-minor, so the transpose moves no data
                                plgpu.transpose_ref(b_smem.at[slot, my_n], (1, 0)))

                def drain(acc, slot, total):
                    """Fold one finished K block into the running total and reset the
                    accumulator. The caller has already waited for this accumulator's
                    wgmma; `wgmma_accumulator_load` rather than `acc[...]` keeps that
                    wait under our control, so a later block can still be in flight."""
                    partial = plgpu.wgmma_accumulator_load(acc)
                    acc[...] = jnp.zeros_like(partial)
                    asf = asf_smem[slot, :tm].astype(jnp.float32)  # (tm,)
                    bsf = bsf_smem[slot, my_n].astype(jnp.float32)  # (tn,)
                    return total + partial * asf[:, None] * bsf[None, :]

                def release(slot, kb):
                    """Report this stage consumed, and refetch it once everyone has.

                    The arrival sits inside the same predicate as the wait so the two
                    stay balanced. A stray arrival would flip the barrier's phase, and
                    these barriers outlive the block -- the next grid step would then
                    wait on the wrong phase. The predicates are nested rather than
                    and-ed together so that a statically-false one skips tracing its
                    body entirely; combining them makes the condition traced, and the
                    body's out-of-range `kb + stages` slice is then built anyway.
                    """
                    @pl.when(kb + stages < num_kb)
                    def _():
                        if nwg == 1:  # nobody else to wait for; skip the barrier entirely
                            fetch(kb + stages, slot)
                            return
                        plgpu.barrier_arrive(consumed.at[slot])

                        @pl.when(wg == 0)
                        def _():
                            plgpu.barrier_wait(consumed.at[slot])  # all consumers done
                            fetch(kb + stages, slot)

                @pl.when(wg == 0)
                def _():
                    for slot in range(stages):  # prologue
                        @pl.when(slot < num_kb)
                        def _(slot=slot):
                            fetch(slot, slot)

                def k_pair(j, total):
                    """Two K blocks at a time, so the first one's rescale overlaps the
                    second one's wgmma instead of stalling the tensor cores."""
                    kb0, kb1 = 2 * j, 2 * j + 1
                    s0, s1 = lax.rem(kb0, stages), lax.rem(kb1, stages)
                    issue(s0, acc0)
                    issue(s1, acc1)

                    plgpu.wgmma_wait(1)  # kb0 done, kb1 still running
                    total = drain(acc0, s0, total)
                    release(s0, kb0)

                    plgpu.wgmma_wait(0)
                    total = drain(acc1, s1, total)
                    release(s1, kb1)
                    return total

                total = lax.fori_loop(0, num_kb // 2, k_pair, jnp.zeros((tm, tn), jnp.float32))
                if num_kb % 2:  # odd K-block count: one unpaired block, drained alone
                    last = num_kb - 1
                    slot = lax.rem(last, stages)
                    issue(slot, acc0)
                    plgpu.wgmma_wait(0)
                    total = drain(acc0, slot, total)
                    # no release: this is the last block, so nothing refetches the stage

                out_smem[wg] = (total * alpha_gmem[e]).astype(jnp.bfloat16)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(out_smem.at[wg],
                                        out_gmem.at[e, m_slice, pl.ds(ni * bn + wg * tn, tn)])
                plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((l, m, n), jnp.bfloat16),
        grid=(l, m // tm, n // bn),
        grid_names=("l", "mi", "ni"),
        num_threads=nwg,
        thread_name="wg",
        # block-shared: one copy per CTA, not per warpgroup
        scratch_shapes=[
            plgpu.SMEM((stages, tm, BLOCK_K), FP8_DTYPE, transforms=ab_transforms),
            plgpu.SMEM((stages, bn, BLOCK_K), FP8_DTYPE, transforms=ab_transforms),
            plgpu.SMEM((stages, msf), FP8_DTYPE),
            plgpu.SMEM((stages, nsf), FP8_DTYPE),
            plgpu.SMEM((nwg, tm, tn), jnp.bfloat16),
            plgpu.Barrier(num_arrivals=4, num_barriers=stages),
            plgpu.Barrier(num_arrivals=nwg, num_barriers=stages),
        ],
    )(a_q, a_sf, b_q, b_sf, alpha, masked_m)
