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
    tile is stored (tn, BLOCK_K) and handed to wgmma as a transposed *view*, which
    leaves the physical layout K-minor.

Hopper has no block-scaled MMA, so the scales are applied at accumulator granularity:
each K block is accumulated by wgmma in fp32, read out, scaled by the outer product of
the block's two scale vectors, and added to a register total. That readout is the
expensive part -- reading an accumulator drains every wgmma in flight -- so the
mainloop runs two accumulators and two K blocks at a time, draining the first while
the second is still executing on the tensor cores. On an H100 PCIe that is worth about
1.2x over draining every block (259 -> 316 TFLOP/s at l=8, m=512, k=n=2048).
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
tile sizes below 128 usable at all.
"""


@dataclasses.dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 64  # a 128x64 fp32 accumulator is 64 registers/thread, leaving room
    stages: int = 8  # for the second accumulator that the two-block mainloop needs
    smem_limit: int = 227 * 1024  # H100 per-SM shared memory


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
    if m % tm or n % tn or k % BLOCK_K:
        raise ValueError(f"({m}, {n}, {k}) must be tiled by ({tm}, {tn}, {BLOCK_K})")
    num_kb = k // BLOCK_K
    ab_transforms = _fp8_transforms(BLOCK_K)  # both operands are K-minor
    msf, nsf = max(tm, SF_ALIGN), max(tn, SF_ALIGN)

    smem = stages * (tm * BLOCK_K + tn * BLOCK_K + msf + nsf) + tm * tn * 2
    if smem > config.smem_limit:
        raise ValueError(f"{smem} bytes of smem exceeds {config.smem_limit}; lower `stages`")

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
                a_smem=plgpu.SMEM((stages, tm, BLOCK_K), FP8_DTYPE, transforms=ab_transforms),
                b_smem=plgpu.SMEM((stages, tn, BLOCK_K), FP8_DTYPE, transforms=ab_transforms),
                asf_smem=plgpu.SMEM((stages, msf), FP8_DTYPE),
                bsf_smem=plgpu.SMEM((stages, nsf), FP8_DTYPE),
                out_smem=plgpu.SMEM((tm, tn), jnp.bfloat16),
                barrier=plgpu.Barrier(num_arrivals=4, num_barriers=stages),
                acc0=plgpu.ACC((tm, tn), jnp.float32),
                acc1=plgpu.ACC((tm, tn), jnp.float32),
            )
            def compute(a_smem, b_smem, asf_smem, bsf_smem, out_smem, barrier, acc0, acc1):
                def fetch(kb, slot):
                    k_slice = pl.ds(kb * BLOCK_K, BLOCK_K)
                    plgpu.copy_gmem_to_smem(a_gmem.at[e, m_slice, k_slice], a_smem.at[slot], barrier.at[slot])
                    plgpu.copy_gmem_to_smem(b_gmem.at[e, n_slice, k_slice], b_smem.at[slot], barrier.at[slot])
                    plgpu.copy_gmem_to_smem(asf_gmem.at[e, kb, m_slice], asf_smem.at[slot, :tm], barrier.at[slot])
                    plgpu.copy_gmem_to_smem(bsf_gmem.at[e, kb, n_slice], bsf_smem.at[slot, :tn], barrier.at[slot])

                def issue(kb, slot, acc):
                    plgpu.barrier_wait(barrier.at[slot])
                    plgpu.wgmma(acc, a_smem.at[slot], plgpu.transpose_ref(b_smem.at[slot], (1, 0)))

                def drain(acc, slot, total):
                    """Fold one finished K block into the running total and reset the accumulator.

                    The caller has already waited for this accumulator's wgmma; using
                    `wgmma_accumulator_load` rather than `acc[...]` keeps that wait
                    under our control, so a later block can still be in flight.
                    """
                    partial = plgpu.wgmma_accumulator_load(acc)
                    acc[...] = jnp.zeros_like(partial)
                    asf = asf_smem[slot, :tm].astype(jnp.float32)  # (tm,)
                    bsf = bsf_smem[slot, :tn].astype(jnp.float32)  # (tn,)
                    return total + partial * asf[:, None] * bsf[None, :]

                for slot in range(stages):  # prologue
                    @pl.when(slot < num_kb)
                    def _(slot=slot):
                        fetch(slot, slot)

                def k_pair(j, total):
                    """Two K blocks at a time, so the first one's rescale overlaps the
                    second one's wgmma instead of stalling the tensor cores."""
                    kb0, kb1 = 2 * j, 2 * j + 1
                    s0, s1 = lax.rem(kb0, stages), lax.rem(kb1, stages)
                    issue(kb0, s0, acc0)
                    issue(kb1, s1, acc1)

                    plgpu.wgmma_wait(1)  # kb0 done, kb1 still running
                    total = drain(acc0, s0, total)

                    @pl.when(kb0 + stages < num_kb)
                    def _():
                        fetch(kb0 + stages, s0)

                    plgpu.wgmma_wait(0)
                    total = drain(acc1, s1, total)

                    @pl.when(kb1 + stages < num_kb)
                    def _():
                        fetch(kb1 + stages, s1)

                    return total

                total = lax.fori_loop(0, num_kb // 2, k_pair, jnp.zeros((tm, tn), jnp.float32))
                if num_kb % 2:  # odd K-block count: one unpaired block, drained alone
                    last = num_kb - 1
                    slot = lax.rem(last, stages)
                    issue(last, slot, acc0)
                    plgpu.wgmma_wait(0)
                    total = drain(acc0, slot, total)

                out_smem[...] = (total * alpha_gmem[e]).astype(jnp.bfloat16)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[e, m_slice, n_slice])
                plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((l, m, n), jnp.bfloat16),
        grid=(l, m // tm, n // tn),
        grid_names=("l", "mi", "ni"),
    )(a_q, a_sf, b_q, b_sf, alpha, masked_m)
