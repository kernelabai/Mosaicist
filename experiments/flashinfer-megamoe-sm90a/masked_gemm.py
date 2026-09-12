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

Each K block is accumulated by wgmma in fp32, then read out, scaled by the outer
product of the block's two scale vectors, and added to a register accumulator
(Hopper has no block-scaled MMA, so the scaling happens at accumulator granularity).
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


@dataclasses.dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 128
    stages: int = 4  # smem buffers in flight over K blocks


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
                asf_smem=plgpu.SMEM((stages, tm), FP8_DTYPE),
                bsf_smem=plgpu.SMEM((stages, tn), FP8_DTYPE),
                out_smem=plgpu.SMEM((tm, tn), jnp.bfloat16),
                barrier=plgpu.Barrier(num_arrivals=4, num_barriers=stages),
                acc_ref=plgpu.ACC((tm, tn), jnp.float32),
            )
            def compute(a_smem, b_smem, asf_smem, bsf_smem, out_smem, barrier, acc_ref):
                def fetch(kb, slot):
                    k_slice = pl.ds(kb * BLOCK_K, BLOCK_K)
                    plgpu.copy_gmem_to_smem(a_gmem.at[e, m_slice, k_slice], a_smem.at[slot], barrier.at[slot])
                    plgpu.copy_gmem_to_smem(b_gmem.at[e, n_slice, k_slice], b_smem.at[slot], barrier.at[slot])
                    plgpu.copy_gmem_to_smem(asf_gmem.at[e, kb, m_slice], asf_smem.at[slot], barrier.at[slot])
                    plgpu.copy_gmem_to_smem(bsf_gmem.at[e, kb, n_slice], bsf_smem.at[slot], barrier.at[slot])

                for slot in range(stages):  # prologue
                    @pl.when(slot < num_kb)
                    def _(slot=slot):
                        fetch(slot, slot)

                def k_block(kb, total):
                    slot = lax.rem(kb, stages)
                    plgpu.barrier_wait(barrier.at[slot])
                    # logical (BLOCK_K, tn) for wgmma; physically K-minor, so no transpose
                    plgpu.wgmma(acc_ref, a_smem.at[slot], plgpu.transpose_ref(b_smem.at[slot], (1, 0)))
                    partial = acc_ref[...]  # waits for the wgmma group
                    acc_ref[...] = jnp.zeros_like(partial)
                    asf = asf_smem[slot].astype(jnp.float32)  # (tm,)
                    bsf = bsf_smem[slot].astype(jnp.float32)  # (tn,)
                    total += partial * asf[:, None] * bsf[None, :]

                    @pl.when(kb + stages < num_kb)
                    def _():
                        fetch(kb + stages, slot)

                    return total

                total = lax.fori_loop(0, num_kb, k_block, jnp.zeros((tm, tn), jnp.float32))
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
