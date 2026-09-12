# FlashInfer MegaMoE → Pallas Mosaic GPU (Blackwell, NVFP4)

The direct port of the masked MoE path in
[`flashinfer_cutedsl_moe.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py)
to Pallas Mosaic GPU on **sm_100a**, keeping NVFP4 end to end: e2m1 data, e4m3 scales
per 16 elements, applied inside `tcgen05.mma.kind.block_scale`.

## Status: not run

**No sm_100a device was available, so none of these kernels have ever executed.** Treat
every performance or correctness claim about the kernels as unproven. What *is* checked
is stated precisely below — the point of this section is that you can tell the two
apart.

| what | how | status |
|---|---|---|
| NVFP4 quantization arithmetic | `test_nvfp4.py`, plain JAX, any backend | **5/5 pass** on H100 |
| the MMA scale tiling (`to_mma_scale_layout`) | `test_nvfp4.py`, checked element-by-element against the index mapping in the PTX spec, and for round-trip | **passes** |
| every kernel builds for Blackwell | `check_lowering.py` — points Mosaic's arch detection at sm_100a and lowers | **4/4 lower**, tcgen05 emitted |
| the lowering check can actually fail | negative control in the same file | **confirmed** — it trips `async_store_scales_smem_to_tmem`'s verifier |
| the kernels compute the right answer | `test_kernels.py` | **never run** — skips on non-Blackwell |
| performance | — | **unmeasured** |

`check_lowering.py` is worth being precise about: it proves the kernels satisfy every
shape, layout and dtype rule Mosaic and the MMA impose, which is most of what is hard
about this port. It proves nothing about whether the arithmetic is right, whether the
pipeline synchronises correctly, or whether it is fast.

A Hopper-runnable analog of the same algorithm — fp8 instead of fp4, scales applied to
the accumulator instead of inside the MMA — is fully verified and benchmarked in
[`../flashinfer-megamoe-sm90a`](../flashinfer-megamoe-sm90a). Where the two disagree
about something testable, that one is the authority.

## Why Blackwell changes the shape of the kernel

The Hopper analog has no block-scaled MMA, so it reads the wgmma accumulator every K
block, scales it by the outer product of the two scale vectors, and adds it to a
register total. Ablation there showed that readout costs more than everything else in
the kernel combined — a pure fp8 GEMM runs at 630 TFLOP/s and the same kernel with a
per-K-block accumulator readout runs at 311, before any scaling arithmetic.

`tcgen05.mma.kind.block_scale` removes that entirely: the scales are consumed by the
MMA, the accumulator stays in TMEM across all of K, and the epilogue reads it once.
That is the whole reason this port exists in fp4 form.

## Files

| file | what it is |
|---|---|
| `nvfp4.py` | NVFP4 semantics, an fp32 reference for every stage, and the MMA scale tiling |
| `masked_gemm.py` | the block-scaled masked grouped GEMM (`tcgen05_mma` with `a_scale`/`b_scale`) |
| `quantize_kernels.py` | the two quantize kernels |
| `moe.py` | the end-to-end path, plus `make_moe_inputs` |
| `test_nvfp4.py` | reference + scale-layout tests; runs anywhere |
| `check_lowering.py` | builds every kernel for sm_100a, with a negative control |
| `test_kernels.py` | numerical tests; skips unless the device is sm_100a |
| `probe_lower.py` | the bisect harness that produced the findings below |

## What the port turned up

Every one of these came from `probe_lower.py`, which bisects formulations by whether
they survive Mosaic's lowering. Without a device, "does it build for sm_100a" was the
only oracle available, and it was enough to find all of them.

**A scale tile must be at least 16 columns wide.** TMA requires 128 bits along the last
dimension, and e4m3 scales are a byte each, so a tile of 8 scales is rejected. That sets
`TILE_K = 256` in the quantize kernels — 256 elements of K give exactly 16 scales per
row. The natural choice of 128 fails, and fails with a message about GMEM strides that
points at the wrong operand.

**Mosaic GPU cannot reduce over a sub-row group of a register tile.** Reshaping a
`(rows, K)` tile to `(rows, K // 16, 16)` and reducing the last axis has no layout
solution — with a `layout_cast` to WGMMA, without one, with the cast on the result
instead, or under `WG_STRIDED`. Since a 16-element block scale is exactly such a
reduction, the kernel instead loops over 16 column slices in Python and does a plain 2D
row reduction on each, which is the one form that works.

**The per-block results cannot be reassembled the obvious way.** Writing each block into
a column slice of the output smem fails outright ("We cannot apply swizzle to
non-contiguous refs"), and `jnp.concatenate` of the per-block arrays has no layout. What
works is accumulating each block into a full-width array through a constant one-hot
mask. It costs 16 full-tile multiply-adds per tile and about a megabyte of unrolled IR,
which is the main thing a Blackwell run should be asked to justify.

**Scales reach the MMA in a tiling, not row-major.** `async_copy_scales_to_tmem` wants
smem shaped `(mn // 128, k_scales // 4, 32, 16)`; `nvfp4.to_mma_scale_layout` produces
it, and `test_nvfp4.py` checks it element-by-element against the PTX spec's index
mapping rather than trusting the reshape. The kernels take weight scales already tiled,
and `moe.py` re-tiles the activation scales between stages.

## Known limitations

* **Unverified.** Listed first because it dominates everything else.
* `tile_m` and `tile_n` are both fixed at 128. The MMA scale path is defined in terms of
  128-row TMEM tiles and rejects more than two of them, so this is not a tuning knob.
* Single warpgroup, no warp specialization, and no persistent scheduler — the structure
  is the simplest one that can be correct, not the fastest. JAX's own
  `blackwell_matmul_mgpu.py` splits the MMA and the epilogue across two warpgroups; that
  is the first thing to do once the kernel is known to run.
* The scale re-tiling between stages is a separate pass over a tensor 1/16 the size of
  the data, rather than being fused into the quantize kernel's epilogue.
* One scale buffer per operand is reused across K blocks, on the assumption that the
  tensor core's queue orders the copy behind the previous MMA. This is the assumption
  most likely to be wrong, and it is marked `ASSUMPTION` in `masked_gemm.py` along with
  the fix (give the scale refs a `stages` leading dimension).
* `m`, `n` must be multiples of 128 and `k` a multiple of `block_k`; there is no ragged
  epilogue. As on Hopper, rows past `masked_m` inside a partially masked tile are
  computed and written rather than zeroed, which is safe because every stage is
  row-independent, but callers must not read above `masked_m`.
