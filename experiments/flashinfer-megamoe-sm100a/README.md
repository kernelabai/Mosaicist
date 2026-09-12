# FlashInfer MegaMoE → Pallas Mosaic GPU (Blackwell, NVFP4)

The masked MoE path from
[`flashinfer_cutedsl_moe.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py)
ported to Pallas Mosaic GPU on **sm_100a**, keeping NVFP4 end to end: e2m1 data, e4m3
scales per 16 elements, applied inside `tcgen05.mma.kind.block_scale`.

## Status: verified on a B200

Everything here runs and is checked against `nvfp4.py`'s fp32 reference on an NVIDIA
B200. The GEMM is **bit-exact**, including for random inputs — fp4 values and e4m3
scales are all small integers times powers of two, so the products and their sums are
exactly representable in the fp32 accumulator.

```
test_nvfp4     5/5   NVFP4 arithmetic and the MMA scale tiling (runs on any backend)
test_kernels   6/6   4 GEMM bit-exact, 2 end-to-end MoE at rms 1.4e-03 - 2.1e-03
verify_b200   12/12  bit-exact across block_k 128-512, 1-6 stages, tile_n 128/256,
                     1-CTA and 2-CTA collective, up to 32 K blocks, and 4 seeds
check_lowering 6/6   still builds for sm_100a from a non-Blackwell host
```

End-to-end is not bit-exact and cannot be: the intermediate quantization turns a 1-ulp
difference in GEMM1's bf16 output into a whole e2m1 step, and e2m1 has three mantissa
bits. The rms figures are the honest measure.

The first run needed no correctness fixes. The one thing flagged in advance as most
likely wrong — reusing a single scale TMEM buffer across K blocks, on the assumption
that the tensor core's queue orders the copy behind the previous MMA — holds, and
`verify_b200.py` stresses it to 32 reuses across stage counts and seeds.

A Hopper analog of the same algorithm (fp8, scales applied to the accumulator) is in
[`../flashinfer-megamoe-sm90a`](../flashinfer-megamoe-sm90a).

## Performance against the kernel this is a port of

`l=8, m=512, k=2048, n=1024` on a B200. Both paths measured the same way, from CUPTI
activity records; `bench_moe.py` runs the Pallas side and `bench_flashinfer.py` the
reference (it needs its own venv — torch, flashinfer, nvidia-cutlass-dsl).

| stage | FlashInfer CuTeDSL | this port | gap |
|---|---:|---:|---:|
| quantize hidden (+ scale retile) | 10.9 µs | 13.0 µs | 1.19× |
| gemm1 + silu_and_mul + quantize | 13.4 + 8.7 = 22.1 µs | 34.7 µs (one fused kernel) | 1.57× |
| gemm2 `(l,m,n)×(l,k,n)` | 9.4 µs · 1821 TFLOP/s | 12.9 µs · 1327 | 1.37× |
| **end to end** | **46.0 µs** | **65.3 µs** | **1.42×** |
| dense bf16 `jnp.einsum` baseline | 62.6 µs | | |

**The port runs at 70% of the reference**, up from 55%, and is now level with the dense
bf16 baseline (0.96×) which it previously lost to.

Measured in isolation the GEMMs are closer than the table suggests -- gemm1 alone is
18.3 µs at 1880 TFLOP/s and gemm2 12.9 at 1327. An earlier version of this table read
20.7 µs for gemm1 because the benchmark had the scale retile inside the timed region
*and* counted it as its own row.

The gap is structural, and the three causes are visible in
`flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py`:

* **Warp specialization at 192 threads** — four epilogue warps plus one TMA warp and one
  MMA warp. This port originally had one warpgroup doing TMA issue, MMA and epilogue in
  sequence.
* **A deep pipeline** — 14 mbarrier inits, so roughly seven stages, against this port's
  one or two.
* **A real mainloop.** The reference's K loop is a loop; this port's is fully unrolled.

An earlier version of this README claimed the reference used a 2-CTA collective MMA,
read off `mma_tiler_mn 256,128 --cluster_shape_mn 2,1` in its source. That is an example
in a docstring, not what runs. The PTX says `cta_group::1` and `cluster [1,1,1]`: at this
shape the reference is **not** collective, which is consistent with the collective
experiment below losing.

Optimization took the port from 120.4 µs to 65.3 µs (1.84×), every step verified
bit-exact:

| change | effect |
|---|---|
| `exp2` instead of `jax.nn.sigmoid` | 42.1 → 27.1 µs on the activation kernel |
| `approx_math=True` | 26.5 → 16.9 µs on the same kernel |
| `SUB_K=64` sub-tiles for the one-hot expansion | 14.2 → 10.3 µs on plain quantize |
| shape-adaptive `block_k` / `stages` | gemm1 27.2 → 18.3, gemm2 18.5 → 12.9 |
| warp split inside one warpgroup | gemm1 1855 → 1910 TFLOP/s |
| **fusing GEMM1 + silu_and_mul + quantize** | **end to end 83.5 → 65.3 µs** |

That last row is the one worth dwelling on. Timed against each other as standalone
kernels, the fused version beats the two it replaces by 2% -- 34.3 µs against 35.0 --
and on that evidence it looks not worth the complexity. End to end it is worth 22%.
The difference is the `(l, m, 2n)` bf16 intermediate: 16.8 MB that has to be allocated,
written, read back and handed between kernels, and none of that shows up when you time
the two kernels back to back. It was most of the gap between this benchmark's
"sum of stages" and its end-to-end row, which is why that gap was worth chasing.

Optimization took it from 120.4 µs to 83.5 µs (1.44×), every step verified bit-exact:

| change | effect |
|---|---|
| `exp2` instead of `jax.nn.sigmoid` | 42.1 → 27.1 µs on the fused kernel |
| `approx_math=True` | 26.5 → 16.9 µs on the same kernel |
| `SUB_K=64` sub-tiles for the one-hot expansion | 14.2 → 10.3 µs on plain quantize |
| shape-adaptive `block_k` / `stages` | gemm1 27.2 → 20.7, gemm2 18.5 → 15.1 |

## Why Blackwell changes the shape of the kernel

The Hopper analog has no block-scaled MMA, so it reads the wgmma accumulator every K
block, rescales it, and adds to a register total. Ablation there showed that readout
cost more than everything else combined: a pure fp8 GEMM ran at 630 TFLOP/s and the same
kernel with a per-K-block readout at 311.

`tcgen05.mma.kind.block_scale` removes it: the scales are consumed by the MMA, the
accumulator stays in TMEM across all of K, and the epilogue reads it once. That is why
these GEMMs reach 1660 TFLOP/s where the Hopper analog reaches 376.

## Files

| file | what it is |
|---|---|
| `nvfp4.py` | NVFP4 semantics, an fp32 reference for every stage, and the MMA scale tiling |
| `masked_gemm.py` | the block-scaled masked grouped GEMM (`tcgen05_mma` with `a_scale`/`b_scale`) |
| `masked_gemm_ws.py` | `masked_grouped_gemm_w1` (one-warpgroup warp split, **used**) and a two-warpgroup persistent variant (slower) |
| `gemm1_silu_quantize.py` | GEMM1 fused with silu_and_mul and the quantize (**used**) |
| `quantize_kernels.py` | the two quantize kernels |
| `moe.py` | the end-to-end path, plus `make_moe_inputs` |
| `test_nvfp4.py` | reference + scale-layout tests; runs anywhere |
| `test_kernels.py` | numerical tests; skips unless the device is sm_100a |
| `verify_b200.py` | non-triviality checks and the scale-buffer stress |
| `check_lowering.py` | builds every kernel for sm_100a from any host, with a negative control |
| `bench_moe.py` | device-time benchmark against a dense bf16 baseline |
| `bench_flashinfer.py` | the same measurement for FlashInfer's CuTeDSL kernel (separate venv) |
| `probe_*.py` | the diagnostics behind the findings below, incl. `probe_collective.py` |

## Comparing PTX with the reference

This is what the repo's tooling is for, and it is the only reason the warp-split win
below was found. Dump both kernels (`CUTE_DSL_KEEP_PTX=1` for the reference,
`MOSAIC_GPU_DUMP_PTX=1` for this port) and run `mosaicist diff`:

```
fingerprint distance D = 0.349    L0 0.43  L1 0.31  L2 0.29  L3 0.67  L4 0.00

  L0.warpgroups         reference 2 (192 threads), candidate 1 (128)
  L1.tma_dims           reference ['3d','4d'], candidate ['5d']
  L1.stmatrix           reference does not use it, candidate does
  L1.mma_per_stage      reference 3.2, candidate 6.4
  L2.pipeline_barriers  mbarrier.init: reference 14, candidate 3
  L3.order              mainloop similarity 0.33
```

Three things came out of this that reading the reference's source had not:

* **192 threads, not 256.** Six warps: four for the epilogue, one for TMA, one for MMA.
  Two Pallas warpgroups cost 256 and buy two idle warps. Specializing *inside* one
  warpgroup instead is the only structural change so far that has been faster — see
  `masked_grouped_gemm_w1`.
* **The reference is not collective at this shape** (`cta_group::1`), correcting what
  this README previously said.
* **A `bar.sync` before every one of our TMA loads** — 24 CTA barriers against the
  reference's 8. That looked like the smoking gun, and it is not:
  `unsafe_no_auto_barriers=True` removes the hazard tracking and changes nothing
  measurable (17.2 vs 17.4 us). Worth recording precisely because it was the most
  obvious-looking lead.

One caveat about the metric itself: after adopting the warp split, runtime improved
(18.5 -> 18.0 us) while D got slightly *worse* (0.349 -> 0.361). The sub-scores tied to
the actual change moved the right way -- `mma_per_stage` 6.4 -> 4 against the reference's
3.2, `pipeline_barriers` 3 -> 5 against 14 -- but the L3 op-order term rose, because
splitting the warps reorders the loop body without making it less like the reference in
any way that matters. D is a search heuristic here, not a scoreboard.

## What the port turned up

**Warp specialization pays only if it does not cost a warpgroup.** The reference's 192
threads were the clue. Splitting warps *within* one 128-thread warpgroup -- warp 0 issues
TMAs and runs ahead, warp 1 issues the MMAs, then all four warps do the epilogue -- is
faster than the plain kernel, while two full warpgroups are slower than both:

| kernel | threads | smem | blocks/SM | TFLOP/s |
|---|---:|---:|---|---:|
| 1-warpgroup warp split, `block_k=256, stages=2` | 128 | 107 KB | 2 | **1910** |
| plain, `block_k=512, stages=1` | 128 | 107 KB | 2 | 1855 |
| plain, `block_k=256, stages=2` | 128 | 107 KB | 2 | 1784 |
| 2-warpgroup, persistent, `block_k=256, stages=4` | 256 | 181 KB | 1 | 1717 |
| 2-warpgroup, persistent, `block_k=256, stages=2` | 256 | 107 KB | 2 | 1502 |

The warp split also flips which pipeline depth wins: the plain kernel prefers one stage
(nothing to overlap a load with), while the split prefers two, because warp 0 needs a
buffer to run ahead *into*. `_resolve_w1` encodes that.

**Two full warpgroups and a 2-CTA MMA do not pay at these tile sizes.** Both are in
the reference, both were implemented and verified bit-exact here, and both are slower
than the plain kernel. `masked_gemm_ws.py` holds the warp-specialized version: a TMA
warp that runs ahead, an MMA warp, a separate store warpgroup, a persistent
`dynamic_scheduling_loop` and a double-buffered TMEM accumulator.

| kernel | smem | blocks/SM | TFLOP/s |
|---|---:|---|---:|
| plain, `block_k=512, stages=1` | 107 KB | 2 | **1878** |
| plain, `block_k=256, stages=2` | 107 KB | 2 | 1849 |
| warp-specialized, `block_k=256, stages=4` | 181 KB | 1 | 1710 |
| warp-specialized, `block_k=512, stages=2` | 181 KB | 1 | 1665 |
| warp-specialized, `block_k=256, stages=2` | 107 KB | 2 | 1502 |
| warp-specialized, no persistence | 107 KB | 2 | 1055 |

The middle rows are the interesting part. Warp specialization *inverts* the occupancy
rule found above: it prefers the deeper pipeline at 181 KB and one block per SM, because
a TMA warp running ahead inside one block substitutes for the overlap the plain kernel
gets from a second resident block. At these tile sizes the second block is worth more
than the specialized pipeline, and the two cannot be had at once. Persistence is not
optional for the split — without it the store warpgroup has no next tile to overlap with
and the same code drops to 1055 TFLOP/s.

Both remain available and tested, off by default, because the reference does use them:
they are presumably what wins at larger tiles than the 128x128 the MMA scale path allows
here, and combining them (collective *and* warp-specialized, which is the reference's
actual structure) is the one configuration not yet tried.

**A 2-CTA collective MMA does not pay without warp specialization.** The reference uses
one, so it looked like the biggest single lever. Implemented and verified bit-exact, it
is slower at every setting tried:

| config | smem | blocks/SM | TFLOP/s |
|---|---:|---|---:|
| 1 CTA, `block_k=512, stages=1` | 107 KB | 2 | **1878** |
| 1 CTA, `block_k=256, stages=2` | 107 KB | 2 | 1856 |
| 2 CTA, `block_k=256, stages=2` | 90 KB | 2 | 1629 |
| 2 CTA, `block_k=512, stages=1` | 90 KB | 2 | 1562 |
| 2 CTA, `block_k=256, stages=1` | 61 KB | 3 | 1260 |

Halving B's traffic is real, and the cluster does free enough shared memory for a third
resident block, but neither pays for the synchronisation. The collective MMA reads both
CTAs' shared memory, so every K block needs a cluster-wide barrier before the MMA and
another before a stage can be refetched, and in a kernel where the same warpgroup issues
the TMAs those round trips sit on the critical path. FlashInfer pairs its collective MMA
with a dedicated TMA warp that runs ahead of exactly these barriers; that is the part to
copy first.

Two things about it were only learnable by running it. The accumulator and scale TMEM
refs must be declared `collective=True` or lowering rejects them. And each CTA needs the
**whole** tile's B scales even though it holds only half of B's columns — the MMA
indexes `b_scale` by the accumulator's N, which spans both halves. Giving each CTA only
its own half produced finite, wrong results rather than an error.

**Shared memory buys occupancy, not pipeline depth.** A block over half the SM's 232 KB
runs alone, and that decides throughput more than anything else:

| config | smem | blocks/SM | TFLOP/s |
|---|---:|---|---:|
| `block_k=512, stages=1` | 107 KB | 2 | 1829 |
| `block_k=256, stages=2` | 107 KB | 2 | 1745 |
| `block_k=128, stages=4` | 107 KB | 2 | 1388 |
| `block_k=128, stages=8` | 180 KB | 1 | 880 |
| `tile_n=256, stages=2` | 176 KB | 1 | 1502 |

At equal shared memory a wider K step beats more stages, and both beat a deeper pipeline
that costs the second resident block. `GemmConfig` now derives both from `k`. This is the
same lesson the Hopper analog learned independently, where shrinking `stages` to fit two
blocks per SM beat adding a second consumer warpgroup.

**`jax.nn.sigmoid` does not reach the hardware exponential.** It and `lax.logistic` both
cost 42.1 µs on the fused kernel; `g / (1 + exp2(-g·log₂e))` costs 27.1 and is
bit-identical. `exp` does not get there either — only the explicit `exp2` does.

**`approx_math=True` is nearly free accuracy-wise and halves that kernel again**, to
16.9 µs. The outputs are e2m1 and e4m3, so the extra error is orders of magnitude below
a representable step, and both kernels stay bit-identical to the reference.

**PTX told the wrong story here, and measurement corrected it.** The fused kernel's PTX
has 448 `div.full.f32` against 256 `ex2.approx.f32`, which reads like division dominating.
Ablation says otherwise: against an 11.9 µs base, the exponential costs 4.3 µs and the
divides 5.3, with the rest interaction. `approx_math` never removed a single
`div.full.f32` — it changed `ex2.approx.f32` to `ex2.approx.ftz.f` — and that is where
its 9.6 µs came from.

**A scale tile must be at least 16 columns wide.** TMA needs 128 bits along the last
dimension and e4m3 scales are a byte each, so `TILE_K` is 256 (16 scales per row), not
128. The failure mode points at the wrong operand: it reports GMEM strides.

**Mosaic GPU cannot reduce over a sub-row group of a register tile.** Reshaping
`(rows, K)` to `(rows, K/16, 16)` and reducing the last axis has no layout solution under
any annotation tried, on hardware as on the lowering-only check. The kernels loop over
column slices and do plain 2D row reductions instead, and reassemble through constant
one-hot masks — a concatenate has no layout either, and a column-slice store into the
swizzled output is rejected outright. That expansion costs `TILE_K × SUB_K / 16` per
tile, so it is done per 64-column sub-tile rather than per 256-column tile. `SUB_K=32`
sits exactly on TMA's 16-byte minimum row and produces wrong values, so 64 is the floor.

**Scales reach the MMA in a tiling, not row-major**, and `test_nvfp4.py` checks
`to_mma_scale_layout` element-by-element against the PTX spec's index mapping rather
than trusting the reshape.

## Known limitations, in the order worth fixing

The order below was revised after measuring: three structural GEMM optimizations were
implemented and all three were slower, so the remaining value is in the quantize path,
not the GEMM.

1. **The fused kernel carries three operand streams per stage** (A, gate-B, up-B) where
   the plain GEMM carries two, so fewer stages and blocks fit and its GEMM half runs
   slower than `masked_gemm` would. That is why fusing wins 22% end to end but only 2%
   against the two kernels it replaces. Recovering the difference means either a wider
   activation tile without a fourth accumulator, or B tiles shared between the two
   passes.
2. **The scale retile is still a separate pass** (2.5 µs plus a launch, and it shows up
   as `input_transpose_fusion` in the profile). Writing the MMA's scale tiling directly
   from the quantize epilogue needs a scatter within the tile that is not expressible as
   a vector store.
3. **2-CTA collective is implemented but off** (`GemmConfig(collective=True)`), measured
   slower, and the PTX shows the reference does not use it at this shape either.
   Two-warpgroup specialization (`masked_gemm_ws.masked_grouped_gemm_ws`) is likewise
   off and slower; the one-warpgroup warp split that *is* on came out of the PTX diff.
3. **`silu_and_mul + quantize` is a separate launch** that writes 16.8 MB of bf16 and
   reads it straight back. With all arithmetic ablated away it still costs 11.9 µs, so
   that traffic is most of the 1.95× gap on the worst stage; it belongs in GEMM1's
   epilogue.
4. **The one-hot expansion** exists only because Mosaic cannot reduce over a sub-row
   group of a register tile. Worth a JAX issue, or an attempt through
   `plgpu.inline_mgpu`.
5. `tile_m` is fixed at 128 by the MMA scale path; `tile_n` may be 128 or 256, and 256
   measured slower because it costs the second resident block.
* The scale re-tiling between stages is a separate pass over a tensor 1/16 the size of
  the data (1.9 µs) rather than being fused into the quantize epilogue; the tiled store
  is a scatter within the tile, so it needs a different epilogue, not a tweak.
* `m` and `n` must be multiples of 128 and `k` of `block_k`; there is no ragged epilogue.
  Rows past `masked_m` inside a partially masked tile are computed and written rather
  than zeroed — safe because every stage is row-independent, but callers must not read
  above `masked_m`.
