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
| quantize hidden | 10.9 µs | 12.4 µs (incl. retile) | 1.14× |
| gemm1 `(l,m,k)×(l,2n,k)` | 13.4 µs · 2572 TFLOP/s | 20.7 µs · 1660 | 1.54× |
| silu_and_mul + quantize | 8.7 µs | 17.0 µs | 1.95× |
| gemm2 `(l,m,n)×(l,k,n)` | 9.4 µs · 1821 TFLOP/s | 15.1 µs · 1136 | 1.61× |
| **end to end** | **46.0 µs** | **83.5 µs** | **1.82×** |
| dense bf16 `jnp.einsum` baseline | 62.6 µs | | |

**The port runs at 55% of the reference.** The reference beats the dense bf16 baseline
by 1.36×; this port is still 0.75× it, so the baseline was the flattering comparison and
the CuTeDSL kernel is the real bar.

The gap is structural, and the three causes are visible in
`flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py`:

* **2-CTA collective MMA** — `mma_tiler_mn 256,128` with `cluster_shape_mn 2,1` and
  `use_2cta_instrs`. Twice the M tile, and B is fetched once per cluster rather than
  once per CTA.
* **A persistent scheduler** — `num_persistent_clusters` and a tile scheduler, so CTAs
  launch once and walk tiles, amortizing the prologue and keeping weights in L2.
* **Warp specialization** — dedicated `tma_warp_id`, `mma_warp_id` and a separate
  epilogue warp group. This port has one warpgroup doing TMA issue, MMA and epilogue in
  sequence.

Optimization so far took the port from 120.4 µs to 83.5 µs (1.44×), every step verified
bit-exact:

| change | effect |
|---|---|
| `exp2` instead of `jax.nn.sigmoid` | 42.1 → 27.1 µs on the fused kernel |
| `approx_math=True` | 26.5 → 16.9 µs on the same kernel |
| `SUB_K=64` sub-tiles for the one-hot expansion | 14.2 → 10.3 µs on plain quantize |
| shape-adaptive `block_k` / `stages` | gemm1 27.2 → 20.7, gemm2 18.5 → 15.1 |

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
| `quantize_kernels.py` | the two quantize kernels |
| `moe.py` | the end-to-end path, plus `make_moe_inputs` |
| `test_nvfp4.py` | reference + scale-layout tests; runs anywhere |
| `test_kernels.py` | numerical tests; skips unless the device is sm_100a |
| `verify_b200.py` | non-triviality checks and the scale-buffer stress |
| `check_lowering.py` | builds every kernel for sm_100a from any host, with a negative control |
| `bench_moe.py` | device-time benchmark against a dense bf16 baseline |
| `bench_flashinfer.py` | the same measurement for FlashInfer's CuTeDSL kernel (separate venv) |
| `probe_*.py` | the diagnostics behind the findings below, incl. `probe_collective.py` |

## What the port turned up

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

1. **No warp specialization or persistent scheduler.** JAX's own
   `blackwell_matmul_mgpu.py` is a working template for both. This moved to the top
   after the collective experiment below: it is the prerequisite, not an alternative.
2. **2-CTA collective MMA is implemented but off** (`GemmConfig(collective=True)`),
   because it measured slower — see below.
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
