# Porting FlashInfer's MegaMoE CuTeDSL kernel to Pallas Mosaic GPU

### Why performance parity is not currently achievable, and what would change that

**Hardware:** NVIDIA B200 (sm_100a) · **Software:** JAX/jaxlib 0.11.1, flashinfer-python
0.6.18.post1, nvidia-cutlass-dsl 4.7.1, CUDA 12.8, driver 570.148.08

---

## 1. Summary

The masked mixture-of-experts path from FlashInfer's `flashinfer_cutedsl_moe.py` was
ported to Pallas Mosaic GPU, keeping NVFP4 end to end: e2m1 data, e4m3 scales per 16
elements, applied inside `tcgen05.mma.kind.block_scale`. The port is numerically faithful
— the grouped GEMM is **bit-exact** against a float64 reference, and the full path agrees
to bf16 (rms 1.4–2.1 × 10⁻³).

On performance it reaches **60.8 µs against the reference's 46.0 µs — 76% of the
reference**, having started at 120.4 µs. Every remaining difference the tooling reports is
structural rather than a tuning parameter, and the decisive one cannot be expressed in
Pallas today:

> **A warp-specialized Blackwell GEMM launches 192 threads — four epilogue warps, one TMA
> warp, one MMA warp. `plgpu.kernel`'s `num_threads` counts *warpgroups*, so the reachable
> block sizes are 128, 256, 384. 192 is not among them, and both neighbours give up
> something the reference keeps.**

This is not an inference from reading source. The reference's own configuration was
reconstructed from its captured launch record and **run**: adopting it makes the Pallas
kernel *slower*, because it only pays with the thread layout that cannot be built.

Two limitations have been reported upstream with self-contained reproductions
(`jax-bug-warpgroup-granularity.txt`, `jax-bug-layout-inference.txt`).

---

## 2. Method

Work was driven by **Mosaicist**, a convergence engine built alongside the port. It
captures both kernels into a common bundle (PTX, SASS, launch record, timing samples),
lifts each into a six-layer fingerprint (launch geometry, instruction mix, loop and
pipeline structure, instruction order, floating-point flavour, machine/SASS), diffs them
layer by layer, and maps each discrepancy to a source-level fix.

Three properties of the method matter for reading the results:

- **Runtime is the objective; numerics are a gate.** A candidate is kept only if it passes
  the numerics gate *and* is faster than the incumbent by more than the reference's own
  run-to-run noise. Fingerprint distance only breaks ties.
- **Timing is device time from CUPTI activity records, not wall clock.** Every kernel here
  finishes well inside the ~1.5 ms host round trip; a `block_until_ready` loop reports that
  floor for all of them and even ranks a fused path faster than one of its own GEMMs.
- **The noise floor is measured, not assumed.** The reference's spread is 1.0%; this port's
  is nearer 2%. Several changes that looked like wins did not survive repetition and are
  reported below as null results.

---

## 3. Results

### 3.1 Correctness

| suite | result |
|---|---|
| NVFP4 arithmetic and MMA scale tiling (any backend) | 5/5 |
| Kernel numerics on B200 | 6/6 — 4 GEMM bit-exact, 2 end-to-end at rms 1.4–2.1e-3 |
| Stress: block_k 128–512, 1–6 stages, tile_n 128/256, 1-CTA and 2-CTA, 32 K blocks, 4 seeds | 12/12 bit-exact |
| Engine unit tests | 84 |

The GEMM is bit-exact even for random inputs because fp4 values and e4m3 scales are all
small integers times powers of two, so products and their fp32 sums are exactly
representable. End-to-end cannot be bit-exact: an intermediate quantization turns a 1-ulp
difference in GEMM1's bf16 output into a whole e2m1 step, and e2m1 has three mantissa bits.

### 3.2 Performance

| stage | FlashInfer CuTeDSL | this port | gap |
|---|---:|---:|---:|
| quantize hidden (+ scale retile) | 10.9 µs | 13.0 µs | 1.19× |
| gemm1 + silu_and_mul + quantize | 13.4 + 8.7 = 22.1 µs | 31.6 µs (one fused kernel) | 1.43× |
| gemm2 | 9.4 µs · 1821 TFLOP/s | 13.0 µs · 1322 TFLOP/s | 1.38× |
| **end to end** | **46.0 µs** | **60.8 µs** | **1.32×** |
| dense bf16 `jnp.einsum` baseline | 65.4 µs | | |

### 3.3 What produced the 120.4 → 60.8 µs improvement

| change | effect |
|---|---|
| `exp2` instead of `jax.nn.sigmoid` | 42.1 → 27.1 µs on the activation kernel |
| `approx_math=True` | 26.5 → 16.9 µs on the same kernel |
| `SUB_K=64` sub-tiles for the scale expansion | 14.2 → 10.3 µs on plain quantize |
| shape-adaptive `block_k` / `stages` | gemm1 27.2 → 18.3, gemm2 18.5 → 12.9 |
| warp split inside one warpgroup | gemm1 1855 → 1910 TFLOP/s |
| **fusing GEMM1 + silu_and_mul + quantize** | **end to end 83.5 → 65.3 µs** |
| select instead of multiply-add in the expansion | fused 34.1 → 32.5 µs |
| values by concatenation | fused 32.0 → 31.2 µs |

The fusion is the largest single win, and it illustrates why isolated benchmarking
misleads: timed head-to-head against the two kernels it replaces, the fused version is 2%
faster. End to end it is worth 22%. The difference is the `(l, m, 2n)` bf16 intermediate —
16.8 MB allocated, written, read back and handed between launches — which does not appear
when two kernels are timed back to back.

---

## 4. The structural experiments, and why each failed

Six structural changes were implemented, verified bit-exact, and measured. All were taken
from the reference's own structure. Measured on the grouped GEMM at l=8, m=512, k=n=2048:

| configuration | threads | smem | blocks/SM | TFLOP/s |
|---|---:|---:|---|---:|
| **1 warpgroup + warp split + persistent** (shipped) | 128 | 107 KB | 2 | **~1930** |
| 1 warpgroup, warp split, `block_k=256, stages=2` | 128 | 107 KB | 2 | 1910 |
| plain, `block_k=512, stages=1` | 128 | 107 KB | 2 | 1878 |
| 2 warpgroups, persistent, `block_k=256, stages=4` | 256 | 181 KB | 1 | 1717 |
| 2-CTA collective, `block_k=256, stages=2` | 128 | 90 KB | 2 | 1629 |
| 2 warpgroups, persistent, chunked epilogue | 256 | 107 KB | 2 | 1456 |
| reference's own reconstructed tile shape | 128 | 205 KB | 1 | ~1790 |

**2-CTA collective MMA** halves B's traffic and frees enough shared memory for a third
resident block, and is slower at every setting. The collective MMA reads both CTAs' shared
memory, so each K block needs a cluster-wide barrier before the MMA and another before a
stage can be refetched; with one warpgroup issuing the TMAs, those round trips sit on the
critical path. *The PTX later showed the reference does not use a collective MMA at this
shape either* — `cta_group::1`, `cluster [1,1,1]` — correcting an assumption taken from a
docstring example in its source.

**Warp specialization across two warpgroups** was implemented with a persistent
`dynamic_scheduling_loop` and a double-buffered TMEM accumulator, following JAX's own
`blackwell_matmul_mgpu.py`. It is slower. The hypothesis was register pressure: at 255
registers two 256-thread blocks cannot share an SM. Copying the reference's chunked
epilogue dropped that to ~98 registers and two blocks now do fit — and it is *still* 33%
slower. Register pressure was never the cause. The epilogue is roughly one ninth of a
tile's time, which cannot pay for two idle warps in the compute warpgroup plus a
synchronisation per tile.

**Pipeline depth.** The reference runs ~7 stages (14 mbarrier inits) against this port's 2.
Thirty configurations were swept. Every configuration whose block exceeds half the SM's
shared memory — and therefore runs alone — loses roughly 25%. Deep pipelines with fine K
steps are far worse (`block_k=64` bottoms out at 29 µs). Depth does not transfer.

**The reference's tile shape.** Its capture reports 226304 bytes of shared memory; solving
for what fills that at ~7 stages yields exactly two candidates, `tile_n=128, block_k=256,
stages=5` and `tile_n=256, block_k=128, stages=6`. Both were built and run: 19.2 µs against
this port's 17.8. **Adopting the reference's own configuration makes this kernel slower.**

The pattern is consistent. The reference's configuration is a package — 226 KB of shared
memory, one block per SM, a deep pipeline — that only pays because 192 threads let a TMA
warp and an MMA warp run ahead while four separate warps drain the previous tile. Take any
piece of it without the thread layout and it costs more than it returns.

---

## 5. The limitations

### 5.1 Block size is a whole number of warpgroups

`plgpu.kernel`'s `num_threads` counts warpgroups. Verified by launching a trivial kernel at
each setting and reading the block size back from CUPTI:

```
 num_threads   threads launched
        None                128
           1                128
           2                256
           3                384
```

The reference launches 192. The two reachable neighbours each give up something:

- **128 threads** — warps 0 and 1 issue TMAs and MMAs, then all four run the epilogue. This
  is the fastest arrangement found, but a tile's epilogue cannot overlap the next tile's
  loads, because the same warps do both.
- **256 threads** — a dedicated epilogue warpgroup makes that overlap possible, but the
  extra warpgroup costs the second resident block, and it measures slower.

192 buys the overlap without the occupancy, because two extra warps is a quarter of a
warpgroup rather than a whole one. `plgpu.warp_map` already provides per-warp roles *inside*
a warpgroup; the gap is only that a block cannot be sized in warps.

Reported: `jax-bug-warpgroup-granularity.txt`.

### 5.2 Layout inference cannot express block-wise reduce-and-broadcast

Every block-scaled format needs the same two steps on a register tile: reduce over
contiguous groups of V columns to get one scale per group, then broadcast those scales back
over their groups. Of eight natural spellings, **two lower**; the reshape-and-reduce form,
`jnp.repeat`, `broadcast_to` + `reshape`, a `jnp.where` with a broadcast condition, and a
concatenate of the reduced scales all fail — checked in both the WGMMA layout reading shared
memory and the TCGEN05 layout reading TMEM.

The working formulation reduces each group as a separate 2-D slice and reassembles through
constant one-hot masks: one full-tile operation per group, O(K²/V) work to express what is
logically a broadcast. It cost 9.5 µs of a 34 µs kernel before mitigation. Three mitigations
brought that down — a select instead of a multiply-add, narrower sub-tiles, and assembling
the *values* by concatenation (which does lower, unlike the scales) — but the workaround
cannot be removed.

This matters beyond one kernel: `plgpu.tcgen05_mma(a_scale=, b_scale=)` already exposes
Blackwell's block-scaled MMA, and producing its operands is the required other half of the
feature.

Reported: `jax-bug-layout-inference.txt`.

### 5.3 Smaller, same root

The scale retile between stages (~2.5 µs plus a launch, visible as `input_transpose_fusion`)
cannot be folded into the quantize epilogue because writing the MMA's scale tiling requires
a scatter within the tile — row *r*, block *c* lands at `[c//4, r%32, 4·((r%128)//32) + c%4]`
— which is not expressible as a vector store. Same root cause as 5.2.

---

## 6. Conclusion

Parity with the CuTeDSL kernel is **not achievable in Pallas Mosaic GPU as of JAX 0.11.1**
for this kernel on this hardware. The evidence is:

1. Tuning is exhausted. Thirty pipeline configurations, six structural variants, and an
   automated convergence loop all converge on the same operating point, and the last several
   candidate improvements landed inside the measurement noise.
2. The reference's configuration was reconstructed from its captured launch record and run.
   It is slower here. The configuration does not transfer without the thread layout.
3. The thread layout cannot be expressed, and this is verified by direct measurement rather
   than inferred: block sizes are multiples of 128, and 192 is required.

What the port *did* achieve is worth stating alongside that: numerical fidelity (bit-exact
GEMM), 76% of the reference's throughput, and a 1.08× margin over the dense bf16 baseline
that the same problem would otherwise use.

### What would change the conclusion

- **Warp-granular block sizing** (a `num_warps=` alternative, or a partial second
  warpgroup). This is the decisive one. A TMA- or MMA-issuing warp uses none of a
  warpgroup's collective machinery, so the 128-lane requirement does not apply to it.
- **A layout rule for block-wise reduce and broadcast**, or a `plgpu.block_reduce`
  primitive. Worth roughly 8 µs here and required by every block-scaled format.
- **A scatter-capable store** for the MMA scale tiling, which would remove the retile pass.

Neither reported limitation is a defect; both are missing capabilities in a young API whose
block-scaled MMA support is otherwise complete enough to build a bit-exact NVFP4 MoE on.

---

## Appendix A — Reproducing

All artifacts are archived in `artifacts/`, since both GPU instances have been released.
The bundles are complete, so everything except running a kernel still works:

```bash
mosaicist fingerprint artifacts/b200/captures/ref
mosaicist diff artifacts/b200/captures/ref artifacts/b200/captures/cand_best
```

That diff reports D = 0.386 and names the differences discussed above: 192 threads against
128, 226 KB of shared memory against 107, and a persistent `grid=[1,1,148]` against a tiled
one.

## Appendix B — Reference kernel, as captured

```
grid [1, 1, 148]      one block per SM: persistent
block [192, 1, 1]     six warps: 4 epilogue + 1 TMA + 1 MMA
dynamic smem 226304   nearly the whole SM
mma  tcgen05 mxf4nvf4, cta_group::1 (not collective at this shape)
mbarrier.init 14      ~7 pipeline stages
```

Neither the persistence nor the warp specialization is visible in the PTX alone — the
fingerprint's control-flow detectors miss both. They come from the launch record, which is
the argument for capturing rather than reading disassembly.
