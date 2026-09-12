# FlashInfer MegaMoE → Pallas Mosaic GPU (Hopper analog)

A port of the masked MoE path in
[`flashinfer_cutedsl_moe.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py)
(SGLang's wrapper around FlashInfer's CuTeDSL NVFP4 grouped GEMM) to Pallas Mosaic GPU,
targeting **sm_90a**. The reference needs Blackwell — `tcgen05.mma.kind.block_scale`,
TMEM, and fp4 tensor cores are sm_100a-only — so this is a Hopper-runnable *analog*:
the same algorithm and the same two-level scaling scheme, with the data type and block
size changed to what Hopper's tensor cores actually have.

An unverified sm_100a port that keeps NVFP4 lives in `../flashinfer-megamoe-sm100a`.

## What changed, and why

| | reference (sm_100a) | this port (sm_90a) |
|---|---|---|
| data | e2m1 (fp4) | e4m3 (fp8) — Hopper's narrowest tensor-core type |
| scale block | 16 elements along K | 128 along K — scales are applied to the wgmma accumulator, and one fp8 wgmma already spans k=32 |
| scales | e4m3 + per-expert global scale | unchanged |
| scale application | inside the MMA (`block_scale`) | at accumulator granularity — Hopper has no block-scaled MMA |

Everything else is the reference's structure: per-expert `masked_m` row counts, the NT
(`lmk,lnk->lmn`) operand layout, per-expert alphas with both global scales folded in,
and the four-stage chain quantize → GEMM1 → silu_and_mul+quantize → GEMM2.

## Files

| file | what it is |
|---|---|
| `quant.py` | the quantization scheme and an fp32 reference for every stage, including `moe_reference` |
| `masked_gemm.py` | the masked block-scaled grouped GEMM (the port's core) |
| `quantize_kernels.py` | `scaled_fp4_grouped_quantize` and `silu_and_mul_scaled_nvfp4_experts_quantize` analogs |
| `moe.py` | the end-to-end path, plus `make_moe_inputs` |
| `test_gemm.py` / `test_quantize.py` / `test_moe.py` | correctness against `quant.py` |
| `bench_moe.py` | device-time benchmark vs a dense bf16 baseline |
| `probe_layout.py` / `probe_ablate.py` / `probe_pipe.py` / `probe_wg.py` | the diagnostics that produced the findings below |

## Correctness

All on an H100 PCIe. The GEMM and both quantization kernels are **bit-exact** against
the fp32 reference wherever the reference's own arithmetic is exact:

```
test_gemm      10/10  7 bit-exact (incl. odd/single K-block tails and both warpgroup
                      counts), 3 random-float at 2.65e-03
test_quantize   2/2   values=exact scales=exact for both kernels
test_moe        3/3   max rel err 1.1e-02 - 1.6e-02, rms 2.6e-04 - 1.1e-03
```

End-to-end is not bit-exact and cannot be: the intermediate quantization turns a 1-ulp
difference in GEMM1's bf16 output into a full e4m3 step on that element. The rms figures
are the honest measure — ~3e-4 relative, i.e. the two implementations agree to bf16.

## Performance

`l=8, m=512, k=2048, n=1024`, device time from CUPTI activity records:

| stage | µs | TFLOP/s |
|---|---:|---:|
| quantize hidden | 9.6 | |
| gemm1 `(l,m,k)×(l,2n,k)` | 91.9 | 373.8 |
| silu_and_mul + quantize | 16.0 | |
| gemm2 `(l,m,n)×(l,k,n)` | 51.1 | 336.2 |
| **moe_masked (end to end)** | **180.9** | **284.9** |
| dense bf16 `jnp.einsum` baseline | 158.2 | 325.8 |

Wall clock is useless at this size — every kernel here finishes well inside the ~1.5 ms
host round-trip, so a `block_until_ready` loop reports the same 1.5 ms for all of them
and even ranks the fused path faster than one of its own GEMMs. `bench_moe.py` uses
CUPTI device time instead.

**Both GEMMs now beat the dense bf16 baseline** (374 and 336 TFLOP/s against its 326),
but the end-to-end path is 0.87× it, because the baseline does no quantization and this
one spends 26 µs on it. That is a real cost of the format, not an inefficiency to tune
away — though fusing the quantization into the previous GEMM's epilogue would remove
most of it, and is the obvious next step.

## What the port turned up

**fp8 wgmma cannot transpose its operands.** Only 16-bit types can, so both operands
must be K-minor in shared memory. B is stored as `(tn, BLOCK_K)` and handed to wgmma as
a transposed *view*, which leaves the physical layout alone. Storing it the natural way
fails with "Only f16 WGMMA supports transposes".

**Scale factors have to be K-major.** `(l, k // BLOCK_K, rows)`, not the natural
`(rows, k // BLOCK_K)`: TMA cannot load a 1-byte inner dimension, so the tile's scale
vector has to be the contiguous one.

**TMA destinations need 128-byte alignment, which sets a floor on tile size.** A
`(stages, 64)` e4m3 scale buffer puts odd slots on a 64-byte boundary, and every
`tile_n=64` config faulted with `CUDA_ERROR_MISALIGNED_ADDRESS` — from the *scale*
copy, not the operand copy that looks like the suspect. Padding each stage's slice to
128 bytes (`SF_ALIGN`) fixes it, and is what makes the current best tile size legal.

**Mosaic GPU layout inference can't solve a row reduction plus a broadcast back.** The
quantize kernels need `layout_cast` to `Layout.WGMMA` for the tile and
`Layout.WGMMA.reduce(1)` for the per-row scale, or they fail with "Layout inference
failed to find a solution". One further shape has no layout even *with* the
annotations, and it took bisecting with `probe_layout.py` to find: a `jnp.where` whose **condition** is broadcast from the reduced layout to the full
tile. A select on the reduced vector is fine, and a broadcast *divide* is fine; it is
specifically the broadcast condition. `_quantize_tile` expresses the reference's
`where(step > 0, ...)` guard as a multiply by a 0/1 mask instead, which is identical
arithmetic (`x * 1.0` is exact, `x * 0.0` is zero).

**The fused silu+quantize must not round to bf16 in between.** The reference fuses the
activation into the quantize kernel, so the fp32 silu output is quantized directly;
`quant.py` originally rounded it to bf16 first, which made the Pallas kernel look wrong
when it was the reference that was off.

**Two resident blocks beat a second warpgroup, and are free.** The kernel's problem is
that the CUDA-core rescale does not overlap the tensor-core MMA. The textbook fix is a
second consumer warpgroup splitting the tile's N — that is implemented, correct, and
*slower*. Two warpgroups sharing one ring of smem buffers must hand off through a
`consumed` barrier before a stage can be refetched, and that handshake puts them back in
lockstep; and doubling the threads per block at 255 registers each leaves room for only
one block per SM. Shrinking `stages` instead until the block fits in half the SM's shared
memory gets two blocks resident, and the hardware interleaves them at no cost:

| config | smem | blocks/SM | TFLOP/s |
|---|---:|---|---:|
| `consumers=1, stages=3` (default) | 92 KB | 2 | **376** |
| `consumers=1, stages=8` | 215 KB | 1 | 318 |
| `consumers=2, stages=4` | 166 KB | 1 | 251 |
| `consumers=2, stages=2` | 100 KB | 1 (register-bound) | 237 |

Deeper pipelines are worth less than the second block, which is the opposite of the
usual advice, and only shows up if you check occupancy rather than assume it.

**Reading a wgmma accumulator drains every wgmma in flight, and that dominates.**
Ablating the mainloop (`probe_ablate.py`, same shape as above):

| mainloop | µs | TFLOP/s |
|---|---:|---:|
| pure fp8 GEMM, one accumulator readout at the end | 54.5 | 630.7 |
| + read and zero the accumulator every K block | 110.4 | 311.1 |
| + scale by the two scale vectors (the real kernel) | 132.7 | 259.0 |

The block scaling itself is cheap. The *readout* costs more than everything else
combined, because `acc_ref[...]` compiles to `wgmma.wait_group 0` and no wgmma ever
overlaps another. Running two accumulators and two K blocks per iteration —
`wgmma_accumulator_load` under an explicit `wgmma_wait(1)`, so block *n*'s rescale runs
while block *n+1* is still on the tensor cores — recovers about 1.2× (259 → 316
TFLOP/s), and a sweep put the sweet spot at `tile_n=64` with 8 stages: a 128×64 fp32
accumulator is 64 registers per thread, which leaves room for the second one without
spilling.

## Known limitations

* **No producer/consumer split.** The 128 threads of a block still do the TMA issue, the
  MMA and the rescale themselves. A *dedicated producer* warpgroup that only issues TMAs
  is the version worth trying next — unlike the second *consumer* warpgroup measured
  above, it does not contend for registers with an accumulator, so it should not cost
  the second resident block. The remaining gap to the 630 TFLOP/s pure-GEMM ceiling is
  what it would be competing for.
* The quantize kernels are separate launches; fusing them into the preceding GEMM's
  epilogue would remove the 26 µs that currently keeps the end-to-end path below the
  bf16 baseline even though both GEMMs are above it.
* **Tile sizes below 128 need the `SF_ALIGN` padding** and were only tested at 64.
* `tile_m` must divide `m` and `tile_n` must divide `n`; there is no epilogue predication
  for ragged tiles. Rows past `masked_m` inside a partially masked tile are computed and
  written with real values rather than zeros — every stage is row-independent, so this
  never affects a valid row, but callers must not read above `masked_m`.
* The dense bf16 baseline is not a fair *accuracy* comparison, only a speed one: it does
  no quantization at all.
