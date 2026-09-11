# CuTeDSL port of CUTLASS example 69: Hopper mixed-dtype grouped GEMM

A CuTeDSL (CUTLASS Python DSL 4.7.1) port of
[`examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mixed_dtype_grouped_gemm.cu`](https://github.com/NVIDIA/cutlass/tree/main/examples/69_hopper_mixed_dtype_grouped_gemm).
It has the same semantics, algorithm and tile configuration, verified bit-exact against a reference, and it matches the C++ kernel's performance.

    D_g = alpha_g * (A_g @ dequant(B_g)^T) + beta_g * C_g      for each group g
    dequant(B)[n, k] = bf16(B_q[n, k]) * scale[n, k / c]        (bf16 multiply)

- A: bf16 activations. B: fp8 weights (e5m2; e4m3 also works). Scales: bf16 with group size `c` along K, or none for convert-only mode. C and D: fp16.
- Like the C++ example, the kernel computes `D^T = dequant(B) A^T`, so the quantized operand is wgmma's register-sourced A operand and is converted in registers.
- Kernel structure:
  - persistent, warp-specialized, 128×16×64 tiles;
  - one TMA producer warp, one scheduler warp, and two cooperative MMA warpgroups;
  - per-group TMA descriptors rewritten in the kernel when a CTA's tile stream enters a new group.

## Files

| file | what |
|---|---|
| `grouped_mixed_gemm.py` | the grouped kernel (`HopperMixedInputGroupedGemmKernel`) |
| `mixed_gemm.py` | single-problem kernel; the grouped kernel inherits its mainloop transform |
| `grouped_utils.py` | group scheduler, per-group tensors, predicated TMA-with-descriptor wrapper |
| `problems.py` | problem generation in the example's layouts, reference |
| `run_grouped.py` / `run_single.py` | drivers (`run_grouped.py` mirrors the C++ options, including `--benchmark` files) |
| `compare_cpp.py` | runs C++ and CuTeDSL alternately on identical problem sets |
| `check_matrix.py` | correctness matrix |
| `debug_patterns.py` | structured-input diagnostics used while debugging the A path |

## Run (sm_90a GPU)

```bash
pip install nvidia-cutlass-dsl torch cupti-python     # CUDA 13 wheels; CuTeDSL 4.7.1 tested
python run_grouped.py --groups 16 --m 2048 --n 5120 --k 8192 --c 512
python run_grouped.py --groups 6                 # the example's default: random M, N=2048, K=512
python check_matrix.py                           # 8 configurations, bit-exact
python compare_cpp.py --cpp /path/to/69_hopper_mixed_dtype_grouped_gemm
```

Building the C++ baseline with CUDA 12.8:

```bash
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -DNDEBUG \
  -I include -I tools/util/include -I examples/common -I examples/55_hopper_mixed_dtype_gemm \
  examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_mixed_dtype_grouped_gemm.cu
```

The two int4 variants also need a shadow copy of `cutlass/subbyte_reference.h` placed first on the include path. Its `__nv_atomic_load_n` guard claims CUDA 12.8 support, but 12.8.93's nvcc rejects that call; the shadow copy raises the guard to ≥ 12.9. This only affects the host-side test-data filler, not the kernel.

## Results (H100 PCIe, unlocked clocks)

Both implementations run the same problem files with α=1 and β=0.5, so every group reads C. Times are the median over interleaved rounds of each tool's CUDA-event average.

| configuration | C++ | CuTeDSL | ratio |
|---|---|---|---|
| 16 groups, 2048x5120x8192, convert-only | 30.65 ms | 30.67 ms | 1.00 |
| 16 groups, 2048x5120x8192, c=512 | 34.40 ms | 34.46 ms | 1.00 |
| 16 groups, 4096x5120x8192, c=8192 (per column) | 72.53 ms | 71.34 ms | 0.98 |
| 6 groups, random M <= 1024, N=2048, K=512 | 0.160 ms | 0.161 ms | 1.01 |
| 100 groups, random M <= 1024, N=2048, K=512 | 1.628 ms | 1.688 ms | 1.04 |
| 100 groups, 2048x512x512 | 1.539 ms | 1.618 ms | 1.05 |
| 100 groups, 128x128x512 | 0.047 ms | 0.040 ms | 0.85 |

Median of 3 interleaved rounds (`compare_cpp.py --rounds 3`). A second, independent 2-round run
gave ratios 1.01, 1.02, 0.99, 1.03, 1.06, 1.04, 0.90 for the same rows. Clocks were not locked on
this H100 PCIe and run-to-run spread is a few percent, so treat ratios within ~5% as parity.

`check_matrix.py` is bit-exact for every group across these configurations:

- e5m2 weights with scale groups of c=512; convert-only; e4m3 weights;
- per-column scales (c = K); fine scale groups (c=128);
- ragged M (1, 17, 250, 1000 rows); N from 128 to 2048; β=0 everywhere;
- random per-group α/β.

The C++ example's own verification prints `Disposition: Failed` on the two 2048×5120×8192 configurations when given scalar `--alpha=1 --beta=0.5`, even though every group line reports `Status: 0`. The same shapes pass with its default random α/β. This is the C++ example's check, not a kernel difference.

## What it took to reach parity

Each item below is a measured step on the 2048×5120×8192 or K=512 benchmarks. The `mosaicist diff` tool compared the C++ kernel's PTX (via `cuobjdump -ptx`) against this port's, and pointed at the first two issues.

| Finding | Fix | Effect |
|---|---|---|
| `make_fragment_like` on a swizzled smem partition copies its stride order. The register-sourced WGMMA packs A pairs assuming the canonical order, so each output row received two neighbouring rows' values. | Build the register fragment with `tiled_mma.make_fragment_A` | Wrong results → bit-exact |
| The C++ kernel commits one wgmma per k-block and keeps 3 in flight. The first port drained to 0 after every k-tile. | Port the C++ schedule: loads 2 k-blocks ahead, conversion 1 ahead, `wait_group(3)`, stage released one k-tile late | Needed to get anything from the fixes below |
| The fp8 register fragment is lowered one byte per register. Packing pairs back together cost about 12 `PRMT` per k-block plus a stack depot. | 16-bit smem loads of each fp8 pair, then an explicit PTX sequence (`cvt.rn.f16x2.e5m2x2` → f32 → `cvt.rn.bf16x2.f32` → `mul.rn.bf16x2`), matching C++'s `F2FP/HADD2/HMUL2` | 2.94 → 2.44 ms |
| `recast_tensor` on the *swizzled* smem view applies the swizzle in the new element size's units, silently permuting K within a row. Tests where one operand is constant along K cannot see this. | Recast only the register fragment. `debug_patterns.py` now includes the both-operands-vary-along-K case. | Wrong results → bit-exact |
| Swizzled smem addresses were recomputed on every load (shift/and/xor, plus stage multiply). | Per-thread swizzled offsets computed once from the layout; loads use `[base + stage*8K + imm]`. The shortcut is exact because the stage stride and pair deltas miss the swizzle bits; `_precompute_a` checks this at trace time. | 2.44 → 2.25 ms |
| The C-fragment prefetch wrote registers, so the mainloop's first `wgmma.fence` waited for it on every tile. | `cp.async` the C tile into shared memory and read it in the epilogue | small-K β≠0: 2.35 → 1.95 ms |
| The group search ran at the start of every tile in 9 warps: a gmem shape load, prefix sums, and runtime divisions. | Stage the shape table in shared memory; a dedicated scheduler warp publishes tile records through a `PipelineAsync` ring, as the C++ kernel's scheduler warp does | small-K: 1.24× → 1.05× of C++ |
| TMA with a gmem descriptor pointer, issued inside `cute.copy`'s elect-one block, produced an illegal `@P0 R2UR` (error 715) once the loop structure changed. | NVVM TMA op with the `elect_sync` predicate (the upstream grouped example's workaround) | crash → correct |
| The DSL finds loop-carried state by scanning the loop body. A `PipelineState.advance()` inside a helper method was silently lost, so the loop re-read the same record forever. | Advance the state in the kernel body | hang → correct |

## Not ported (yet)

- **int4 variants.** `69_hopper_int4_bf16_grouped_gemm.cu` needs a host-side reordered int4 layout and a shift-based converter. `69_hopper_int4_fp8_grouped_gemm.cu` needs a scale-fused lookup-table conversion. The C++ baselines for both build and run with the header shim above.
- **Zero-points.** Neither is the C++ grouped example: its README lists them as unsupported.
- **Remaining 4–5% on many small-K groups.** C++ loads C with TMA through an epilogue-load warp, and its stage count differs by one. `mosaicist diff` lists these as its remaining rows, together with its `ldmatrix`/TMA-rank rows.
