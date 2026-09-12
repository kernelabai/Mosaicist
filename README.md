# Mosaicist

Converge a Pallas Mosaic GPU kernel on a CuTeDSL reference kernel. The reference is lowered to PTX. A naive Pallas translation is checked for numerical equivalence, then edited until its runtime and accuracy match the original. PTX/SASS fingerprint diffs choose each edit. The full design is in [DESIGN.md](DESIGN.md).

## Status

| Piece | Design § | State |
|---|---|---|
| PTX parser, CFG, loops, warp roles | §7 | done, tested |
| Fingerprint layers L0–L5, alignment, diff | §7 | done, tested on fixtures |
| Diagnose (diff rows → ranked, tagged fixes) | §7–8 | done, initial rule set |
| Numerics gate, input suites, guard bands | §6 | done, tested |
| Timing statistics, noise floor, acceptance rule, beam | §8–9 | done, tested |
| Bundle schema | §4 | done |
| CUPTI launch records + device timing (`bench/cupti_trace.py`), `mosaicist time` | §4, §9 | done; used on an H100 |
| Capture harnesses for one kernel pair (`experiments/hopper_gemm/`) | §4 | prototype; generalizing them into workers is next |
| `ncu` profiling | §9 | not started; needs profiling permissions on the host |
| Naive translator, knob tuner, LLM rewriter, loop driver | §5, §8 | not started |

`tests/fixtures/*.ptx` are hand-abbreviated examples. `tests/fixtures/real/` holds real compiler output captured on an H100.

## First H100 run (2026-09-11)

CuTeDSL `hopper/kernel/dense_gemm.py` (CUTLASS 4.7.1) vs Pallas `hopper_matmul_mgpu.py` (JAX 0.11.1). Both run fp16 8192³ with a 128×256 CTA tile and a 2×1 cluster, on an H100 PCIe with unlocked clocks.

| | median | vs ref | numerics |
|---|---|---|---|
| CuTeDSL reference | 2477.7 µs | — | — |
| Pallas as shipped | 2457.3 µs | 0.992× | 100.0% bit-identical |
| Pallas + `delay_release=1` (Diagnose's P3 fix) | 2486.4 µs | 1.004× | 100.0% bit-identical |

The noise floor from three A/A runs of the reference is 5.8%, so every row counts as *equal*. The P3 fix closes its fingerprint row, raises mainloop alignment from 0.64 to 0.73, and lowers D from 0.497 to 0.438. It has no runtime effect this noise floor can resolve. Real PTX surfaced bugs the hand-written fixtures hid, all now fixed and covered by `tests/test_real_ptx.py`:

- Scoped inline-asm labels (`LAB_WAIT`/`DONE`) were resolved across scopes, so no loops were found.
- A prefetch prologue was mistaken for warp specialization.
- Persistent tile loops hid warp specialization and were mislabeled as producer loops.

Reproduce with `experiments/hopper_gemm/{gen_inputs,ref_cutedsl,cand_pallas}.py`. Each script's docstring has its invocation. Use one pinned `ptxas` (≥ 12.9, for PTX ISA 8.8) for both kernels.

## Ports

| port | direction | status |
|---|---|---|
| [`ports/hopper_mixed_dtype_grouped_gemm`](ports/hopper_mixed_dtype_grouped_gemm/README.md) | CUTLASS C++ → CuTeDSL | CUTLASS example 69 (fp8 x bf16 mixed-input grouped GEMM): bit-exact, at parity with C++ on the example's benchmark configurations; int4 variants not yet ported |
| [`experiments/flashinfer-megamoe-sm90a`](experiments/flashinfer-megamoe-sm90a/README.md) | FlashInfer CuTeDSL → Pallas | FlashInfer/SGLang MegaMoE masked MoE, as a Hopper analog (fp8 with 128-element block scales instead of NVFP4): full path in Pallas, 12/12 tests, GEMM bit-exact, end to end agrees to bf16. 0.75x a dense bf16 baseline — the gap is measured and explained |
| [`experiments/flashinfer-megamoe-sm100a`](experiments/flashinfer-megamoe-sm100a/README.md) | FlashInfer CuTeDSL → Pallas | the same MoE on Blackwell, keeping NVFP4 and `tcgen05.mma.kind.block_scale`. **Verified on a B200**: GEMM bit-exact (21/21 tests), end to end at rms ~2e-3. 83.5 µs end to end with GEMMs at 1660/1136 TFLOP/s; 0.75x a dense bf16 baseline, the gap being quantization the baseline never pays |

## Known limitations

- Diagnose's discrepancy rows are compiler-agnostic, but its suggested fixes name Pallas Mosaic GPU levers. Diffing two non-Pallas kernels (as the port below does) gives correct rows with inapplicable advice.
- `L1.ldmatrix` and similar inventory rows are kernel-wide, so they do not say which loop (mainloop vs epilogue) the difference is in.
- The convergence loop assumes both kernels run on the same GPU. The Blackwell port was written with no such hardware, using Mosaic's lowering as the only oracle; that was enough to find real constraints and the kernels needed no correctness fixes when a B200 arrived, but it could not have proven correctness on its own.

## Usage

```bash
pip install -e '.[dev]'        # or run from source: PYTHONPATH=src python -m mosaicist.cli ...
python -m pytest

# Diff a candidate against the reference and rank fixes
mosaicist diff tests/fixtures/hopper_gemm_ref.ptx tests/fixtures/hopper_gemm_v0.ptx \
  --ref-log tests/fixtures/hopper_gemm_ref.ptxas.log --cand-log tests/fixtures/hopper_gemm_v0.ptxas.log

# Inspect one kernel's fingerprint (PTX file or bundle directory)
mosaicist fingerprint tests/fixtures/hopper_gemm_ref.ptx

# Numerics gate on saved outputs (float64 oracle)
mosaicist check ref.npy cand.npy oracle.npy --fmt bf16
```

Getting PTX from real kernels:

```bash
# CuTeDSL
CUTE_DSL_KEEP_PTX=1 CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=runs/ref/ python my_kernel.py
# Pallas Mosaic GPU
MOSAIC_GPU_DUMP_PTX=1 MOSAIC_GPU_DUMP_PTXAS=1 MOSAIC_GPU_DUMP_SASS=1 MOSAIC_GPU_DUMP_TO=runs/cand/ python my_pallas.py
```

## Layout

```
src/mosaicist/
  ptx/        parse.py · cfg.py · ops.py · features.py · align.py · diff.py
  diagnose.py rule catalog: discrepancy rows -> fixes (knob / rewrite / low-level / gap / investigate)
  verify/     formats.py (ULP math for f32/f16/bf16/fp8) · numerics.py · suites.py · guards.py
  bench/      stats.py
  converge/   accept.py
  bundle.py   capture schema shared by both compilers
  cli.py
tests/        pytest suite + PTX fixtures
```
