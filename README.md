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
| Capture workers (CuTeDSL, Pallas), CUPTI timing, `ncu` | §4, §9 | not started; needs a Linux host with an sm_90a/sm_100a GPU |
| Naive translator, knob tuner, LLM rewriter, loop driver | §5, §8 | not started |

The GEMM fixtures in `tests/fixtures/` are hand-abbreviated PTX modeled on the two compilers' output, not real compiler output. Checking the fingerprint rules against real CuTeDSL and Mosaic GPU PTX is the first job once a GPU host is available.

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
