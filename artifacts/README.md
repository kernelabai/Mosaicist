# Captured artifacts

Everything here needed a GPU to produce, and both machines that produced it have been
released. Kept so the fingerprint, diff and Diagnose paths stay exercisable — and so the
claims in the experiment READMEs remain checkable — without hardware.

Regenerable things were left behind: `.npy` outputs (up to 67 MB each), Mosaic's
`mlir-passes.log` dumps (6–40 MB each), and the virtualenvs. What is here is ~2 MB.

## b200/ — NVIDIA B200, sm_100a, CUDA 12.8, driver 570.148.08

| path | what it is |
|---|---|
| `captures/ref/` | FlashInfer's CuTeDSL masked grouped GEMM: PTX, SASS, and 30 timing samples (13.4 µs median). Needs a B200 *and* a working flashinfer + nvidia-cutlass-dsl install to regenerate. |
| `captures/refmoe/` | The same reference's full MoE path — quantize → GEMM1 → silu+quantize → GEMM2 (46.0 µs median). |
| `captures/cand_best/` | The Pallas candidate at its best setting, for diffing against the reference. |
| `captures/ref_cutedsl.ptx`, `cand_pallas.ptx`, `cand_w1.ptx` | The bare PTX pair the first hand diffs were run on. |
| `transcripts/` | Every convergence run's `converge.json`, the `port` pipeline's `contract.json` and `report.md`, and the MoE run's console log. |
| `v0_sm_100a.py` | A generated v0 kernel, as emitted by the translator. |
| `env/` | `uv pip freeze` for both virtualenvs, `nvidia-smi`, `nvcc --version`. |

Key versions: jax/jaxlib 0.11.1, torch 2.8.0+cu128, flashinfer-python 0.6.18.post1,
nvidia-cutlass-dsl 4.7.1.

## h100/ — NVIDIA H100 PCIe, sm_90a

| path | what it is |
|---|---|
| `runs/ref*/` | The CuTeDSL Hopper dense GEMM reference, including the two extra A/A runs the noise floor was measured from. |
| `runs/cand*/` | Pallas candidates for the same problem, one per `delay_release` setting. |
| `runs/ex69/cpp/` | CUTLASS C++ example 69: PTX and ptxas logs. Regenerating these means building CUTLASS. |
| `runs/ex69/cute/` | The CuTeDSL port's SASS at each stage of its optimization — `v3`, `v4`, `v5`, `own`, `own16`. This is the record of how that port reached parity. |
| `runs/ex69/port/` | The port's final PTX and ptxas log. |
| `env/` | Package freezes for `venv-jax` and `venv-cute`, plus GPU and toolchain versions. |

Large `.sass` and `.ptx` files are gzipped; the tools read them after `gunzip`.

## Using them

The bundles are complete, so everything except running a kernel still works:

```bash
mosaicist fingerprint artifacts/b200/captures/ref
mosaicist diff artifacts/b200/captures/ref artifacts/b200/captures/cand_best
```

That diff prints D = 0.386 and names the differences the experiment READMEs discuss —
192 threads against 128, 226 KB of shared memory against 107, and a persistent
`grid=[1,1,148]` against a tiled one.
