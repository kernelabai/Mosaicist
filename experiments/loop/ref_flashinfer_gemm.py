"""Capture module for the reference: FlashInfer's CuTeDSL masked grouped GEMM.

Runs under the torch/flashinfer virtualenv. `build()` returns the callable and its
arguments; `mosaicist capture --compiler cutedsl` sets CUTE_DSL_KEEP_PTX before this
imports anything, so the kernel's PTX lands in the bundle.
"""

import torch
from flashinfer.fp4_quantization import scaled_fp4_grouped_quantize
from flashinfer.gemm import grouped_gemm_nt_masked

L, M, K, N = 8, 512, 2048, 2048
GEMM_KW = dict(ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
               c_dtype="bfloat16", sf_vec_size=16)


def build():
    dev = "cuda"
    a = torch.randn(L, M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(L, N, K, device=dev, dtype=torch.bfloat16) * 0.05
    masked_m = torch.full((L,), M, device=dev, dtype=torch.int32)
    full_n = torch.full((L,), N, device=dev, dtype=torch.int32)

    gs = lambda t: (448.0 * 6.0) / t.abs().amax(dim=(-2, -1)).float().clamp(min=1e-6)
    a_q, a_sf = scaled_fp4_grouped_quantize(a, masked_m, gs(a))
    b_q, b_sf = scaled_fp4_grouped_quantize(b, full_n, gs(b))
    # c_major='n' wants (l, m, n) allocated and viewed as (m, n, l)
    out = torch.empty(L, M, N, device=dev, dtype=torch.bfloat16).permute(1, 2, 0)

    def run():
        grouped_gemm_nt_masked((a_q, a_sf), (b_q, b_sf), out, masked_m, **GEMM_KW)
        return out

    return run, ()
