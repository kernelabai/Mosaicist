"""Reference capture: FlashInfer's full CuTeDSL masked MoE path.

quantize -> GEMM1 -> silu_and_mul+quantize -> GEMM2, the whole thing this port is a
port of. Runs under the torch/flashinfer virtualenv.
"""

import torch
from flashinfer import silu_and_mul_scaled_nvfp4_experts_quantize
from flashinfer.fp4_quantization import scaled_fp4_grouped_quantize
from flashinfer.gemm import grouped_gemm_nt_masked

L, M, K, N = 8, 512, 2048, 1024
GEMM_KW = dict(ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
               c_dtype="bfloat16", sf_vec_size=16)


def build():
    dev = "cuda"
    hidden = torch.randn(L, M, K, device=dev, dtype=torch.bfloat16)
    w1 = torch.randn(L, 2 * N, K, device=dev, dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(L, K, N, device=dev, dtype=torch.bfloat16) * 0.05
    masked_m = torch.full((L,), M, device=dev, dtype=torch.int32)

    gs = lambda t: (448.0 * 6.0) / t.abs().amax(dim=(-2, -1)).float().clamp(min=1e-6)
    in_gs, w1_gs, w2_gs = gs(hidden), gs(w1), gs(w2)
    w1_q, w1_sf = scaled_fp4_grouped_quantize(
        w1, torch.full((L,), 2 * N, device=dev, dtype=torch.int32), w1_gs)
    w2_q, w2_sf = scaled_fp4_grouped_quantize(
        w2, torch.full((L,), K, device=dev, dtype=torch.int32), w2_gs)

    gateup_buf = torch.empty(L, M, 2 * N, device=dev, dtype=torch.bfloat16)
    gateup = gateup_buf.permute(1, 2, 0)
    out = torch.empty(L, M, K, device=dev, dtype=torch.bfloat16).permute(1, 2, 0)

    a_q0, a_sf0 = scaled_fp4_grouped_quantize(hidden, masked_m, in_gs)
    grouped_gemm_nt_masked((a_q0, a_sf0), (w1_q, w1_sf), gateup, masked_m, **GEMM_KW)
    a2_gs = gs(gateup_buf)

    def run():
        a_q, a_sf = scaled_fp4_grouped_quantize(hidden, masked_m, in_gs)
        grouped_gemm_nt_masked((a_q, a_sf), (w1_q, w1_sf), gateup, masked_m, **GEMM_KW)
        d_q, d_sf = silu_and_mul_scaled_nvfp4_experts_quantize(gateup_buf, masked_m, a2_gs)
        grouped_gemm_nt_masked((d_q, d_sf), (w2_q, w2_sf), out, masked_m, **GEMM_KW)
        return out

    return run, ()
