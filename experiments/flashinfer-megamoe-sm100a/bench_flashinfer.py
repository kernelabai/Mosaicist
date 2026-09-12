"""Time FlashInfer's CuTeDSL masked MoE -- the kernel this experiment is a port of.

Runs in its own venv (torch + flashinfer + nvidia-cutlass-dsl), because the Pallas side
needs JAX. Device time comes from the same CUPTI activity records `bench_moe.py` uses,
so the two are measured the same way and the numbers can be put side by side.

Mirrors `flashinfer_cutedsl_moe_masked`: quantize -> GEMM1 -> silu_and_mul+quantize ->
GEMM2, at the shapes `bench_moe.py` reports.
"""

import argparse
import pathlib
import sys

import torch
from flashinfer import silu_and_mul_scaled_nvfp4_experts_quantize
from flashinfer.fp4_quantization import scaled_fp4_grouped_quantize
from flashinfer.gemm import grouped_gemm_nt_masked

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace  # noqa: E402

GEMM_KW = dict(ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
               c_dtype="bfloat16", sf_vec_size=16)


def device_us(fn, reps=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with KernelTrace() as tr:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    per_name = {}
    for r in tr.records:
        per_name.setdefault(r.name, []).append(r.duration_us)
    return {n: sum(t) / reps for n, t in per_name.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    l, m, k, n = a.experts, a.rows, a.hidden, a.inter
    dev = "cuda"

    hidden = torch.randn(l, m, k, device=dev, dtype=torch.bfloat16)
    w1 = torch.randn(l, 2 * n, k, device=dev, dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(l, k, n, device=dev, dtype=torch.bfloat16) * 0.05
    masked_m = torch.full((l,), m, device=dev, dtype=torch.int32)
    full_w1 = torch.full((l,), 2 * n, device=dev, dtype=torch.int32)
    full_w2 = torch.full((l,), k, device=dev, dtype=torch.int32)

    gs = lambda t: (448.0 * 6.0) / t.abs().amax(dim=(-2, -1)).float().clamp(min=1e-6)
    in_gs, w1_gs, w2_gs = gs(hidden), gs(w1), gs(w2)

    w1_q, w1_sf = scaled_fp4_grouped_quantize(w1, full_w1, w1_gs)
    w2_q, w2_sf = scaled_fp4_grouped_quantize(w2, full_w2, w2_gs)
    a_q, a_sf = scaled_fp4_grouped_quantize(hidden, masked_m, in_gs)
    # the kernel requires c_major='n': allocate (l, m, n) then view it as (m, n, l)
    gateup_buf = torch.empty(l, m, 2 * n, device=dev, dtype=torch.bfloat16)
    gateup = gateup_buf.permute(1, 2, 0)
    grouped_gemm_nt_masked((a_q, a_sf), (w1_q, w1_sf), gateup, masked_m, **GEMM_KW)
    a2_gs = gs(gateup_buf)
    d_q, d_sf = silu_and_mul_scaled_nvfp4_experts_quantize(gateup_buf, masked_m, a2_gs)
    out_buf = torch.empty(l, m, k, device=dev, dtype=torch.bfloat16)
    out = out_buf.permute(1, 2, 0)

    flops1, flops2 = 2 * l * m * (2 * n) * k, 2 * l * m * k * n

    def full_path():
        aq, asf = scaled_fp4_grouped_quantize(hidden, masked_m, in_gs)
        grouped_gemm_nt_masked((aq, asf), (w1_q, w1_sf), gateup, masked_m, **GEMM_KW)
        dq, dsf = silu_and_mul_scaled_nvfp4_experts_quantize(gateup_buf, masked_m, a2_gs)
        grouped_gemm_nt_masked((dq, dsf), (w2_q, w2_sf), out, masked_m, **GEMM_KW)

    print(f"FlashInfer CuTeDSL  l={l} m={m} k={k} n={n}   device time, {a.reps} reps\n")
    print(f"{'stage':<30}{'us':>9}{'TFLOP/s':>10}")
    total = 0.0
    for label, fn, fl in [
        ("quantize hidden", lambda: scaled_fp4_grouped_quantize(hidden, masked_m, in_gs), 0),
        ("gemm1 (l,m,k)x(l,2n,k)",
         lambda: grouped_gemm_nt_masked((a_q, a_sf), (w1_q, w1_sf), gateup, masked_m, **GEMM_KW),
         flops1),
        ("silu_and_mul + quantize",
         lambda: silu_and_mul_scaled_nvfp4_experts_quantize(gateup_buf, masked_m, a2_gs), 0),
        ("gemm2 (l,m,n)x(l,k,n)",
         lambda: grouped_gemm_nt_masked((d_q, d_sf), (w2_q, w2_sf), out, masked_m, **GEMM_KW),
         flops2),
    ]:
        us = sum(device_us(fn, reps=a.reps).values())
        total += us
        tf = f"{fl / (us * 1e-6) / 1e12:8.1f}" if fl else " " * 8
        print(f"{label:<30}{us:9.1f}{tf:>10}")
    print("-" * 49)
    us_full = sum(device_us(full_path, reps=a.reps).values())
    print(f"{'full path (end to end)':<30}{us_full:9.1f}"
          f"{(flops1 + flops2) / (us_full * 1e-6) / 1e12:10.1f}")
    print(f"{'  sum of stages above':<30}{total:9.1f}")


if __name__ == "__main__":
    main()
