"""Benchmark the masked MoE path against a dense bf16 baseline (run on an sm_90a GPU).

Reports each Pallas kernel separately and the fused path, next to `jnp.einsum` in
bf16 over the same shapes. The baseline is the honest thing to beat: it is what you
would write without a custom kernel, and it does no quantization, so the fp8 path has
to pay for its own quantize/dequantize work out of the tensor-core speedup.

Rows past `masked_m` are skipped by the kernels but not by the baseline, so the
default case runs every expert full (masked_m = m) to keep the comparison fair.
"""

import argparse
import statistics
import time

import jax
import jax.numpy as jnp

from masked_gemm import GemmConfig, masked_grouped_gemm
from moe import make_moe_inputs, moe_masked
from quantize_kernels import quantize_blockwise_pallas, silu_mul_quantize_pallas


def bench(fn, *args, iters=50, warmup=10):
    """Median wall-clock ms over `iters` runs, after warmup."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def dense_baseline(hidden, w1, w2, alpha1, alpha2):
    gateup = jnp.einsum("lmk,lnk->lmn", hidden, w1).astype(jnp.bfloat16) * alpha1.reshape(-1, 1, 1)
    gate, up = jnp.split(gateup.astype(jnp.float32), 2, axis=-1)
    act = (jax.nn.sigmoid(gate) * gate * up).astype(jnp.bfloat16)
    out = jnp.einsum("lmn,lkn->lmk", act, w2).astype(jnp.bfloat16)
    return out * alpha2.reshape(-1, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--rows", type=int, default=512, help="per-expert capacity m")
    ap.add_argument("--hidden", type=int, default=2048, help="k")
    ap.add_argument("--inter", type=int, default=1024, help="n (the FFN's inner size)")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    l, m, k, n = args.experts, args.rows, args.hidden, args.inter

    masked_m = jnp.full((l,), m, jnp.int32)
    kw = make_moe_inputs(jax.random.key(0), l, m, k, n, masked_m)
    gs, mask = kw["input_global_scale"], kw["masked_m"]

    a_q, a_sf = jax.block_until_ready(quantize_blockwise_pallas(kw["hidden"], gs, mask))
    gemm1_alpha = kw["w1_alpha"] / (gs * kw["w1_gs"])
    gateup = jax.block_until_ready(
        masked_grouped_gemm(a_q, a_sf, kw["w1_q"], kw["w1_sf"], gemm1_alpha, mask, GemmConfig())
    )
    d_q, d_sf = jax.block_until_ready(silu_mul_quantize_pallas(gateup, kw["a2_global_scale"], mask))
    gemm2_alpha = kw["w2_alpha"] / (kw["a2_global_scale"] * kw["w2_gs"])

    flops1 = 2 * l * m * (2 * n) * k
    flops2 = 2 * l * m * k * n
    stages = [
        ("quantize hidden", lambda: quantize_blockwise_pallas(kw["hidden"], gs, mask), 0),
        ("gemm1 (l,m,k)x(l,2n,k)",
         lambda: masked_grouped_gemm(a_q, a_sf, kw["w1_q"], kw["w1_sf"], gemm1_alpha, mask, GemmConfig()),
         flops1),
        ("silu_and_mul + quantize",
         lambda: silu_mul_quantize_pallas(gateup, kw["a2_global_scale"], mask), 0),
        ("gemm2 (l,m,n)x(l,k,n)",
         lambda: masked_grouped_gemm(d_q, d_sf, kw["w2_q"], kw["w2_sf"], gemm2_alpha, mask, GemmConfig()),
         flops2),
    ]

    print(f"l={l} m={m} k={k} n={n}   (fp8 e4m3 data, e4m3 block scales, BLOCK_K=128)\n")
    print(f"{'stage':<26}{'ms':>9}{'TFLOP/s':>10}")
    total = 0.0
    for label, fn, fl in stages:
        ms = bench(fn, iters=args.iters)
        total += ms
        tf = f"{fl / (ms * 1e-3) / 1e12:8.1f}" if fl else " " * 8
        print(f"{label:<26}{ms:9.3f}{tf:>10}")
    print(f"{'-' * 45}")

    fused = jax.jit(lambda **kws: moe_masked(**kws))
    ms_fused = bench(lambda: fused(**kw), iters=args.iters)
    print(f"{'moe_masked (end to end)':<26}{ms_fused:9.3f}{(flops1 + flops2) / (ms_fused * 1e-3) / 1e12:10.1f}")
    print(f"{'  sum of stages above':<26}{total:9.3f}")

    w1 = jax.random.normal(jax.random.key(1), (l, 2 * n, k), jnp.float32).astype(jnp.bfloat16)
    w2 = jax.random.normal(jax.random.key(2), (l, k, n), jnp.float32).astype(jnp.bfloat16)
    base = jax.jit(dense_baseline)
    ms_base = bench(base, kw["hidden"], w1, w2, kw["w1_alpha"], kw["w2_alpha"], iters=args.iters)
    print(f"{'bf16 einsum baseline':<26}{ms_base:9.3f}{(flops1 + flops2) / (ms_base * 1e-3) / 1e12:10.1f}")
    print(f"\nspeedup over bf16 baseline: {ms_base / ms_fused:.2f}x")


if __name__ == "__main__":
    main()
