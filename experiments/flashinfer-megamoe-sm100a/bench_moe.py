"""Benchmark the Blackwell masked MoE path against a dense bf16 baseline (needs sm_100a).

Device time from CUPTI activity records, not wall clock: these kernels run well inside
the host round-trip, which would otherwise report the same figure for all of them.

The dense bf16 `jnp.einsum` baseline does no quantization at all, so it is a speed
comparison and not an accuracy one -- the fp4 path has to pay for its own quantize work
out of the tensor-core speedup.
"""

import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace  # noqa: E402

from masked_gemm import GemmConfig, masked_grouped_gemm  # noqa: E402
from moe import make_moe_inputs, moe_masked  # noqa: E402
from nvfp4 import to_mma_scale_layout  # noqa: E402
from quantize_kernels import quantize_nvfp4_pallas, silu_mul_quantize_nvfp4_pallas  # noqa: E402

_retile = jax.vmap(to_mma_scale_layout)


def device_us(fn, reps=30, warmup=5):
    """Per-rep device time (us) of every kernel `fn` launches, summed by name."""
    for _ in range(warmup):
        jax.block_until_ready(fn())
    with KernelTrace() as tr:
        for _ in range(reps):
            jax.block_until_ready(fn())
    per_name: dict[str, list[float]] = {}
    for r in tr.records:
        per_name.setdefault(r.name, []).append(r.duration_us)
    return {name: sum(times) / reps for name, times in per_name.items()}


def dense_baseline(hidden, w1, w2, alpha1, alpha2):
    gateup = jnp.einsum("lmk,lnk->lmn", hidden, w1).astype(jnp.bfloat16) * alpha1.reshape(-1, 1, 1)
    gate, up = jnp.split(gateup.astype(jnp.float32), 2, axis=-1)
    act = (jax.nn.sigmoid(gate) * gate * up).astype(jnp.bfloat16)
    return jnp.einsum("lmn,lkn->lmk", act, w2).astype(jnp.bfloat16) * alpha2.reshape(-1, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--rows", type=int, default=512, help="per-expert capacity m")
    ap.add_argument("--hidden", type=int, default=2048, help="k")
    ap.add_argument("--inter", type=int, default=1024, help="n")
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    l, m, k, n = args.experts, args.rows, args.hidden, args.inter

    masked_m = jnp.full((l,), m, jnp.int32)
    kw, _ = make_moe_inputs(jax.random.key(0), l, m, k, n, masked_m)
    gs, mask = kw["input_global_scale"], kw["masked_m"]

    jq, jsq = jax.jit(quantize_nvfp4_pallas), jax.jit(silu_mul_quantize_nvfp4_pallas)
    jg = jax.jit(masked_grouped_gemm, static_argnums=6)
    jr = jax.jit(_retile)
    a_q, a_sf = jax.block_until_ready(jq(kw["hidden"], gs, mask))
    g1_alpha = kw["w1_alpha"] / (gs * kw["w1_gs"])
    gateup = jax.block_until_ready(
        jg(a_q, jr(a_sf), kw["w1_q"], kw["w1_sf"], g1_alpha, mask, GemmConfig()))
    d_q, d_sf = jax.block_until_ready(jsq(gateup, kw["a2_global_scale"], mask))
    g2_alpha = kw["w2_alpha"] / (kw["a2_global_scale"] * kw["w2_gs"])

    flops1, flops2 = 2 * l * m * (2 * n) * k, 2 * l * m * k * n

    print(f"l={l} m={m} k={k} n={n}   NVFP4: e2m1 data, e4m3 scales per 16 elements")
    print(f"device time, {args.reps} reps\n")
    print(f"{'stage':<30}{'us':>9}{'TFLOP/s':>10}")
    total = 0.0
    for label, fn, fl in [
        ("quantize hidden", lambda: jq(kw["hidden"], gs, mask), 0),
        ("  + retile scales", lambda: jr(a_sf), 0),
        ("gemm1 (l,m,k)x(l,2n,k)",
         lambda: jg(a_q, jr(a_sf), kw["w1_q"], kw["w1_sf"], g1_alpha, mask, GemmConfig()), flops1),
        ("silu_and_mul + quantize", lambda: jsq(gateup, kw["a2_global_scale"], mask), 0),
        ("gemm2 (l,m,n)x(l,k,n)",
         lambda: jg(d_q, jr(d_sf), kw["w2_q"], kw["w2_sf"], g2_alpha, mask, GemmConfig()), flops2),
    ]:
        us = sum(device_us(fn, reps=args.reps).values())
        total += us
        tf = f"{fl / (us * 1e-6) / 1e12:8.1f}" if fl else " " * 8
        print(f"{label:<30}{us:9.1f}{tf:>10}")
    print("-" * 49)

    fused = jax.jit(moe_masked)
    per_kernel = device_us(lambda: fused(**kw), reps=args.reps)
    us_fused = sum(per_kernel.values())
    print(f"{'moe_masked (end to end)':<30}{us_fused:9.1f}"
          f"{(flops1 + flops2) / (us_fused * 1e-6) / 1e12:10.1f}")
    print(f"{'  sum of stages above':<30}{total:9.1f}")
    for name, us in sorted(per_kernel.items(), key=lambda kv: -kv[1])[:4]:
        print(f"      {name[:42]:<42}{us:8.1f}")

    w1 = jax.random.normal(jax.random.key(1), (l, 2 * n, k), jnp.float32).astype(jnp.bfloat16)
    w2 = jax.random.normal(jax.random.key(2), (l, k, n), jnp.float32).astype(jnp.bfloat16)
    base = jax.jit(dense_baseline)
    us_base = sum(device_us(
        lambda: base(kw["hidden"], w1, w2, kw["w1_alpha"], kw["w2_alpha"]), reps=args.reps).values())
    print("-" * 49)
    print(f"{'bf16 einsum baseline':<30}{us_base:9.1f}"
          f"{(flops1 + flops2) / (us_base * 1e-6) / 1e12:10.1f}")
    print(f"\nspeedup over bf16 baseline: {us_base / us_fused:.2f}x")


if __name__ == "__main__":
    main()
