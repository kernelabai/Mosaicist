"""Correctness tests for the masked block-scaled grouped GEMM (run on an sm_90a GPU)."""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from masked_gemm import GemmConfig, masked_grouped_gemm
from quant import BLOCK_K, FP8_DTYPE, masked_grouped_gemm_reference, quantize_blockwise


def exact_operands(key, l, rows, k):
    """Quantized values and scales that make the whole computation exact in fp32:
    small integers times powers of two, so the reference and the kernel must agree bit for bit."""
    kq, ks = jax.random.split(key)
    q = jax.random.randint(kq, (l, rows, k), -4, 5).astype(jnp.float32).astype(FP8_DTYPE)
    sf = jnp.exp2(jax.random.randint(ks, (l, k // BLOCK_K, rows), -2, 3).astype(jnp.float32)).astype(FP8_DTYPE)
    return q, sf


def run_case(l=3, m=256, n=256, k=512, exact=True, seed=0, config=GemmConfig()):
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    if exact:
        a_q, a_sf = exact_operands(k1, l, m, k)
        b_q, b_sf = exact_operands(k2, l, n, k)
        alpha = jnp.exp2(jax.random.randint(k3, (l,), -1, 2)).astype(jnp.float32)
    else:
        a = jax.random.normal(k1, (l, m, k), jnp.float32)
        b = jax.random.normal(k2, (l, n, k), jnp.float32)
        gs = jnp.ones((l,), jnp.float32) * 256.0
        a_q, a_sf = quantize_blockwise(a, gs)
        b_q, b_sf = quantize_blockwise(b, gs)
        alpha = jax.random.uniform(k3, (l,), jnp.float32, 0.5, 2.0)
    masked_m = jnp.array([m, max(1, m // 2 - 3), 1][:l] + [m] * max(0, l - 3), jnp.int32)

    ones = jnp.ones((l,), jnp.float32)
    ref = masked_grouped_gemm_reference(a_q, a_sf, ones, b_q, b_sf, ones, alpha, masked_m)
    out = masked_grouped_gemm(a_q, a_sf, b_q, b_sf, alpha, masked_m, config)

    rows = np.arange(m)[None, :, None] < np.asarray(masked_m)[:, None, None]
    got = np.where(rows, np.asarray(out, np.float32), 0.0)
    want = np.where(rows, np.asarray(ref, np.float32), 0.0)
    return got, want


def main():
    failures = 0
    for label, kwargs in [
        ("exact 3x256x256x512", dict(exact=True)),
        ("exact 2x128x256x1024", dict(exact=True, l=2, m=128, n=256, k=1024)),
        ("exact 5x384x128x256", dict(exact=True, l=5, m=384, n=128, k=256)),
        # k=384 is 3 K blocks: exercises the unpaired tail block of the mainloop
        ("exact 3x256x256x384 (odd K blocks)", dict(exact=True, k=384)),
        ("exact 2x128x192x128 (1 K block)", dict(exact=True, l=2, m=128, n=192, k=128)),
        ("random floats 3x256x256x512", dict(exact=False)),
        ("random floats, stages=2", dict(exact=False, config=GemmConfig(stages=2))),
    ]:
        got, want = run_case(**kwargs)
        exact = kwargs.get("exact", True)
        if exact:
            ok = np.array_equal(got, want)
            detail = f"max|diff|={np.abs(got - want).max():g}"
        else:
            denom = np.maximum(np.abs(want).max(), 1e-6)
            rel = np.abs(got - want).max() / denom
            ok = rel < 5e-3  # bf16 output rounding
            detail = f"max rel err={rel:.2e}"
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {label:<36} {'bit-exact' if exact and ok else detail}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
