"""End-to-end test for the masked MoE path (run on an sm_90a GPU).

Compares `moe.moe_masked` (three Pallas kernels) against `quant.moe_reference` (the
same arithmetic in fp32 JAX). Only rows below each expert's `masked_m` are compared:
the kernels skip whole tiles that start past the mask, so the rows above it are
whatever was in the output buffer, and within a partially masked tile the kernel
computes real values where the reference writes zeros. Neither affects a valid row --
every stage is row-independent, so row m only ever depends on row m.

Agreement is not bit-exact end to end, and can't be: the intermediate quantization
turns a 1-ulp difference in GEMM1's bf16 output into a whole e4m3 step on that
element. The bound below is what two honest implementations of this scheme differ by.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from moe import make_moe_inputs, moe_masked
from quant import moe_reference


def run_case(l=3, m=256, k=512, n=256, seed=0):
    masked_m = jnp.array([m, max(1, m // 2 - 3), 1][:l] + [m] * max(0, l - 3), jnp.int32)
    kwargs = make_moe_inputs(jax.random.key(seed), l, m, k, n, masked_m)

    got = np.asarray(moe_masked(**kwargs), np.float32)
    want = np.asarray(moe_reference(**kwargs), np.float32)

    rows = np.arange(m)[None, :, None] < np.asarray(masked_m)[:, None, None]
    return np.where(rows, got, 0.0), np.where(rows, want, 0.0)


def main():
    failures = 0
    for label, kwargs in [
        ("3 experts, m=256 k=512 n=256", dict()),
        ("2 experts, m=128 k=256 n=512", dict(l=2, m=128, k=256, n=512)),
        ("5 experts, m=384 k=512 n=128", dict(l=5, m=384, k=512, n=128)),
    ]:
        got, want = run_case(**kwargs)
        scale = max(np.abs(want).max(), 1e-6)
        rel = np.abs(got - want).max() / scale
        rms = np.sqrt(np.mean((got - want) ** 2)) / scale
        ok = rel < 5e-2 and rms < 5e-3
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {label:<32} max rel err={rel:.2e}  rms={rms:.2e}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
