"""Numerical tests for the Blackwell kernels. Needs an sm_100a device.

On anything else this exits 0 after saying so -- the checks that do run without
Blackwell are `test_nvfp4.py` (the NVFP4 arithmetic and the MMA scale tiling) and
`check_lowering.py` (that every kernel builds for sm_100a). Nothing here has been run.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from masked_gemm import GemmConfig, masked_grouped_gemm
from moe import make_moe_inputs, moe_masked
from nvfp4 import (FP4_DTYPE, SF_DTYPE, SF_VEC_SIZE, global_scale_for,
                   masked_grouped_gemm_reference, moe_reference, quantize_nvfp4,
                   to_mma_scale_layout)

_retile = jax.vmap(to_mma_scale_layout)


def is_blackwell():
    cc = getattr(jax.devices()[0], "compute_capability", "0.0")
    return tuple(int(p) for p in cc.split(".")) >= (10, 0)


def exact_operands(key, l, rows, k):
    """Operands that make the whole computation exact in fp32: small integers times
    powers of two, all on the e2m1 grid, so kernel and reference must agree bit for bit."""
    kq, ks = jax.random.split(key)
    q = jax.random.choice(kq, jnp.array([-4., -3., -2., -1., 0., 1., 2., 3., 4., 6.]),
                          (l, rows, k)).astype(FP4_DTYPE)
    sf = jnp.exp2(jax.random.randint(ks, (l, rows, k // SF_VEC_SIZE), -2, 3)
                  .astype(jnp.float32)).astype(SF_DTYPE)
    return q, sf


def test_gemm(l=2, m=256, n=256, k=512, exact=True, seed=0, config=GemmConfig()):
    key = jax.random.split(jax.random.key(seed), 3)
    if exact:
        a_q, a_sf = exact_operands(key[0], l, m, k)
        b_q, b_sf = exact_operands(key[1], l, n, k)
        alpha = jnp.exp2(jax.random.randint(key[2], (l,), -1, 2)).astype(jnp.float32)
    else:
        gs = jnp.full((l,), 64.0, jnp.float32)
        a_q, a_sf = quantize_nvfp4(jax.random.normal(key[0], (l, m, k), jnp.float32), gs)
        b_q, b_sf = quantize_nvfp4(jax.random.normal(key[1], (l, n, k), jnp.float32), gs)
        alpha = jax.random.uniform(key[2], (l,), jnp.float32, 0.5, 2.0)
    masked_m = jnp.array([m, max(1, m // 2 - 3)][:l] + [m] * max(0, l - 2), jnp.int32)

    ones = jnp.ones((l,), jnp.float32)
    ref = masked_grouped_gemm_reference(a_q, a_sf, ones, b_q, b_sf, ones, alpha, masked_m)
    out = masked_grouped_gemm(a_q, _retile(a_sf), b_q, _retile(b_sf), alpha, masked_m, config)

    rows = np.arange(m)[None, :, None] < np.asarray(masked_m)[:, None, None]
    return (np.where(rows, np.asarray(out, np.float32), 0.0),
            np.where(rows, np.asarray(ref, np.float32), 0.0))


def test_moe(l=2, m=256, k=512, n=256, seed=0):
    masked_m = jnp.array([m, max(1, m // 2 - 3)][:l] + [m] * max(0, l - 2), jnp.int32)
    kernel_kw, ref_kw = make_moe_inputs(jax.random.key(seed), l, m, k, n, masked_m)
    got = np.asarray(moe_masked(**kernel_kw), np.float32)
    want = np.asarray(moe_reference(**ref_kw), np.float32)
    rows = np.arange(m)[None, :, None] < np.asarray(masked_m)[:, None, None]
    return np.where(rows, got, 0.0), np.where(rows, want, 0.0)


def main():
    if not is_blackwell():
        cc = getattr(jax.devices()[0], "compute_capability", "?")
        print(f"SKIP  every kernel test: needs sm_100a, this device is sm_{cc}")
        print("      run test_nvfp4.py and check_lowering.py instead")
        return 0

    failures = 0
    for label, kwargs in [
        ("exact 2x256x256x512", dict(exact=True)),
        ("exact 3x128x256x1024", dict(exact=True, l=3, m=128, n=256, k=1024)),
        ("random floats 2x256x256x512", dict(exact=False)),
        ("random floats, block_k=256", dict(exact=False, config=GemmConfig(block_k=256, stages=2))),
    ]:
        got, want = test_gemm(**kwargs)
        if kwargs.get("exact"):
            ok, detail = np.array_equal(got, want), f"max|diff|={np.abs(got - want).max():g}"
        else:
            rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-6)
            ok, detail = rel < 5e-3, f"max rel err={rel:.2e}"
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  gemm {label:<32} "
              f"{'bit-exact' if kwargs.get('exact') and ok else detail}")

    for label, kwargs in [("2 experts, m=256 k=512 n=256", dict()),
                          ("3 experts, m=128 k=512 n=512", dict(l=3, m=128, k=512, n=512))]:
        got, want = test_moe(**kwargs)
        scale = max(np.abs(want).max(), 1e-6)
        rel, rms = (np.abs(got - want).max() / scale,
                    np.sqrt(np.mean((got - want) ** 2)) / scale)
        ok = rel < 1e-1 and rms < 1e-2  # fp4 is coarse; an intermediate requantization
        failures += not ok               # turns one ulp of GEMM1 into a whole e2m1 step
        print(f"{'PASS' if ok else 'FAIL'}  moe  {label:<32} max rel err={rel:.2e}  rms={rms:.2e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
