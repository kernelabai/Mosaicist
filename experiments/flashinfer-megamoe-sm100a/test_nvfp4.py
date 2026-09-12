"""Tests for the NVFP4 semantics and the MMA scale layout.

These need no Blackwell GPU -- they are plain JAX, and they cover the two parts of the
port that are easiest to get wrong and cheapest to check: the quantization arithmetic,
and the `.scale_vec::1X` tiling that `async_copy_scales_to_tmem` expects. The kernels
themselves are checked by `test_gemm.py` / `test_moe.py`, which need sm_100a.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from nvfp4 import (FP4_MAX, SF_VEC_SIZE, dequantize_nvfp4, from_mma_scale_layout,
                   global_scale_for, moe_reference, quantize_nvfp4, silu_mul,
                   to_mma_scale_layout)

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label:<46} {detail}")


def test_exact_values_survive():
    """Values that are exactly representable must quantize with zero error.

    e2m1 holds {0, .5, 1, 1.5, 2, 3, 4, 6}; scaling a block by a power of two keeps
    every element on the grid, and the block scale itself is then exact in e4m3.
    """
    grid = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.5, -3.0, -6.0])
    x = jnp.tile(grid, (1, 4, 4))[:, :, :48]  # (1, 4, 48) = 3 blocks of 16
    gs = jnp.array([448.0 * FP4_MAX / 6.0], jnp.float32)  # makes the block scale 1.0
    q, sf = quantize_nvfp4(x, gs)
    back = dequantize_nvfp4(q, sf, gs)
    check("exactly-representable values round-trip",
          np.array_equal(np.asarray(back), np.asarray(x, np.float32)),
          f"max|diff|={np.abs(np.asarray(back) - np.asarray(x, np.float32)).max():g}")


def test_quantization_error_bounded():
    """Error must be within one block step of the value.

    e2m1's grid {0, .5, 1, 1.5, 2, 3, 4, 6} is not uniform -- its widest gap is 4->6 --
    so round-to-nearest can be off by half of 2.0, i.e. a full step, not half of one.
    Clipping cannot exceed that either: e4m3 rounds the scale by at most 2^-4, so a
    value can land at most 6.375 steps out and is clipped to 6.
    """
    x = jax.random.normal(jax.random.key(0), (2, 128, 256), jnp.float32)
    gs = global_scale_for(x)
    q, sf = quantize_nvfp4(x, gs)
    back = np.asarray(dequantize_nvfp4(q, sf, gs))
    xf = np.asarray(x)
    step = np.asarray(sf.astype(jnp.float32)) / np.asarray(gs).reshape(2, 1, 1)
    bound = np.repeat(step, SF_VEC_SIZE, axis=-1)
    worst = np.max(np.abs(back - xf) - bound)
    check("quantization error within one e2m1 step", worst <= 0,
          f"worst overshoot={worst:.2e}, mean|err|/step="
          f"{np.mean(np.abs(back - xf) / np.maximum(bound, 1e-30)):.3f}")


def test_scale_layout_roundtrip():
    mn, ks = 256, 32
    sf = jax.random.randint(jax.random.key(1), (mn, ks), 0, 100).astype(jnp.float8_e4m3fn)
    tiled = to_mma_scale_layout(sf)
    shape_ok = tiled.shape == (mn // 128, ks // 4, 32, 16)
    back = from_mma_scale_layout(tiled, mn)
    check("scale layout round-trips",
          shape_ok and np.array_equal(np.asarray(back, np.float32), np.asarray(sf, np.float32)),
          f"tiled shape {tiled.shape}")


def test_scale_layout_indices():
    """The tiling must place sf[mn, k] where the PTX .scale_vec::1X layout says.

    Derived independently of the implementation: an element's row splits as
    mn = 128*tile + 32*a + b (b the 32 lanes, a one of 4 groups), its column as
    k = 4*c + d, and the tile holds it at [tile, c, b, 4*a + d].
    """
    mn, ks = 256, 16
    sf = jnp.arange(mn * ks, dtype=jnp.float32).reshape(mn, ks) % 240
    tiled = np.asarray(to_mma_scale_layout(sf.astype(jnp.float8_e4m3fn)), np.float32)
    src = np.asarray(sf.astype(jnp.float8_e4m3fn), np.float32)
    bad = None
    for row in range(mn):
        for col in range(ks):
            t, a, b = row // 128, (row % 128) // 32, row % 32
            c, d = col // 4, col % 4
            if tiled[t, c, b, 4 * a + d] != src[row, col]:
                bad = (row, col)
                break
        if bad:
            break
    check("scale layout matches the PTX index mapping", bad is None,
          "" if bad is None else f"first mismatch at {bad}")


def test_moe_reference_runs():
    l, m, k, n = 2, 128, 256, 128
    key = jax.random.split(jax.random.key(2), 3)
    hidden = jax.random.normal(key[0], (l, m, k), jnp.float32).astype(jnp.bfloat16)
    w1 = jax.random.normal(key[1], (l, 2 * n, k), jnp.float32) * 0.05
    w2 = jax.random.normal(key[2], (l, k, n), jnp.float32) * 0.05
    w1_gs, w2_gs = global_scale_for(w1), global_scale_for(w2)
    w1_q, w1_sf = quantize_nvfp4(w1, w1_gs)
    w2_q, w2_sf = quantize_nvfp4(w2, w2_gs)
    ones = jnp.ones((l,), jnp.float32)
    masked_m = jnp.array([m, m // 2], jnp.int32)
    gs_in = global_scale_for(hidden.astype(jnp.float32))
    out = moe_reference(hidden, w1_q, w1_sf, w1_gs, ones, w2_q, w2_sf, w2_gs, ones,
                        masked_m, gs_in, jnp.full((l,), 8.0, jnp.float32))
    finite = bool(jnp.all(jnp.isfinite(out.astype(jnp.float32))))
    masked_zero = bool(jnp.all(out[1, m // 2:] == 0))
    check("moe_reference runs and masks rows", finite and masked_zero and out.shape == (l, m, k),
          f"shape={out.shape}")


for t in (test_exact_values_survive, test_quantization_error_bounded,
          test_scale_layout_roundtrip, test_scale_layout_indices, test_moe_reference_runs):
    t()
sys.exit(0 if all(results) else 1)
