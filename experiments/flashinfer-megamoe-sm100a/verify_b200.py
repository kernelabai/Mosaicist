"""Sanity + stress for the Blackwell kernels, now that they can actually run.

Two things the passing test suite does not by itself establish:
  * that the comparisons are non-trivial (an all-zero output matches an all-zero
    reference perfectly), and
  * that reusing one scale TMEM buffer across K blocks is safe. `masked_gemm.py` marks
    that as an ASSUMPTION: the copy and the MMA are both issued from the same thread
    into the tensor core's queue, so the copy for block kb+1 should not overtake the
    MMA for block kb. A race there would need many K blocks in flight to show up.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from masked_gemm import GemmConfig, masked_grouped_gemm
from nvfp4 import (SF_VEC_SIZE, masked_grouped_gemm_reference, quantize_nvfp4,
                   to_mma_scale_layout)

_retile = jax.vmap(to_mma_scale_layout)
ok = True


def case(label, l, m, n, k, cfg, seed=0):
    """One config. Errors are reported and moved past: a config that cannot be built is
    a result, not a reason to lose the other cases."""
    global ok
    try:
        _case(label, l, m, n, k, cfg, seed)
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"FAIL  {label:<38} {' '.join(str(e).split())[:80]}")


def _case(label, l, m, n, k, cfg, seed=0):
    global ok
    key = jax.random.split(jax.random.key(seed), 3)
    gs = jnp.full((l,), 64.0, jnp.float32)
    a_q, a_sf = quantize_nvfp4(jax.random.normal(key[0], (l, m, k), jnp.float32), gs)
    b_q, b_sf = quantize_nvfp4(jax.random.normal(key[1], (l, n, k), jnp.float32), gs)
    alpha = jax.random.uniform(key[2], (l,), jnp.float32, 0.5, 2.0)
    masked_m = jnp.array([m] + [max(1, m // 2 - 3)] * (l - 1), jnp.int32)

    ones = jnp.ones((l,), jnp.float32)
    ref = np.asarray(masked_grouped_gemm_reference(a_q, a_sf, ones, b_q, b_sf, ones,
                                                   alpha, masked_m), np.float32)
    out = np.asarray(masked_grouped_gemm(a_q, _retile(a_sf), b_q, _retile(b_sf),
                                         alpha, masked_m, cfg), np.float32)
    rows = np.arange(m)[None, :, None] < np.asarray(masked_m)[:, None, None]
    got, want = np.where(rows, out, 0.0), np.where(rows, ref, 0.0)

    # the comparison has to be worth making: real spread, mostly non-zero, and the
    # reference must not be trivially reproducible by returning zeros
    nonzero = float((want != 0).mean())
    spread = float(np.abs(want).max())
    trivial = nonzero < 0.5 or spread < 1e-3
    match = np.array_equal(got, want)
    rel = np.abs(got - want).max() / max(spread, 1e-9)
    good = match and not trivial
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {label:<38} "
          f"{'bit-exact' if match else f'rel={rel:.2e}'}  "
          f"nonzero={nonzero:.2f} max|ref|={spread:.1f}"
          f"{'  TRIVIAL COMPARISON' if trivial else ''}")


# the scale TMEM buffer is reused once per K block, so many blocks is the stress case:
# k=4096 is 32 of them at block_k=128, and 8 at block_k=512
for label, k, cfg in [
    ("k=512,  block_k=256, 2 stages", 512, GemmConfig()),
    ("k=4096, block_k=256, 2 stages", 4096, GemmConfig()),
    ("k=4096, block_k=128, 4 stages", 4096, GemmConfig(block_k=128, stages=4)),
    ("k=4096, block_k=128, 6 stages", 4096, GemmConfig(block_k=128, stages=6)),
    ("k=4096, block_k=512, 1 stage", 4096, GemmConfig(block_k=512, stages=1)),
    ("k=4096, tile_n=256", 4096, GemmConfig(tile_n=256)),
    ("k=4096, auto block_k/stages", 4096, GemmConfig()),
    # 2-CTA collective is slower (see the README) but supported, so keep it covered
    ("k=4096, 2-CTA collective", 4096, GemmConfig(collective=True)),
    ("k=512,  2-CTA collective", 512, GemmConfig(collective=True)),
]:
    case(label, 2, 256, 256, k, cfg)

# repeat the deepest pipeline across seeds: a race is not necessarily deterministic
for seed in range(1, 4):
    case(f"k=4096, block_k=128, 6 stages, seed {seed}", 2, 256, 256, 4096,
         GemmConfig(block_k=128, stages=6), seed)

sys.exit(0 if ok else 1)
