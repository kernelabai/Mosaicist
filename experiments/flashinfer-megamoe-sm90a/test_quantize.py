"""Correctness tests for the Pallas quantization kernels (run on an sm_90a GPU)."""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from quant import BLOCK_K, global_scale_for, quantize_blockwise, silu_mul
from quantize_kernels import TILE_M, quantize_blockwise_pallas, silu_mul_quantize_pallas


def check(label, got, want, mask_rows):
    gq, gsf = got
    wq, wsf = want
    # compare only rows the kernel computes (tiles fully past masked_m are skipped)
    rows = mask_rows[:, :, None]
    q_ok = np.array_equal(np.where(rows, np.asarray(gq, np.float32), 0),
                          np.where(rows, np.asarray(wq, np.float32), 0))
    sf_rows = mask_rows[:, None, :]
    sf_ok = np.array_equal(np.where(sf_rows, np.asarray(gsf, np.float32), 0),
                           np.where(sf_rows, np.asarray(wsf, np.float32), 0))
    print(f"{'PASS' if q_ok and sf_ok else 'FAIL'}  {label:<44} values={'exact' if q_ok else 'MISMATCH'} "
          f"scales={'exact' if sf_ok else 'MISMATCH'}")
    return q_ok and sf_ok


def main():
    failures = 0
    l, m, k, n = 3, 256, 512, 256
    key = jax.random.key(0)
    masked_m = jnp.array([m, m // 2, 1][:l], jnp.int32)
    # a tile is computed in full iff its first row is inside the mask
    tile_rows = (np.arange(m)[None, :] // TILE_M * TILE_M) < np.asarray(masked_m)[:, None]

    x = jax.random.normal(key, (l, m, k), jnp.float32).astype(jnp.bfloat16)
    gs = global_scale_for(x)
    failures += not check("quantize (l,m,k) -> q, sf", quantize_blockwise_pallas(x, gs, masked_m),
                          quantize_blockwise(x.astype(jnp.float32), gs), tile_rows)

    gateup = jax.random.normal(jax.random.key(1), (l, m, 2 * n), jnp.float32).astype(jnp.bfloat16) * 4
    gs2 = global_scale_for(gateup)
    failures += not check("silu_and_mul + quantize", silu_mul_quantize_pallas(gateup, gs2, masked_m),
                          quantize_blockwise(silu_mul(gateup), gs2), tile_rows)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
