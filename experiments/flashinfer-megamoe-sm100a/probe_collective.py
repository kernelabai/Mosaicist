"""2-CTA collective MMA: correctness first, then speed, against the 1-CTA path.

A cluster of two CTAs shares one (2*tile_m, tile_n) tile. Each CTA holds half of B's
columns and half of the rows, and the MMA reads both halves -- so B is fetched once per
cluster instead of once per CTA, which is the saving. B scales must then be tiled per
64-column group (see nvfp4.to_mma_scale_layout_grouped), because half of a 128-column
tile is not contiguous in the MMA's scale layout.
"""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace  # noqa: E402

from masked_gemm import GemmConfig, masked_grouped_gemm  # noqa: E402
from nvfp4 import (masked_grouped_gemm_reference, quantize_nvfp4,
                   to_mma_scale_layout)  # noqa: E402

L, M, K, N = 8, 512, 2048, 2048


def run(label, cfg):
    gs = jnp.full((L,), 64.0, jnp.float32)
    a_q, a_sf = quantize_nvfp4(jax.random.normal(jax.random.key(0), (L, M, K), jnp.float32), gs)
    b_q, b_sf = quantize_nvfp4(jax.random.normal(jax.random.key(1), (L, N, K), jnp.float32), gs)
    alpha = jax.random.uniform(jax.random.key(2), (L,), jnp.float32, 0.5, 2.0)
    masked_m = jnp.full((L,), M, jnp.int32)

    asf = jax.vmap(to_mma_scale_layout)(a_sf)
    bsf = jax.vmap(to_mma_scale_layout)(b_sf)

    ones = jnp.ones((L,), jnp.float32)
    ref = np.asarray(masked_grouped_gemm_reference(a_q, a_sf, ones, b_q, b_sf, ones,
                                                   alpha, masked_m), np.float32)
    try:
        f = jax.jit(masked_grouped_gemm, static_argnums=6)
        out = np.asarray(jax.block_until_ready(
            f(a_q, asf, b_q, bsf, alpha, masked_m, cfg)), np.float32)
    except Exception as e:
        print(f"  --     {label}: {' '.join(str(e).split())[:100]}")
        return
    exact = np.array_equal(out, ref)
    rel = np.abs(out - ref).max() / max(np.abs(ref).max(), 1e-9)
    with KernelTrace() as tr:
        for _ in range(20):
            jax.block_until_ready(f(a_q, asf, b_q, bsf, alpha, masked_m, cfg))
    recs = [r for r in tr.records if "mosaic" in r.name]
    us = sum(r.duration_us for r in recs) / 20
    print(f"{us:7.1f} us  {2 * L * M * N * K / (us * 1e-6) / 1e12:7.1f} TFLOP/s  "
          f"{'exact' if exact else f'rel={rel:.1e}':<12} smem={recs[0].dynamic_smem:<7}"
          f"grid={recs[0].grid} cluster={recs[0].cluster}  {label}")


run("1 CTA (current default)", GemmConfig())
run("2 CTA collective", GemmConfig(collective=True))
run("2 CTA collective, block_k=256", GemmConfig(collective=True, block_k=256, stages=2))
run("2 CTA collective, bk=256 st1", GemmConfig(collective=True, block_k=256, stages=1))
run("2 CTA collective, bk=512 st1", GemmConfig(collective=True, block_k=512, stages=1))
run("2 CTA collective, bk=128 st2", GemmConfig(collective=True, block_k=128, stages=2))
run("1 CTA, bk=512 st1", GemmConfig(block_k=512, stages=1))
