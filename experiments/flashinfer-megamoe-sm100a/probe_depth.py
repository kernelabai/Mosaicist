"""Sweep pipeline depth on the warp-split kernel.

The PTX diff puts the reference at ~7 stages (14 mbarrier inits) against this port's 2.
Depth costs shared memory, and shared memory buys occupancy, so the two trade off -- but
a finer K step shrinks the per-stage cost, and with warp 0 running ahead the deeper
pipeline may now pay where it did not for the plain kernel.
"""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace  # noqa: E402

from masked_gemm import SF_TILE_K, SMEM_PER_SM, GemmConfig  # noqa: E402
from masked_gemm_ws import masked_grouped_gemm_w1  # noqa: E402
from nvfp4 import masked_grouped_gemm_reference, quantize_nvfp4, to_mma_scale_layout  # noqa: E402

L, M, K, N = 8, 512, 2048, 2048
TM = TN = 128
gs = jnp.full((L,), 64.0, jnp.float32)
a_q, a_sf = quantize_nvfp4(jax.random.normal(jax.random.key(0), (L, M, K), jnp.float32), gs)
b_q, b_sf = quantize_nvfp4(jax.random.normal(jax.random.key(1), (L, N, K), jnp.float32), gs)
alpha = jnp.ones((L,), jnp.float32)
masked_m = jnp.full((L,), M, jnp.int32)
asf, bsf = jax.vmap(to_mma_scale_layout)(a_sf), jax.vmap(to_mma_scale_layout)(b_sf)
ones = jnp.ones((L,), jnp.float32)
ref = np.asarray(masked_grouped_gemm_reference(a_q, a_sf, ones, b_q, b_sf, ones,
                                               alpha, masked_m), np.float32)
f = jax.jit(masked_grouped_gemm_w1, static_argnums=6)
best = (1e9, None)

for bk in (64, 128, 256, 512):
    if K % bk:
        continue
    per_stage = TM * bk // 2 + TN * bk // 2 + 2 * (bk // SF_TILE_K) * 512
    for st in range(1, 13):
        smem = st * per_stage + TM * TN * 2
        if smem > SMEM_PER_SM or st > K // bk:
            continue
        cfg = GemmConfig(block_k=bk, stages=st)
        try:
            out = np.asarray(jax.block_until_ready(
                f(a_q, asf, b_q, bsf, alpha, masked_m, cfg)), np.float32)
        except Exception as e:
            print(f"bk={bk:<4}st={st:<3} ERROR {' '.join(str(e).split())[:60]}")
            continue
        exact = np.array_equal(out, ref)
        with KernelTrace() as tr:
            for _ in range(20):
                jax.block_until_ready(f(a_q, asf, b_q, bsf, alpha, masked_m, cfg))
        us = sum(r.duration_us for r in tr.records if "mosaic" in r.name) / 20
        blocks = SMEM_PER_SM // smem
        tf = 2 * L * M * N * K / (us * 1e-6) / 1e12
        if exact and us < best[0]:
            best = (us, (bk, st))
        print(f"bk={bk:<4}st={st:<3} smem={smem:<7}{blocks} blk/SM  {us:7.1f} us  "
              f"{tf:7.1f} TFLOP/s  {'exact' if exact else 'MISMATCH'}")
print(f"\nbest: {best[1]} at {best[0]:.1f} us")
