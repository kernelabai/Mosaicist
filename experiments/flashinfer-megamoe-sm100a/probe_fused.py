"""The fused GEMM1+silu+quantize against the two kernels it replaces."""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace  # noqa: E402

from gemm1_silu_quantize import FusedConfig, gemm1_silu_quantize  # noqa: E402
from masked_gemm import GemmConfig  # noqa: E402
from masked_gemm_ws import masked_grouped_gemm_w1  # noqa: E402
from nvfp4 import (masked_grouped_gemm_reference, quantize_nvfp4, silu_mul,
                   to_mma_scale_layout)  # noqa: E402
from quantize_kernels import silu_mul_quantize_nvfp4_pallas  # noqa: E402

L, M, K, N = 8, 512, 2048, 1024
gs = jnp.full((L,), 64.0, jnp.float32)
a_q, a_sf = quantize_nvfp4(jax.random.normal(jax.random.key(0), (L, M, K), jnp.float32), gs)
w1_q, w1_sf = quantize_nvfp4(
    jax.random.normal(jax.random.key(1), (L, 2 * N, K), jnp.float32) * 0.05, gs)
alpha = jax.random.uniform(jax.random.key(2), (L,), jnp.float32, 0.5, 1.5)
a2gs = jnp.full((L,), 32.0, jnp.float32)
masked_m = jnp.full((L,), M, jnp.int32)
asf, wsf = jax.vmap(to_mma_scale_layout)(a_sf), jax.vmap(to_mma_scale_layout)(w1_sf)
ones = jnp.ones((L,), jnp.float32)

# reference: unfused gemm1 (bf16 out) -> silu -> quantize
gateup_ref = masked_grouped_gemm_reference(a_q, a_sf, ones, w1_q, w1_sf, ones,
                                           alpha, masked_m, jnp.bfloat16)
wq, wsf_ref = quantize_nvfp4(silu_mul(gateup_ref), a2gs)

jg = jax.jit(masked_grouped_gemm_w1, static_argnums=6)
jsq = jax.jit(silu_mul_quantize_nvfp4_pallas)
jf = jax.jit(gemm1_silu_quantize, static_argnums=7)


def timed(fn, reps=20):
    jax.block_until_ready(fn())
    with KernelTrace() as tr:
        for _ in range(reps):
            jax.block_until_ready(fn())
    return sum(r.duration_us for r in tr.records if "mosaic" in r.name) / reps


gateup = jax.block_until_ready(jg(a_q, asf, w1_q, wsf, alpha, masked_m, GemmConfig()))
us_unfused = timed(lambda: jg(a_q, asf, w1_q, wsf, alpha, masked_m, GemmConfig())) \
    + timed(lambda: jsq(gateup, a2gs, masked_m))
print(f"unfused gemm1 + silu_quantize      {us_unfused:7.1f} us")

for bk, st in [(256, 1), (256, 2), (128, 2), (128, 3), (128, 4), (512, 1)]:
    cfg = FusedConfig(block_k=bk, stages=st)
    try:
        q, sf = jax.block_until_ready(jf(a_q, asf, w1_q, wsf, alpha, a2gs, masked_m, cfg))
    except Exception as ex:
        print(f"fused bk={bk} st={st}: {' '.join(str(ex).split())[:95]}")
        continue
    ok_q = np.array_equal(np.asarray(q, np.float32), np.asarray(wq, np.float32))
    ok_sf = np.array_equal(np.asarray(sf, np.float32), np.asarray(wsf_ref, np.float32))
    us = timed(lambda: jf(a_q, asf, w1_q, wsf, alpha, a2gs, masked_m, cfg))
    smem = st * (3 * 128 * bk // 2 + 3 * (bk // 64) * 512) + 2 * 2 * 128 * 64 // 2 + 128 * 16
    print(f"fused bk={bk:<4}st={st:<3} smem={smem:<7}{232448 // smem} blk/SM {us:7.1f} us  "
          f"values={'exact' if ok_q else 'MISMATCH'} scales={'exact' if ok_sf else 'MISMATCH'}")
