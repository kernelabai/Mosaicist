"""Split the fused kernel's time between its GEMM half and its epilogue.

The fused kernel runs two passes, each a full K loop followed by silu+quantize on the
accumulator. Nothing overlaps the two, so if the epilogue is a large share, hiding it is
the next lever -- and if it is small, the GEMM half is what to attack.
"""
import pathlib
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace  # noqa: E402

import gemm1_silu_quantize as g  # noqa: E402
from gemm1_silu_quantize import FusedConfig  # noqa: E402
from nvfp4 import FP4_MAX, quantize_nvfp4, to_mma_scale_layout  # noqa: E402

L, M, K, N = 8, 512, 2048, 1024
gs = jnp.full((L,), 64.0, jnp.float32)
a_q, a_sf = quantize_nvfp4(jax.random.normal(jax.random.key(0), (L, M, K), jnp.float32), gs)
w1_q, w1_sf = quantize_nvfp4(
    jax.random.normal(jax.random.key(1), (L, 2 * N, K), jnp.float32) * 0.05, gs)
alpha = jnp.ones((L,), jnp.float32)
a2gs = jnp.full((L,), 32.0, jnp.float32)
mask = jnp.full((L,), M, jnp.int32)
asf, wsf = jax.vmap(to_mma_scale_layout)(a_sf), jax.vmap(to_mma_scale_layout)(w1_sf)

real_quant, real_silu = g._quantize_half, None
SRC = open(g.__file__).read()


def no_quant(vals, gsc, q_smem, p, sf_full, nb_total):
    """Skip the one-hot expansion: store a raw cast and a constant scale."""
    nsub = g.HALF // g.SUB_K
    vals = jax.experimental.pallas.mosaic_gpu.layout_cast(
        vals, jax.experimental.pallas.mosaic_gpu.Layout.TCGEN05)
    for s in range(nsub):
        sub = vals[:, s * g.SUB_K:(s + 1) * g.SUB_K]
        q_smem[p, s] = jnp.clip(sub, -FP4_MAX, FP4_MAX).astype(g.FP4_DTYPE)
    return sf_full + 1.0


def timed(cfg, reps=20):
    f = jax.jit(g.gemm1_silu_quantize, static_argnums=7)
    jax.block_until_ready(f(a_q, asf, w1_q, wsf, alpha, a2gs, mask, cfg))
    with KernelTrace() as tr:
        for _ in range(reps):
            jax.block_until_ready(f(a_q, asf, w1_q, wsf, alpha, a2gs, mask, cfg))
    return sum(r.duration_us for r in tr.records if "mosaic" in r.name) / reps


for cfg in (FusedConfig(block_k=128, stages=3), FusedConfig(block_k=256, stages=2)):
    g._quantize_half = real_quant
    jax.clear_caches()
    full = timed(cfg)
    g._quantize_half = no_quant
    jax.clear_caches()
    noq = timed(cfg)
    print(f"bk={cfg.block_k:<4}st={cfg.stages}  full={full:6.1f} us   "
          f"without the quantize expansion={noq:6.1f} us   epilogue share={full - noq:5.1f} us")
