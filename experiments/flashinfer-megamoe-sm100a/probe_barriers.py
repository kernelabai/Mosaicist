"""Do Mosaic's automatically inserted barriers cost us?

The PTX diff against the CuTeDSL reference shows a `bar.sync` -- a full CTA barrier --
before every one of our TMA loads (24 in the kernel against the reference's 8). Our
pipeline already orders itself with explicit mbarriers, so `unsafe_no_auto_barriers`
may be safe here. "May" is the operative word, hence the correctness check.
"""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import mosaic_gpu as plgpu

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace  # noqa: E402

import masked_gemm as mg  # noqa: E402
from masked_gemm import GemmConfig  # noqa: E402
from nvfp4 import masked_grouped_gemm_reference, quantize_nvfp4, to_mma_scale_layout  # noqa: E402

L, M, K, N = 8, 512, 2048, 2048
gs = jnp.full((L,), 64.0, jnp.float32)
a_q, a_sf = quantize_nvfp4(jax.random.normal(jax.random.key(0), (L, M, K), jnp.float32), gs)
b_q, b_sf = quantize_nvfp4(jax.random.normal(jax.random.key(1), (L, N, K), jnp.float32), gs)
alpha = jax.random.uniform(jax.random.key(2), (L,), jnp.float32, 0.5, 2.0)
masked_m = jnp.array([M if i % 3 == 0 else max(1, M // 2 - 3) for i in range(L)], jnp.int32)
asf, bsf = jax.vmap(to_mma_scale_layout)(a_sf), jax.vmap(to_mma_scale_layout)(b_sf)
ones = jnp.ones((L,), jnp.float32)
ref = np.asarray(masked_grouped_gemm_reference(a_q, a_sf, ones, b_q, b_sf, ones,
                                               alpha, masked_m), np.float32)
rows = np.arange(M)[None, :, None] < np.asarray(masked_m)[:, None, None]
want = np.where(rows, ref, 0.0)

for no_auto in (False, True):
    for bk, st in ((512, 1), (256, 2), (128, 4)):
        mg.NO_AUTO_BARRIERS = no_auto
        jax.clear_caches()  # the flag is read at trace time; jit caches on the callable
        cfg = GemmConfig(block_k=bk, stages=st)
        f = jax.jit(mg.masked_grouped_gemm, static_argnums=6)
        try:
            out = np.asarray(jax.block_until_ready(
                f(a_q, asf, b_q, bsf, alpha, masked_m, cfg)), np.float32)
        except Exception as e:
            print(f"no_auto={no_auto!s:<6}bk={bk:<4}st={st}  ERROR {' '.join(str(e).split())[:60]}")
            continue
        got = np.where(rows, out, 0.0)
        exact = np.array_equal(got, want)
        with KernelTrace() as tr:
            for _ in range(20):
                jax.block_until_ready(f(a_q, asf, b_q, bsf, alpha, masked_m, cfg))
        us = sum(r.duration_us for r in tr.records if "mosaic" in r.name) / 20
        print(f"no_auto={no_auto!s:<6}bk={bk:<4}st={st}  {us:7.1f} us  "
              f"{2 * L * M * N * K / (us * 1e-6) / 1e12:7.1f} TFLOP/s  "
              f"{'exact' if exact else 'MISMATCH'}")
