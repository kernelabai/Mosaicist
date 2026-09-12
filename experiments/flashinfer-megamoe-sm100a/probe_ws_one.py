"""One warp-specialized config per process (a CUDA fault poisons the rest).

  usage: probe_ws_one.py <block_k> <stages> [all_active]
"""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace  # noqa: E402

from masked_gemm import GemmConfig  # noqa: E402
from masked_gemm_ws import masked_grouped_gemm_ws  # noqa: E402
from nvfp4 import masked_grouped_gemm_reference, quantize_nvfp4, to_mma_scale_layout  # noqa: E402

bk, st = int(sys.argv[1]), int(sys.argv[2])
all_active = len(sys.argv) > 3 and sys.argv[3] == "all_active"
# tiles must outnumber the 148 SMs several times over or the persistent loop never
# runs a second iteration and the cross-tile protocol goes untested
L, M, K, N = (int(v) for v in (sys.argv[4:8] if len(sys.argv) > 7 else (2, 256, 512, 256)))
gs = jnp.full((L,), 64.0, jnp.float32)
a_q, a_sf = quantize_nvfp4(jax.random.normal(jax.random.key(0), (L, M, K), jnp.float32), gs)
b_q, b_sf = quantize_nvfp4(jax.random.normal(jax.random.key(1), (L, N, K), jnp.float32), gs)
alpha = jnp.ones((L,), jnp.float32)
masked_m = (jnp.full((L,), M, jnp.int32) if all_active else
            jnp.array([M if i % 3 == 0 else max(1, M // 2 - 3) for i in range(L)], jnp.int32))
asf, bsf = jax.vmap(to_mma_scale_layout)(a_sf), jax.vmap(to_mma_scale_layout)(b_sf)
ones = jnp.ones((L,), jnp.float32)
ref = np.asarray(masked_grouped_gemm_reference(a_q, a_sf, ones, b_q, b_sf, ones,
                                               alpha, masked_m), np.float32)
cfg = GemmConfig(block_k=bk, stages=st)
f = jax.jit(masked_grouped_gemm_ws, static_argnums=6)
out = np.asarray(jax.block_until_ready(f(a_q, asf, b_q, bsf, alpha, masked_m, cfg)), np.float32)
rows = np.arange(M)[None, :, None] < np.asarray(masked_m)[:, None, None]
got, want = np.where(rows, out, 0.0), np.where(rows, ref, 0.0)
ntiles = L * (M // 128) * (N // 128)
print(f"L={L} M={M} K={K} N={N} tiles={ntiles:<5} bk={bk} st={st} "
      f"{'all_active' if all_active else 'masked':<11}"
      f"{'exact' if np.array_equal(got, want) else 'MISMATCH rel=%.1e' % (np.abs(got-want).max()/max(np.abs(want).max(),1e-9))}")
