"""Sweep the Blackwell GEMM's tile_n / block_k / stages at gemm1's shape."""
import pathlib, sys
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace
from masked_gemm import GemmConfig, masked_grouped_gemm
from nvfp4 import masked_grouped_gemm_reference, quantize_nvfp4, to_mma_scale_layout

_retile = jax.vmap(to_mma_scale_layout)
tn, bk, st = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cfg = GemmConfig(tile_n=tn, block_k=bk, stages=st)
l, m, k, n = 8, 512, 2048, 2048
gs = jnp.full((l,), 64.0, jnp.float32)
a_q, a_sf = quantize_nvfp4(jax.random.normal(jax.random.key(0), (l, m, k), jnp.float32), gs)
b_q, b_sf = quantize_nvfp4(jax.random.normal(jax.random.key(1), (l, n, k), jnp.float32), gs)
alpha, mask = jnp.ones((l,), jnp.float32), jnp.full((l,), m, jnp.int32)
asf, bsf = _retile(a_sf), _retile(b_sf)
f = jax.jit(masked_grouped_gemm, static_argnums=6)
out = jax.block_until_ready(f(a_q, asf, b_q, bsf, alpha, mask, cfg))
ref = masked_grouped_gemm_reference(a_q, a_sf, jnp.ones((l,)), b_q, b_sf, jnp.ones((l,)), alpha, mask)
exact = np.array_equal(np.asarray(out, np.float32), np.asarray(ref, np.float32))
with KernelTrace() as tr:
    for _ in range(20):
        jax.block_until_ready(f(a_q, asf, b_q, bsf, alpha, mask, cfg))
recs = [r for r in tr.records if "mosaic" in r.name]
us = sum(r.duration_us for r in recs) / 20
print(f"tile_n={tn:<5}block_k={bk:<5}stages={st}  smem={recs[0].dynamic_smem:<7}"
      f"{'exact' if exact else 'MISMATCH':<9}{us:7.1f} us  {2*l*m*n*k/(us*1e-6)/1e12:7.1f} TFLOP/s")
