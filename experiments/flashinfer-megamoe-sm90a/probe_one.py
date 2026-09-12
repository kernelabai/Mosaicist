"""One GEMM config per process: CUDA errors are async, so a crash poisons later configs."""
import pathlib, sys
import jax, jax.numpy as jnp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace
from masked_gemm import GemmConfig, masked_grouped_gemm
from moe import make_moe_inputs
from quantize_kernels import quantize_blockwise_pallas

tm, tn, st = (int(v) for v in sys.argv[1:4])
l, m, k, n = 8, 512, 2048, 1024
mask = jnp.full((l,), m, jnp.int32)
kw = make_moe_inputs(jax.random.key(0), l, m, k, n, mask)
a_q, a_sf = jax.jit(quantize_blockwise_pallas)(kw["hidden"], kw["input_global_scale"], mask)
alpha = kw["w1_alpha"] / (kw["input_global_scale"] * kw["w1_gs"])
cfg = GemmConfig(tile_m=tm, tile_n=tn, stages=st)
f = jax.jit(masked_grouped_gemm, static_argnums=6)
jax.block_until_ready(f(a_q, a_sf, kw["w1_q"], kw["w1_sf"], alpha, mask, cfg))
with KernelTrace() as tr:
    for _ in range(10):
        jax.block_until_ready(f(a_q, a_sf, kw["w1_q"], kw["w1_sf"], alpha, mask, cfg))
recs = [r for r in tr.records if "mosaic" in r.name]
r, us = recs[0], sum(x.duration_us for x in recs) / 10
print(f"tile_m={tm:<4}tile_n={tn:<4}stages={st}  regs={r.registers:<4}local={r.local_mem_per_thread:<5}"
      f"smem={r.dynamic_smem:<7}grid={r.grid}  {us:7.1f} us  "
      f"{2*l*m*(2*n)*k/(us*1e-6)/1e12:6.1f} TFLOP/s")
