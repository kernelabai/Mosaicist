"""Registers / local-memory spill per thread for the GEMM kernel, from CUPTI records."""
import pathlib, sys
import jax, jax.numpy as jnp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace
from masked_gemm import GemmConfig, masked_grouped_gemm
from moe import make_moe_inputs

l, m, k, n = 8, 512, 2048, 1024
mask = jnp.full((l,), m, jnp.int32)
kw = make_moe_inputs(jax.random.key(0), l, m, k, n, mask)
from quantize_kernels import quantize_blockwise_pallas
a_q, a_sf = jax.jit(quantize_blockwise_pallas)(kw["hidden"], kw["input_global_scale"], mask)
alpha = kw["w1_alpha"] / (kw["input_global_scale"] * kw["w1_gs"])

for cfg in [GemmConfig(), GemmConfig(tile_m=64), GemmConfig(tile_n=64), GemmConfig(tile_m=64, tile_n=64, stages=6)]:
    f = jax.jit(masked_grouped_gemm, static_argnums=6)
    try:
        jax.block_until_ready(f(a_q, a_sf, kw["w1_q"], kw["w1_sf"], alpha, mask, cfg))
        with KernelTrace() as tr:
            for _ in range(10):
                jax.block_until_ready(f(a_q, a_sf, kw["w1_q"], kw["w1_sf"], alpha, mask, cfg))
        recs = [r for r in tr.records if "mosaic" in r.name]
        r = recs[0]
        us = sum(x.duration_us for x in recs) / 10
        flops = 2 * l * m * (2 * n) * k
        print(f"tile_m={cfg.tile_m:<4}tile_n={cfg.tile_n:<4}stages={cfg.stages}  "
              f"regs={r.registers:<4}local={r.local_mem_per_thread:<5}smem={r.dynamic_smem:<7}"
              f"grid={r.grid} block={r.block}  {us:7.1f} us  {flops/(us*1e-6)/1e12:6.1f} TFLOP/s")
    except Exception as e:
        print(f"tile_m={cfg.tile_m} tile_n={cfg.tile_n} stages={cfg.stages}: {str(e).splitlines()[-1][:90]}")
