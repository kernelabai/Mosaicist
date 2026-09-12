"""A/B the consumer-warpgroup count on the GEMM shapes the MoE actually runs.

  usage: probe_wg.py <consumers> <stages> [tile_n]

One config per process: a CUDA error is asynchronous, so a crash would poison every
config measured after it in the same process.
"""
import pathlib
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace  # noqa: E402

from masked_gemm import GemmConfig, masked_grouped_gemm  # noqa: E402
from moe import make_moe_inputs  # noqa: E402
from quantize_kernels import quantize_blockwise_pallas  # noqa: E402

nwg, stages = int(sys.argv[1]), int(sys.argv[2])
tn = int(sys.argv[3]) if len(sys.argv) > 3 else 64
cfg = GemmConfig(tile_n=tn, stages=stages, consumers=nwg)

l, m, k, n = 8, 512, 2048, 1024
mask = jnp.full((l,), m, jnp.int32)
kw = make_moe_inputs(jax.random.key(0), l, m, k, n, mask)
a_q, a_sf = jax.jit(quantize_blockwise_pallas)(kw["hidden"], kw["input_global_scale"], mask)
alpha = kw["w1_alpha"] / (kw["input_global_scale"] * kw["w1_gs"])
f = jax.jit(masked_grouped_gemm, static_argnums=6)

jax.block_until_ready(f(a_q, a_sf, kw["w1_q"], kw["w1_sf"], alpha, mask, cfg))
with KernelTrace() as tr:
    for _ in range(20):
        jax.block_until_ready(f(a_q, a_sf, kw["w1_q"], kw["w1_sf"], alpha, mask, cfg))
recs = [r for r in tr.records if "mosaic" in r.name]
r, us = recs[0], sum(x.duration_us for x in recs) / 20
print(f"consumers={nwg} stages={stages} tile_n={tn:<4} regs={r.registers:<4}"
      f"smem={r.dynamic_smem:<7}block={r.block[0]:<4}grid={r.grid}  "
      f"{us:7.1f} us  {2 * l * m * (2 * n) * k / (us * 1e-6) / 1e12:6.1f} TFLOP/s")
