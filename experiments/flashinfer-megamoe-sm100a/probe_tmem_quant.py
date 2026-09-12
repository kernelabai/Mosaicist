"""How do you quantize a value that came out of TMEM?

The fused kernel's epilogue reduces over 16-column blocks of the accumulator, but
`async_load_tmem` hands back a TMEM-native layout, not the WGMMA layout the standalone
quantize kernels start from. This finds which formulation Mosaic can actually lower.
"""
import functools
import sys

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from nvfp4 import FP4_DTYPE, FP4_MAX, SF_DTYPE, SF_MAX, SF_VEC_SIZE

M, TN = 128, 256  # TN=256 -> 16 scale columns, the narrowest legal TMA store
NB = TN // SF_VEC_SIZE
print("Layout has TCGEN05:", hasattr(plgpu.Layout, "TCGEN05"),
      "| reduce on TCGEN05:", hasattr(getattr(plgpu.Layout, "TCGEN05", None), "reduce"))


def build(mode):
    def body(a_gmem, q_gmem, sf_gmem, acc_tmem, buf_smem, q_smem, sf_smem, bar):
        @pl.when(lax.axis_index("i") < 1)
        def _():
            plgpu.copy_gmem_to_smem(a_gmem.at[:], buf_smem, bar)
            plgpu.barrier_wait(bar)
            plgpu.async_store_tmem(acc_tmem, buf_smem[...].astype(jnp.float32))
            plgpu.commit_tmem()
            vals = plgpu.async_load_tmem(acc_tmem)
            plgpu.wait_load_tmem()
            if mode == "wgmma":
                vals = plgpu.layout_cast(vals, plgpu.Layout.WGMMA)
                red = plgpu.Layout.WGMMA.reduce(1)
            elif mode == "tcgen05":
                vals = plgpu.layout_cast(vals, plgpu.Layout.TCGEN05)
                red = plgpu.Layout.TCGEN05.reduce(1)
            elif mode == "infer":
                red = None
            elif mode == "via_smem":
                buf_smem[...] = vals.astype(jnp.bfloat16)
                plgpu.commit_smem()
                vals = plgpu.layout_cast(buf_smem[...].astype(jnp.float32),
                                         plgpu.Layout.WGMMA)
                red = plgpu.Layout.WGMMA.reduce(1)
            inv_full = jnp.zeros((M, TN), jnp.float32)
            sf_full = jnp.zeros((M, NB), jnp.float32)
            for b in range(NB):
                blk = vals[:, b * SF_VEC_SIZE:(b + 1) * SF_VEC_SIZE]
                amax = jnp.max(jnp.abs(blk), axis=-1)
                if red is not None:
                    amax = plgpu.layout_cast(amax, red)
                sf = jnp.clip(amax / FP4_MAX * 32.0, 0.0, SF_MAX).astype(SF_DTYPE).astype(jnp.float32)
                inv = jnp.where(sf > 0.0, 1.0, 0.0) / jnp.where(sf == 0.0, 1.0, sf)
                inv_full += inv[:, None] * (
                    jnp.arange(TN) // SF_VEC_SIZE == b).astype(jnp.float32)[None, :]
                sf_full += sf[:, None] * (jnp.arange(NB) == b).astype(jnp.float32)[None, :]
            q_smem[...] = jnp.clip(vals * inv_full, -FP4_MAX, FP4_MAX).astype(FP4_DTYPE)
            sf_smem[...] = sf_full.astype(SF_DTYPE)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(q_smem, q_gmem.at[:])
            plgpu.copy_smem_to_gmem(sf_smem, sf_gmem.at[:])
            plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        body, out_shape=(jax.ShapeDtypeStruct((M, TN), FP4_DTYPE),
                         jax.ShapeDtypeStruct((M, NB), SF_DTYPE)),
        grid=(1,), grid_names=("i",),
        scratch_shapes=[plgpu.TMEM((M, TN), jnp.float32),
                        plgpu.SMEM((M, TN), jnp.bfloat16),
                        plgpu.SMEM((M, TN), FP4_DTYPE),
                        plgpu.SMEM((M, NB), SF_DTYPE),
                        plgpu.Barrier()])


x = jnp.ones((M, TN), jnp.bfloat16)
for mode in ("wgmma", "tcgen05", "infer", "via_smem"):
    try:
        jax.block_until_ready(jax.jit(build(mode))(x))
        print(f"PASS  {mode}")
    except Exception as e:
        print(f"FAIL  {mode}: {' '.join(str(e).split())[:95]}")
