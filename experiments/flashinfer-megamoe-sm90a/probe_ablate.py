"""Ablate the per-K-block rescale to see what it costs against a pure-GEMM upper bound.

  full      what masked_gemm.py does: read the accumulator every K block, scale it by
            the outer product of the two scale vectors, add into a register total
  noscale   same accumulator readout, but no scaling (isolates the FMA cost)
  pure      no readout at all -- wgmma accumulates across all of K, one epilogue
            (this is the ceiling: a plain fp8 GEMM with no block scales)
"""
import functools, pathlib, sys
import jax, jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace
from quant import BLOCK_K, FP8_DTYPE
from masked_gemm import _fp8_transforms

MODE = sys.argv[1]
tm = tn = 128
stages = 4
l, m, k, n2 = 8, 512, 2048, 2048


def gemm(a_q, a_sf, b_q, b_sf, alpha, masked_m):
    num_kb = k // BLOCK_K
    tr = _fp8_transforms(BLOCK_K)

    def body(a_gmem, asf_gmem, b_gmem, bsf_gmem, alpha_gmem, mask_gmem, out_gmem):
        e, mi, ni = (lax.axis_index(x) for x in ("l", "mi", "ni"))
        m_slice, n_slice = pl.ds(mi * tm, tm), pl.ds(ni * tn, tn)

        @pl.when(mi * tm < mask_gmem[e])
        def _():
            @functools.partial(
                pl.run_scoped,
                a_smem=plgpu.SMEM((stages, tm, BLOCK_K), FP8_DTYPE, transforms=tr),
                b_smem=plgpu.SMEM((stages, tn, BLOCK_K), FP8_DTYPE, transforms=tr),
                asf_smem=plgpu.SMEM((stages, tm), FP8_DTYPE),
                bsf_smem=plgpu.SMEM((stages, tn), FP8_DTYPE),
                out_smem=plgpu.SMEM((tm, tn), jnp.bfloat16),
                barrier=plgpu.Barrier(num_arrivals=4, num_barriers=stages),
                acc_ref=plgpu.ACC((tm, tn), jnp.float32),
            )
            def compute(a_smem, b_smem, asf_smem, bsf_smem, out_smem, barrier, acc_ref):
                def fetch(kb, slot):
                    ks = pl.ds(kb * BLOCK_K, BLOCK_K)
                    plgpu.copy_gmem_to_smem(a_gmem.at[e, m_slice, ks], a_smem.at[slot], barrier.at[slot])
                    plgpu.copy_gmem_to_smem(b_gmem.at[e, n_slice, ks], b_smem.at[slot], barrier.at[slot])
                    plgpu.copy_gmem_to_smem(asf_gmem.at[e, kb, m_slice], asf_smem.at[slot], barrier.at[slot])
                    plgpu.copy_gmem_to_smem(bsf_gmem.at[e, kb, n_slice], bsf_smem.at[slot], barrier.at[slot])

                for slot in range(stages):
                    @pl.when(slot < num_kb)
                    def _(slot=slot):
                        fetch(slot, slot)

                def k_block(kb, total):
                    slot = lax.rem(kb, stages)
                    plgpu.barrier_wait(barrier.at[slot])
                    plgpu.wgmma(acc_ref, a_smem.at[slot], plgpu.transpose_ref(b_smem.at[slot], (1, 0)))
                    if MODE != "pure":
                        partial = acc_ref[...]
                        acc_ref[...] = jnp.zeros_like(partial)
                        if MODE == "full":
                            asf = asf_smem[slot].astype(jnp.float32)
                            bsf = bsf_smem[slot].astype(jnp.float32)
                            total += partial * asf[:, None] * bsf[None, :]
                        else:
                            total += partial

                    @pl.when(kb + stages < num_kb)
                    def _():
                        fetch(kb + stages, slot)
                    return total

                total = lax.fori_loop(0, num_kb, k_block, jnp.zeros((tm, tn), jnp.float32))
                if MODE == "pure":
                    total = acc_ref[...]
                out_smem[...] = (total * alpha_gmem[e]).astype(jnp.bfloat16)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[e, m_slice, n_slice])
                plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(body, out_shape=jax.ShapeDtypeStruct((l, m, n2), jnp.bfloat16),
                        grid=(l, m // tm, n2 // tn), grid_names=("l", "mi", "ni"))(
        a_q, a_sf, b_q, b_sf, alpha, masked_m)


key = jax.random.key(0)
a_q = jax.random.normal(key, (l, m, k), jnp.float32).astype(FP8_DTYPE)
b_q = jax.random.normal(jax.random.key(1), (l, n2, k), jnp.float32).astype(FP8_DTYPE)
a_sf = jnp.ones((l, k // BLOCK_K, m), FP8_DTYPE)
b_sf = jnp.ones((l, k // BLOCK_K, n2), FP8_DTYPE)
alpha = jnp.ones((l,), jnp.float32)
mask = jnp.full((l,), m, jnp.int32)

f = jax.jit(gemm)
jax.block_until_ready(f(a_q, a_sf, b_q, b_sf, alpha, mask))
with KernelTrace() as tr:
    for _ in range(20):
        jax.block_until_ready(f(a_q, a_sf, b_q, b_sf, alpha, mask))
recs = [r for r in tr.records if "mosaic" in r.name]
us = sum(r.duration_us for r in recs) / 20
print(f"{MODE:<8} regs={recs[0].registers:<4}local={recs[0].local_mem_per_thread:<5}"
      f"{us:8.1f} us  {2*l*m*n2*k/(us*1e-6)/1e12:6.1f} TFLOP/s")
