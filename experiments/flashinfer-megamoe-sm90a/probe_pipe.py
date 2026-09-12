"""Double-buffered accumulator: overlap one K block's wgmma with the previous block's
readout+rescale, instead of draining the pipeline every block.

  usage: probe_pipe.py <tile_n> <mode>     mode in {full, pipe}

`pipe` issues two wgmmas into two accumulators before draining either, so the second
block's tensor work overlaps the first block's CUDA-core rescale. The scale smem is
padded to >=128 bytes per stage: a 64-byte per-stage slice puts TMA's destination on a
64-byte boundary, which faults (this is why tile_n=64 misaligned).
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

tn = int(sys.argv[1])
MODE = sys.argv[2]
tm = int(sys.argv[3]) if len(sys.argv) > 3 else 128
stages = int(sys.argv[4]) if len(sys.argv) > 4 else 4
l, m, k, n2 = 8, 512, 2048, 2048
SF_PAD = 128  # per-stage scale slice, in bytes, to keep TMA destinations 128B-aligned


def gemm(a_q, a_sf, b_q, b_sf, alpha, masked_m):
    num_kb = k // BLOCK_K
    tr = _fp8_transforms(BLOCK_K)
    mpad, npad = max(tm, SF_PAD), max(tn, SF_PAD)

    def body(a_gmem, asf_gmem, b_gmem, bsf_gmem, alpha_gmem, mask_gmem, out_gmem):
        e, mi, ni = (lax.axis_index(x) for x in ("l", "mi", "ni"))
        m_slice, n_slice = pl.ds(mi * tm, tm), pl.ds(ni * tn, tn)

        @pl.when(mi * tm < mask_gmem[e])
        def _():
            @functools.partial(
                pl.run_scoped,
                a_smem=plgpu.SMEM((stages, tm, BLOCK_K), FP8_DTYPE, transforms=tr),
                b_smem=plgpu.SMEM((stages, tn, BLOCK_K), FP8_DTYPE, transforms=tr),
                asf_smem=plgpu.SMEM((stages, mpad), FP8_DTYPE),
                bsf_smem=plgpu.SMEM((stages, npad), FP8_DTYPE),
                out_smem=plgpu.SMEM((tm, tn), jnp.bfloat16),
                barrier=plgpu.Barrier(num_arrivals=4, num_barriers=stages),
                acc0=plgpu.ACC((tm, tn), jnp.float32),
                acc1=plgpu.ACC((tm, tn), jnp.float32),
            )
            def compute(a_smem, b_smem, asf_smem, bsf_smem, out_smem, barrier, acc0, acc1):
                def fetch(kb, slot):
                    ks = pl.ds(kb * BLOCK_K, BLOCK_K)
                    plgpu.copy_gmem_to_smem(a_gmem.at[e, m_slice, ks], a_smem.at[slot], barrier.at[slot])
                    plgpu.copy_gmem_to_smem(b_gmem.at[e, n_slice, ks], b_smem.at[slot], barrier.at[slot])
                    plgpu.copy_gmem_to_smem(asf_gmem.at[e, kb, m_slice], asf_smem.at[slot, :tm], barrier.at[slot])
                    plgpu.copy_gmem_to_smem(bsf_gmem.at[e, kb, n_slice], bsf_smem.at[slot, :tn], barrier.at[slot])

                for slot in range(stages):
                    @pl.when(slot < num_kb)
                    def _(slot=slot):
                        fetch(slot, slot)

                def drain(acc, slot, total):
                    partial = plgpu.wgmma_accumulator_load(acc)
                    acc[...] = jnp.zeros_like(partial)
                    asf = asf_smem[slot, :tm].astype(jnp.float32)
                    bsf = bsf_smem[slot, :tn].astype(jnp.float32)
                    return total + partial * asf[:, None] * bsf[None, :]

                if MODE == "full":
                    def k_block(kb, total):
                        slot = lax.rem(kb, stages)
                        plgpu.barrier_wait(barrier.at[slot])
                        plgpu.wgmma(acc0, a_smem.at[slot], plgpu.transpose_ref(b_smem.at[slot], (1, 0)))
                        plgpu.wgmma_wait(0)
                        total = drain(acc0, slot, total)

                        @pl.when(kb + stages < num_kb)
                        def _():
                            fetch(kb + stages, slot)
                        return total

                    total = lax.fori_loop(0, num_kb, k_block, jnp.zeros((tm, tn), jnp.float32))
                else:
                    def k_pair(j, total):
                        kb0, kb1 = 2 * j, 2 * j + 1
                        s0, s1 = lax.rem(kb0, stages), lax.rem(kb1, stages)
                        plgpu.barrier_wait(barrier.at[s0])
                        plgpu.wgmma(acc0, a_smem.at[s0], plgpu.transpose_ref(b_smem.at[s0], (1, 0)))
                        plgpu.barrier_wait(barrier.at[s1])
                        plgpu.wgmma(acc1, a_smem.at[s1], plgpu.transpose_ref(b_smem.at[s1], (1, 0)))
                        # acc0's rescale runs while acc1's wgmma is still in flight
                        plgpu.wgmma_wait(1)
                        total = drain(acc0, s0, total)

                        @pl.when(kb0 + stages < num_kb)
                        def _():
                            fetch(kb0 + stages, s0)
                        plgpu.wgmma_wait(0)
                        total = drain(acc1, s1, total)

                        @pl.when(kb1 + stages < num_kb)
                        def _():
                            fetch(kb1 + stages, s1)
                        return total

                    total = lax.fori_loop(0, num_kb // 2, k_pair, jnp.zeros((tm, tn), jnp.float32))

                out_smem[...] = (total * alpha_gmem[e]).astype(jnp.bfloat16)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(out_smem, out_gmem.at[e, m_slice, n_slice])
                plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(body, out_shape=jax.ShapeDtypeStruct((l, m, n2), jnp.bfloat16),
                        grid=(l, m // tm, n2 // tn), grid_names=("l", "mi", "ni"))(
        a_q, a_sf, b_q, b_sf, alpha, masked_m)


a_q = jax.random.normal(jax.random.key(0), (l, m, k), jnp.float32).astype(FP8_DTYPE)
b_q = jax.random.normal(jax.random.key(1), (l, n2, k), jnp.float32).astype(FP8_DTYPE)
a_sf = jnp.ones((l, k // BLOCK_K, m), FP8_DTYPE)
b_sf = jnp.ones((l, k // BLOCK_K, n2), FP8_DTYPE)
alpha, mask = jnp.ones((l,), jnp.float32), jnp.full((l,), m, jnp.int32)

f = jax.jit(gemm)
out = jax.block_until_ready(f(a_q, a_sf, b_q, b_sf, alpha, mask))
ref = jnp.einsum("lmk,lnk->lmn", a_q.astype(jnp.float32), b_q.astype(jnp.float32))
err = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref)) / jnp.max(jnp.abs(ref)))
with KernelTrace() as tr:
    for _ in range(20):
        jax.block_until_ready(f(a_q, a_sf, b_q, b_sf, alpha, mask))
recs = [r for r in tr.records if "mosaic" in r.name]
us = sum(r.duration_us for r in recs) / 20
print(f"tm={tm:<5}tn={tn:<5}st={stages} {MODE:<6} regs={recs[0].registers:<4}local={recs[0].local_mem_per_thread:<5}"
      f"{us:8.1f} us  {2*l*m*n2*k/(us*1e-6)/1e12:6.1f} TFLOP/s  rel_err={err:.1e}")
