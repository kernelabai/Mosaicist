"""Formulations of silu(gate)*up: time, and whether they change the quantized result.

jax.nn.sigmoid costs 47.6 of the kernel's 59.4 us. The output is e2m1 -- three bits of
mantissa -- so a formulation only has to agree with the reference to well within a
quantization step to be indistinguishable, and that is what the exactness column checks.
"""
import functools
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace  # noqa: E402

from nvfp4 import (FP4_DTYPE, FP4_MAX, SF_DTYPE, SF_MAX, SF_VEC_SIZE, quantize_nvfp4,
                   silu_mul)  # noqa: E402

M, TK = 128, 256
NB = TK // SF_VEC_SIZE
L, ROWS, N = 8, 512, 1024

SILU = {
    "jax.nn.sigmoid (shipped)": lambda g: jax.nn.sigmoid(g) * g,
    "lax.logistic": lambda g: lax.logistic(g) * g,
    "g / (1 + exp(-g))": lambda g: g / (1.0 + jnp.exp(-g)),
    "g/2 * (1 + tanh(g/2))": lambda g: g * 0.5 * (1.0 + jnp.tanh(g * 0.5)),
    # exp2 is the hardware primitive (ex2.approx); exp(x) is exp2(x * log2 e)
    "g / (1 + exp2(-g*log2e))": lambda g: g / (1.0 + jnp.exp2(-g * 1.4426950408889634)),
    "g * (1/(1+exp2(...))) via recip": lambda g: g * (1.0 / (1.0 + jnp.exp2(-g * 1.4426950408889634))),
    # the denominator is always >= 1 here, so approx reciprocal's edge cases cannot bite
    # deliberately wrong: isolates what each operation costs
    "ABLATION exp2, no divide": lambda g: g * jnp.exp2(-g * 1.4426950408889634),
    "ABLATION divide, no exp2": lambda g: g / (1.0 + g * g),
    "ABLATION neither": lambda g: g * 1.5,
}


def quantize(vals, gs):
    inv_full = jnp.zeros((M, TK), jnp.float32)
    sf_full = jnp.zeros((M, NB), jnp.float32)
    for b in range(NB):
        block = vals[:, b * SF_VEC_SIZE:(b + 1) * SF_VEC_SIZE]
        amax = plgpu.layout_cast(jnp.max(jnp.abs(block), axis=-1), plgpu.Layout.WGMMA.reduce(1))
        sf = jnp.clip(amax / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE).astype(jnp.float32)
        step = sf / gs
        inv = jnp.where(step > 0.0, 1.0, 0.0) / jnp.where(step == 0.0, 1.0, step)
        inv_full += inv[:, None] * (jnp.arange(TK) // SF_VEC_SIZE == b).astype(jnp.float32)[None, :]
        sf_full += sf[:, None] * (jnp.arange(NB) == b).astype(jnp.float32)[None, :]
    return jnp.clip(vals * inv_full, -FP4_MAX, FP4_MAX).astype(FP4_DTYPE), sf_full.astype(SF_DTYPE)


def build(silu):
    def body(x_gmem, gs_gmem, q_gmem, sf_gmem):
        e, mi, kb = (lax.axis_index(a) for a in ("l", "mi", "kb"))
        ms = pl.ds(mi * M, M)

        @functools.partial(
            pl.run_scoped,
            gate_smem=plgpu.SMEM((M, TK), jnp.bfloat16), up_smem=plgpu.SMEM((M, TK), jnp.bfloat16),
            q_smem=plgpu.SMEM((M, TK), FP4_DTYPE), sf_smem=plgpu.SMEM((M, NB), SF_DTYPE),
            barrier=plgpu.Barrier(num_arrivals=2))
        def scoped(gate_smem, up_smem, q_smem, sf_smem, barrier):
            cols = pl.ds(kb * TK, TK)
            plgpu.copy_gmem_to_smem(x_gmem.at[e, ms, cols], gate_smem, barrier)
            plgpu.copy_gmem_to_smem(x_gmem.at[e, ms, pl.ds(N + kb * TK, TK)], up_smem, barrier)
            plgpu.barrier_wait(barrier)
            gate = plgpu.layout_cast(gate_smem[...].astype(jnp.float32), plgpu.Layout.WGMMA)
            up = plgpu.layout_cast(up_smem[...].astype(jnp.float32), plgpu.Layout.WGMMA)
            q_smem[...], sf_smem[...] = quantize(silu(gate) * up, gs_gmem[e])
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(q_smem, q_gmem.at[e, ms, cols])
            plgpu.copy_smem_to_gmem(sf_smem, sf_gmem.at[e, ms, pl.ds(kb * NB, NB)])
            plgpu.wait_smem_to_gmem(0)

    return plgpu.kernel(
        body, out_shape=(jax.ShapeDtypeStruct((L, ROWS, N), FP4_DTYPE),
                         jax.ShapeDtypeStruct((L, ROWS, N // SF_VEC_SIZE), SF_DTYPE)),
        grid=(L, ROWS // M, N // TK), grid_names=("l", "mi", "kb"))


x = jax.random.normal(jax.random.key(0), (L, ROWS, 2 * N), jnp.float32).astype(jnp.bfloat16) * 3
gs = jnp.full((L,), 64.0, jnp.float32)
want_q, want_sf = quantize_nvfp4(silu_mul(x), gs)
wq, wsf = np.asarray(want_q, np.float32), np.asarray(want_sf, np.float32)

for name, silu in SILU.items():
    f = jax.jit(build(lambda g, s=silu: s(g)))
    q, sf = jax.block_until_ready(f(x, gs))
    gq, gsf = np.asarray(q, np.float32), np.asarray(sf, np.float32)
    same = np.array_equal(gq, wq) and np.array_equal(gsf, wsf)
    differing = float((gq != wq).mean())
    with KernelTrace() as tr:
        for _ in range(20):
            jax.block_until_ready(f(x, gs))
    us = sum(r.duration_us for r in tr.records if "mosaic" in r.name) / 20
    tag = "bit-exact" if same else f"{differing:.2%} of values differ"
    print(f"{us:7.1f} us  {tag:<26} {name}")
