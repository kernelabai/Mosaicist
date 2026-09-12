"""The full masked MoE path in Pallas Mosaic GPU -- the Hopper analog of
`flashinfer_cutedsl_moe_masked` (sglang/srt/layers/moe/flashinfer_cutedsl_moe.py).

The reference chains four device kernels per layer:

    scaled_fp4_grouped_quantize(hidden)                 -> a_q, a_sf
    grouped_gemm_nt_masked(a, w1) -> gateup             (l, m, 2n)
    silu_and_mul_scaled_nvfp4_experts_quantize(gateup)  -> d_q, d_sf
    grouped_gemm_nt_masked(d, w2)                       (l, m, k)

and this module does the same with the three Pallas kernels in `quantize_kernels.py`
and `masked_gemm.py`. See `quant.py` for how the NVFP4 scheme is mapped onto Hopper.

Global scales are inputs, as in the reference: they come from calibration, not from the
activations of the current step, so no kernel has to reduce over the whole tensor.

The two-level scales are folded into the GEMM's alpha. `masked_grouped_gemm` computes

    out = alpha_eff * sum_kb sfa * sfb * (qa . qb)

while the mathematical result wants each operand dequantized as `q * sf / gs`, so

    alpha_eff = alpha / (gs_a * gs_b)

which is exactly what the reference folds into its own alpha.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from masked_gemm import GemmConfig, masked_grouped_gemm
from quantize_kernels import quantize_blockwise_pallas, silu_mul_quantize_pallas


def moe_masked(
    hidden: jax.Array,  # (l, m, k) bf16 -- m is the per-expert capacity
    w1_q: jax.Array,  # (l, 2n, k) e4m3
    w1_sf: jax.Array,  # (l, k // BLOCK_K, 2n) e4m3
    w1_gs: jax.Array,  # (l,) f32
    w1_alpha: jax.Array,  # (l,) f32
    w2_q: jax.Array,  # (l, k, n) e4m3
    w2_sf: jax.Array,  # (l, n // BLOCK_K, k) e4m3
    w2_gs: jax.Array,  # (l,) f32
    w2_alpha: jax.Array,  # (l,) f32
    masked_m: jax.Array,  # (l,) int32 -- valid rows per expert
    input_global_scale: jax.Array,  # (l,) f32
    a2_global_scale: jax.Array,  # (l,) f32
    gemm1_config: GemmConfig = GemmConfig(),
    gemm2_config: GemmConfig = GemmConfig(),
) -> jax.Array:
    """(l, m, k) bf16 in, (l, m, k) bf16 out. Rows >= masked_m[l] are undefined."""
    a_q, a_sf = quantize_blockwise_pallas(hidden, input_global_scale, masked_m)
    gateup = masked_grouped_gemm(
        a_q, a_sf, w1_q, w1_sf,
        w1_alpha / (input_global_scale * w1_gs), masked_m, gemm1_config,
    )
    d_q, d_sf = silu_mul_quantize_pallas(gateup, a2_global_scale, masked_m)
    return masked_grouped_gemm(
        d_q, d_sf, w2_q, w2_sf,
        w2_alpha / (a2_global_scale * w2_gs), masked_m, gemm2_config,
    )


def make_moe_inputs(key, l: int, m: int, k: int, n: int, masked_m: jax.Array):
    """Random weights in the quantized form both the kernel and `moe_reference` take.

    The per-expert alphas are the routing weights; the global scales are calibrated,
    which here means `a2_global_scale` is derived from a reference forward pass, the
    way a serving stack would derive it from profiling data.
    """
    from quant import global_scale_for, masked_grouped_gemm_reference, quantize_blockwise, silu_mul

    kh, k1, k2, ka = jax.random.split(key, 4)
    hidden = (jax.random.normal(kh, (l, m, k), jnp.float32)).astype(jnp.bfloat16)
    w1 = jax.random.normal(k1, (l, 2 * n, k), jnp.float32) * 0.05
    w2 = jax.random.normal(k2, (l, k, n), jnp.float32) * 0.05

    w1_gs, w2_gs = global_scale_for(w1), global_scale_for(w2)
    w1_q, w1_sf = quantize_blockwise(w1, w1_gs)
    w2_q, w2_sf = quantize_blockwise(w2, w2_gs)
    w1_alpha, w2_alpha = jax.random.uniform(ka, (2, l), jnp.float32, 0.5, 1.5)

    input_gs = global_scale_for(hidden.astype(jnp.float32))
    a_q, a_sf = quantize_blockwise(hidden.astype(jnp.float32), input_gs)
    gateup = masked_grouped_gemm_reference(
        a_q, a_sf, input_gs, w1_q, w1_sf, w1_gs, w1_alpha, masked_m, jnp.bfloat16
    )
    a2_gs = global_scale_for(silu_mul(gateup))

    return dict(
        hidden=hidden,
        w1_q=w1_q, w1_sf=w1_sf, w1_gs=w1_gs, w1_alpha=w1_alpha,
        w2_q=w2_q, w2_sf=w2_sf, w2_gs=w2_gs, w2_alpha=w2_alpha,
        masked_m=masked_m, input_global_scale=input_gs, a2_global_scale=a2_gs,
    )
