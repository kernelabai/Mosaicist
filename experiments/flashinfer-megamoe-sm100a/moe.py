"""The full masked MoE path on Blackwell -- `flashinfer_cutedsl_moe_masked` in Pallas.

UNVERIFIED on hardware; see `masked_gemm.py`. `check_lowering.py` shows the whole path
builds for sm_100a.

    quantize_nvfp4_pallas(hidden)             -> a_q, a_sf
    masked_grouped_gemm(a, w1)                -> gateup   (l, m, 2n)
    silu_mul_quantize_nvfp4_pallas(gateup)    -> d_q, d_sf
    masked_grouped_gemm(d, w2)                            (l, m, k)

Global scales are inputs, as in the reference -- they come from calibration, not from
the current step's activations, so no kernel reduces over a whole tensor.

The two-level scales are folded into each GEMM's alpha. `masked_grouped_gemm` computes
`alpha_eff * sum sfa * sfb * (qa . qb)`, while the mathematical result dequantizes each
operand as `q * sf / gs`, so `alpha_eff = alpha / (gs_a * gs_b)`.

The scales are re-tiled into the MMA's layout between stages rather than written that
way by the quantize kernels; `quantize_kernels.py` explains why. It is a pass over a
tensor 1/16 the size of the data, and a production version would fuse it away.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from masked_gemm import GemmConfig
from masked_gemm_ws import masked_grouped_gemm_w1 as masked_grouped_gemm
from nvfp4 import to_mma_scale_layout
from quantize_kernels import quantize_nvfp4_pallas, silu_mul_quantize_nvfp4_pallas

_retile = jax.vmap(to_mma_scale_layout)  # over the expert axis


def moe_masked(
    hidden: jax.Array,  # (l, m, k) bf16 -- m is the per-expert capacity
    w1_q: jax.Array,  # (l, 2n, k) e2m1
    w1_sf: jax.Array,  # (l, 2n // 128, k // 64, 32, 16) e4m3, MMA layout
    w1_gs: jax.Array,  # (l,) f32
    w1_alpha: jax.Array,  # (l,) f32
    w2_q: jax.Array,  # (l, k, n) e2m1
    w2_sf: jax.Array,  # (l, k // 128, n // 64, 32, 16) e4m3, MMA layout
    w2_gs: jax.Array,  # (l,) f32
    w2_alpha: jax.Array,  # (l,) f32
    masked_m: jax.Array,  # (l,) int32
    input_global_scale: jax.Array,  # (l,) f32
    a2_global_scale: jax.Array,  # (l,) f32
    gemm1_config: GemmConfig = GemmConfig(),
    gemm2_config: GemmConfig = GemmConfig(),
) -> jax.Array:
    """(l, m, k) bf16 in, (l, m, k) bf16 out. Rows >= masked_m[l] are undefined."""
    a_q, a_sf = quantize_nvfp4_pallas(hidden, input_global_scale, masked_m)
    gateup = masked_grouped_gemm(
        a_q, _retile(a_sf), w1_q, w1_sf,
        w1_alpha / (input_global_scale * w1_gs), masked_m, gemm1_config,
    )
    d_q, d_sf = silu_mul_quantize_nvfp4_pallas(gateup, a2_global_scale, masked_m)
    return masked_grouped_gemm(
        d_q, _retile(d_sf), w2_q, w2_sf,
        w2_alpha / (a2_global_scale * w2_gs), masked_m, gemm2_config,
    )


def make_moe_inputs(key, l: int, m: int, k: int, n: int, masked_m: jax.Array):
    """Random weights in the quantized form both `moe_masked` and `nvfp4.moe_reference`
    take. The reference wants row-major scales and the kernel wants them tiled, so both
    are returned; `a2_global_scale` is derived from a reference forward pass, the way a
    serving stack would derive it from calibration.
    """
    from nvfp4 import (global_scale_for, masked_grouped_gemm_reference, quantize_nvfp4,
                       silu_mul)

    kh, k1, k2, ka = jax.random.split(key, 4)
    hidden = jax.random.normal(kh, (l, m, k), jnp.float32).astype(jnp.bfloat16)
    w1 = jax.random.normal(k1, (l, 2 * n, k), jnp.float32) * 0.05
    w2 = jax.random.normal(k2, (l, k, n), jnp.float32) * 0.05

    w1_gs, w2_gs = global_scale_for(w1), global_scale_for(w2)
    w1_q, w1_sf = quantize_nvfp4(w1, w1_gs)
    w2_q, w2_sf = quantize_nvfp4(w2, w2_gs)
    w1_alpha, w2_alpha = jax.random.uniform(ka, (2, l), jnp.float32, 0.5, 1.5)

    input_gs = global_scale_for(hidden.astype(jnp.float32))
    a_q, a_sf = quantize_nvfp4(hidden.astype(jnp.float32), input_gs)
    gateup = masked_grouped_gemm_reference(
        a_q, a_sf, input_gs, w1_q, w1_sf, w1_gs, w1_alpha, masked_m, jnp.bfloat16
    )
    a2_gs = global_scale_for(silu_mul(gateup))

    reference = dict(
        hidden=hidden,
        w1_q=w1_q, w1_sf=w1_sf, w1_gs=w1_gs, w1_alpha=w1_alpha,
        w2_q=w2_q, w2_sf=w2_sf, w2_gs=w2_gs, w2_alpha=w2_alpha,
        masked_m=masked_m, input_global_scale=input_gs, a2_global_scale=a2_gs,
    )
    kernel = dict(reference, w1_sf=_retile(w1_sf), w2_sf=_retile(w2_sf))
    return kernel, reference
