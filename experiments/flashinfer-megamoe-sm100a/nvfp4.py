"""NVFP4 semantics for the Blackwell port, and an fp32 reference for every stage.

This is the real thing the sm_90a experiment was an analog of: e2m1 (fp4) data with
per-16-element e4m3 block scales and a per-expert global scale, exactly as
FlashInfer's `scaled_fp4_grouped_quantize` / `grouped_gemm_nt_masked` produce and
consume them:

    x[i, k] ~= q[i, k] * (sf[i, k // 16] / global_scale)     q: e2m1, sf: e4m3

Blackwell applies the block scales inside `tcgen05.mma.kind.block_scale`, so unlike
the Hopper analog the kernel never reads the accumulator mid-K.

Everything in this module runs on any backend -- it is plain JAX, and it is what the
kernels are checked against.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

SF_VEC_SIZE = 16  # elements per block scale; the MMA's e4m3 scale mode requires 16
FP4_DTYPE = jnp.float4_e2m1fn
FP4_MAX = 6.0  # max finite e2m1
SF_DTYPE = jnp.float8_e4m3fn
SF_MAX = 448.0  # max finite e4m3


def global_scale_for(x: jax.Array) -> jax.Array:
    """Per-expert global scale, FlashInfer's `(448 * 6) / amax` recipe.

    It maps the tensor's amax so the per-block scales land inside e4m3's range: a block
    scale is `amax_block / FP4_MAX * global`, and `amax_block <= amax`.
    """
    amax = jnp.max(jnp.abs(x.astype(jnp.float32)), axis=(-2, -1))
    return (FP4_MAX * SF_MAX) / jnp.where(amax == 0, 1.0, amax)


def quantize_nvfp4(x: jax.Array, global_scale: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(l, m, k) -> (q: e2m1 (l, m, k), sf: e4m3 (l, m, k // 16)).

    Scales stay row-major here, which is the layout FlashInfer documents for its
    quantize kernels (`[num_experts, m, k // 16]`). The MMA wants them tiled instead;
    `to_mma_scale_layout` does that separately, so the two concerns stay testable apart.
    """
    l, m, k = x.shape
    if k % SF_VEC_SIZE:
        raise ValueError(f"K={k} must be a multiple of SF_VEC_SIZE={SF_VEC_SIZE}")
    xf = x.astype(jnp.float32).reshape(l, m, k // SF_VEC_SIZE, SF_VEC_SIZE)
    amax = jnp.max(jnp.abs(xf), axis=-1)
    gs = global_scale.reshape(l, 1, 1)
    sf = jnp.clip(amax / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE)
    step = sf.astype(jnp.float32) / gs  # the dequantization step actually represented
    safe = jnp.where(step == 0.0, 1.0, step)
    q = jnp.clip(xf / safe[..., None] * jnp.where(step > 0.0, 1.0, 0.0)[..., None],
                 -FP4_MAX, FP4_MAX)
    return q.astype(FP4_DTYPE).reshape(l, m, k), sf


def dequantize_nvfp4(q: jax.Array, sf: jax.Array, global_scale: jax.Array) -> jax.Array:
    """Inverse of `quantize_nvfp4`, in fp32."""
    l, m, k = q.shape
    step = sf.astype(jnp.float32) / global_scale.reshape(l, 1, 1)
    qf = q.astype(jnp.float32).reshape(l, m, k // SF_VEC_SIZE, SF_VEC_SIZE)
    return (qf * step[..., None]).reshape(l, m, k)


def masked_grouped_gemm_reference(
    a_q: jax.Array, a_sf: jax.Array, a_gs: jax.Array,
    b_q: jax.Array, b_sf: jax.Array, b_gs: jax.Array,
    alpha: jax.Array, masked_m: jax.Array, out_dtype=jnp.bfloat16,
) -> jax.Array:
    """out[l, m, n] = alpha[l] * dequant(A)[l] @ dequant(B)[l].T, fp32, masked per expert."""
    a = dequantize_nvfp4(a_q, a_sf, a_gs)
    b = dequantize_nvfp4(b_q, b_sf, b_gs)
    out = jnp.einsum("lmk,lnk->lmn", a, b) * alpha.reshape(-1, 1, 1)
    rows = jnp.arange(out.shape[1])[None, :, None]
    return jnp.where(rows < masked_m.reshape(-1, 1, 1), out, 0.0).astype(out_dtype)


def silu_mul(gateup: jax.Array) -> jax.Array:
    """SiLU-and-mul over the last axis: (l, m, 2n) -> (l, m, n), gate half first.

    Returns fp32: the reference fuses this into the quantize kernel, so the activation
    is never rounded to bf16 on the way.
    """
    gate, up = jnp.split(gateup.astype(jnp.float32), 2, axis=-1)
    return jax.nn.sigmoid(gate) * gate * up


def moe_reference(
    hidden: jax.Array,  # (l, m, k) bf16
    w1_q: jax.Array, w1_sf: jax.Array, w1_gs: jax.Array, w1_alpha: jax.Array,  # (l, 2n, k)
    w2_q: jax.Array, w2_sf: jax.Array, w2_gs: jax.Array, w2_alpha: jax.Array,  # (l, k, n)
    masked_m: jax.Array,
    input_global_scale: jax.Array,
    a2_global_scale: jax.Array,
) -> jax.Array:
    """`flashinfer_cutedsl_moe_masked` in fp32: quantize -> GEMM1 -> silu_and_mul +
    quantize -> GEMM2, with per-expert alphas."""
    a_q, a_sf = quantize_nvfp4(hidden, input_global_scale)
    gateup = masked_grouped_gemm_reference(
        a_q, a_sf, input_global_scale, w1_q, w1_sf, w1_gs, w1_alpha, masked_m, jnp.bfloat16
    )
    d_q, d_sf = quantize_nvfp4(silu_mul(gateup), a2_global_scale)
    return masked_grouped_gemm_reference(
        d_q, d_sf, a2_global_scale, w2_q, w2_sf, w2_gs, w2_alpha, masked_m, jnp.bfloat16
    )


# --- scale layout for tcgen05's block-scaled MMA -----------------------------------
#
# `async_copy_scales_to_tmem` reads scales from smem in the `.scale_vec::1X` tiling
# (PTX ISA, "tcgen05 MMA scale factor A layout 1x"), not row-major. Mosaic documents
# the transform from a row-major (MN, K // SF_VEC_SIZE) array; these two functions are
# that transform and its inverse, kept here so they can be tested without a Blackwell
# GPU -- the tiling is pure index arithmetic and is the easiest part of this port to
# get wrong.


def to_mma_scale_layout(sf: jax.Array) -> jax.Array:
    """(mn, k_scales) e4m3 -> (mn // 128, k_scales // 4, 32, 16), the smem layout that
    `async_copy_scales_to_tmem` expects for a TMEM ref of shape (mn, k_scales)."""
    mn, ks = sf.shape
    pad_mn = (mn + 127) // 128 * 128
    if ks % 4:
        raise ValueError(f"k_scales={ks} must be a multiple of 4")
    return (jnp.pad(sf, ((0, pad_mn - mn), (0, 0)))
            .reshape(pad_mn // 128, 4, 32, ks // 4, 4)
            .transpose(0, 3, 2, 1, 4)
            .reshape(pad_mn // 128, ks // 4, 32, 16))


def from_mma_scale_layout(tiled: jax.Array, mn: int) -> jax.Array:
    """Inverse of `to_mma_scale_layout`, dropping the padding rows."""
    mn_tiles, k_tiles, _, _ = tiled.shape
    return (tiled.reshape(mn_tiles, k_tiles, 32, 4, 4)
            .transpose(0, 3, 2, 1, 4)
            .reshape(mn_tiles * 128, k_tiles * 4)[:mn])
