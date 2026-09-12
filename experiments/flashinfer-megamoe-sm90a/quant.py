"""Quantization semantics for the Hopper analog of FlashInfer's NVFP4 masked MoE.

The reference (`flashinfer.gemm.grouped_gemm_nt_masked`, sm_100a) multiplies NVFP4
operands carrying two-level scales:

    x[i, k] ~= q[i, k] * (sf[i, k // 16] * global_scale)          q: e2m1, sf: e4m3

Blackwell applies those per-16-element scales inside `tcgen05.mma.kind.block_scale`.
Hopper has no block-scaled MMA and no fp4 tensor cores, so this analog keeps the
two-level scheme and changes only what the hardware forces:

    data      e2m1 (fp4)  ->  e4m3 (fp8), Hopper's native tensor-core type
    block     16 along K  ->  128 along K (scales are applied to the wgmma
                              accumulator, and one fp8 wgmma already spans k=32)
    scales    e4m3 + per-expert global scale   (unchanged)

With `BLOCK_K` scales, a GEMM becomes

    out[m, n] = alpha * gs_a * gs_b * sum_kb  sfa[m, kb] * sfb[n, kb]
                                             * sum_{k in kb} qa[m, k] * qb[n, k]

so the kernel accumulates one K block in fp32, scales it by the outer product of
the two scale vectors, and adds it to the running result.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

BLOCK_K = 128  # scale-factor vector size along K (the reference's sf_vec_size = 16)
FP8_DTYPE = jnp.float8_e4m3fn
FP8_MAX = 448.0  # max finite e4m3
SF_MAX = 448.0  # scales are stored in e4m3 as well


def global_scale_for(x: jax.Array) -> jax.Array:
    """Per-expert global scale, mirroring FlashInfer's `(448 * 6) / amax` recipe.

    It maps the tensor's amax so that the per-block scales land inside e4m3's range:
    a block scale is `amax_block / FP8_MAX / global`, and `amax_block <= amax`.
    """
    amax = jnp.max(jnp.abs(x.astype(jnp.float32)), axis=(-2, -1))
    return (FP8_MAX * SF_MAX) / jnp.where(amax == 0, 1.0, amax)


def quantize_blockwise(x: jax.Array, global_scale: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Quantize (l, m, k) -> (q: e4m3 (l, m, k), sf: e4m3 (l, k // BLOCK_K, m)).

    Scales are stored K-major (block index first, rows innermost) so that a tile's
    scale vector is contiguous: TMA cannot load the 1-byte inner dimension that the
    natural (m, k // BLOCK_K) layout would give.

    `x` is dequantized as `q * (sf * (1 / global_scale))`; `global_scale` is per expert,
    as in the reference, where the kernel folds it into alpha.
    """
    l, m, k = x.shape
    if k % BLOCK_K:
        raise ValueError(f"K={k} must be a multiple of BLOCK_K={BLOCK_K}")
    xf = x.astype(jnp.float32).reshape(l, m, k // BLOCK_K, BLOCK_K)
    amax = jnp.max(jnp.abs(xf), axis=-1)  # (l, m, k // BLOCK_K)
    gs = global_scale.reshape(l, 1, 1)
    # scale in units of 1/global, stored in e4m3 (this rounding is part of the format)
    sf = jnp.clip(amax / FP8_MAX * gs, 0.0, SF_MAX).astype(FP8_DTYPE)
    step = sf.astype(jnp.float32) / gs  # the dequantization step actually represented
    q = jnp.where(step[..., None] > 0, xf / jnp.where(step[..., None] == 0, 1.0, step[..., None]), 0.0)
    q = jnp.clip(q, -FP8_MAX, FP8_MAX).astype(FP8_DTYPE)
    return q.reshape(l, m, k), sf.transpose(0, 2, 1)  # (l, k // BLOCK_K, m)


def dequantize_blockwise(q: jax.Array, sf: jax.Array, global_scale: jax.Array) -> jax.Array:
    """Inverse of `quantize_blockwise`, in fp32."""
    l, m, k = q.shape
    step = sf.transpose(0, 2, 1).astype(jnp.float32) / global_scale.reshape(l, 1, 1)  # (l, m, k // BLOCK_K)
    return (q.astype(jnp.float32).reshape(l, m, k // BLOCK_K, BLOCK_K) * step[..., None]).reshape(l, m, k)


def masked_grouped_gemm_reference(
    a_q: jax.Array, a_sf: jax.Array, a_gs: jax.Array,
    b_q: jax.Array, b_sf: jax.Array, b_gs: jax.Array,
    alpha: jax.Array, masked_m: jax.Array, out_dtype=jnp.bfloat16,
) -> jax.Array:
    """out[l, m, n] = alpha[l] * dequant(A)[l] @ dequant(B)[l].T, in fp32, masked per expert.

    Rows >= masked_m[l] are zero here; the kernel leaves them untouched, so tests
    compare only valid rows.
    """
    a = dequantize_blockwise(a_q, a_sf, a_gs)
    b = dequantize_blockwise(b_q, b_sf, b_gs)
    out = jnp.einsum("lmk,lnk->lmn", a, b) * alpha.reshape(-1, 1, 1)
    rows = jnp.arange(out.shape[1])[None, :, None]
    return jnp.where(rows < masked_m.reshape(-1, 1, 1), out, 0.0).astype(out_dtype)


def silu_mul(gateup: jax.Array) -> jax.Array:
    """SiLU-and-mul over the last axis: (l, m, 2n) -> (l, m, n), as in the reference's
    `silu_and_mul_scaled_nvfp4_experts_quantize` (gate half first, then up).

    Returns fp32: the reference fuses this with the quantization, so the activation is
    never rounded to bf16 in between. GEMM1's bf16 output is the last rounding step.
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
    """The full masked MoE with this analog's quantization, evaluated in fp32.

    Mirrors `flashinfer_cutedsl_moe_masked`: quantize hidden -> GEMM1 -> silu_and_mul +
    quantize -> GEMM2, with per-expert alphas.
    """
    a_q, a_sf = quantize_blockwise(hidden, input_global_scale)
    gateup = masked_grouped_gemm_reference(
        a_q, a_sf, input_global_scale, w1_q, w1_sf, w1_gs, w1_alpha, masked_m, jnp.bfloat16
    )
    d_q, d_sf = quantize_blockwise(silu_mul(gateup), a2_global_scale)
    return masked_grouped_gemm_reference(
        d_q, d_sf, a2_global_scale, w2_q, w2_sf, w2_gs, w2_alpha, masked_m, jnp.bfloat16
    )
