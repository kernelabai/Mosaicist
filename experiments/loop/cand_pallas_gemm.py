"""Capture module for the candidate: the Pallas masked grouped GEMM, with knobs.

`KNOBS` declares the space the tuner searches; `build()` reads the setting the loop put
in the environment. Same problem shape as the reference module beside it.
"""

import pathlib
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "flashinfer-megamoe-sm100a"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from masked_gemm import GemmConfig, masked_grouped_gemm  # noqa: E402
from masked_gemm_ws import masked_grouped_gemm_w1  # noqa: E402
from mosaicist.converge.knobs import from_env  # noqa: E402
from nvfp4 import quantize_nvfp4, to_mma_scale_layout  # noqa: E402

L, M, K, N = 8, 512, 2048, 2048

#: the space the tuner walks, best-guess value first
KNOBS = {
    "block_k": [128, 256, 512],
    "stages": [1, 2, 3, 4],
    "warp_split": [False, True],
    "collective": [False, True],
}
#: knobs that swap in a different kernel rather than retune this one. The tuner leaves
#: these alone; the rewriter reaches for them once the parameter knobs are spent, which
#: is the design's knobs-before-rewrites ordering made concrete.
STRUCTURAL = ("warp_split", "collective")

DEFAULTS = {"block_k": 128, "stages": 1, "warp_split": False, "collective": False}


def build():
    k = from_env(DEFAULTS)
    cfg = GemmConfig(block_k=k["block_k"], stages=k["stages"], collective=k["collective"])
    gemm = masked_grouped_gemm_w1 if k["warp_split"] else masked_grouped_gemm

    gs = jnp.full((L,), 64.0, jnp.float32)
    a_q, a_sf = quantize_nvfp4(
        jax.random.normal(jax.random.key(0), (L, M, K), jnp.float32), gs)
    b_q, b_sf = quantize_nvfp4(
        jax.random.normal(jax.random.key(1), (L, N, K), jnp.float32) * 0.05, gs)
    asf, bsf = jax.vmap(to_mma_scale_layout)(a_sf), jax.vmap(to_mma_scale_layout)(b_sf)
    alpha = jnp.ones((L,), jnp.float32)
    masked_m = jnp.full((L,), M, jnp.int32)

    fn = jax.jit(gemm, static_argnums=6)
    return (lambda: fn(a_q, asf, b_q, bsf, alpha, masked_m, cfg)), ()
