"""Candidate capture: the Pallas MoE path, with the knobs the loop searches."""

import pathlib
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "flashinfer-megamoe-sm100a"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import moe as moe_mod  # noqa: E402
from gemm1_silu_quantize import FusedConfig  # noqa: E402
from masked_gemm import GemmConfig  # noqa: E402
from moe import make_moe_inputs  # noqa: E402
from mosaicist.converge.knobs import from_env  # noqa: E402

L, M, K, N = 8, 512, 2048, 1024

KNOBS = {
    "gemm2_block_k": [128, 256, 512],
    "gemm2_stages": [1, 2, 3],
    "fused_block_k": [128, 256],
    "fused_stages": [2, 3],
    "gemm2_persistent": [False, True],
    "fuse_gemm1": [True, False],
}
#: swap in a different kernel rather than retune this one
STRUCTURAL = ("gemm2_persistent", "fuse_gemm1")
DEFAULTS = {"gemm2_block_k": 512, "gemm2_stages": 1, "fused_block_k": 128,
            "fused_stages": 3, "gemm2_persistent": False, "fuse_gemm1": True}


def build():
    k = from_env(DEFAULTS)
    moe_mod.FUSE_GEMM1 = k["fuse_gemm1"]

    if k["gemm2_persistent"]:
        from masked_gemm_ws import masked_grouped_gemm_w1p as gemm2
    else:
        from masked_gemm_ws import masked_grouped_gemm_w1 as gemm2
    moe_mod.masked_grouped_gemm = gemm2

    masked_m = jnp.full((L,), M, jnp.int32)
    kw, _ = make_moe_inputs(jax.random.key(0), L, M, K, N, masked_m)
    kw["gemm2_config"] = GemmConfig(block_k=k["gemm2_block_k"], stages=k["gemm2_stages"])
    kw["fused_config"] = FusedConfig(block_k=k["fused_block_k"], stages=k["fused_stages"])

    # the configs are frozen dataclasses, not arrays: jit needs them marked static
    fn = jax.jit(moe_mod.moe_masked,
                 static_argnames=("gemm1_config", "gemm2_config", "fused_config"))
    return (lambda: fn(**kw)), ()
