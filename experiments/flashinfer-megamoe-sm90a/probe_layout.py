"""Isolate which step of the quantize tile breaks Mosaic GPU layout inference."""

import functools
import sys
import traceback

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

M, K = 128, 128
FP8 = jnp.float8_e4m3fn


def run(name, compute, out_shapes, loader, sf_dtype=jnp.float32, with_scalar=False):
    def body(x_gmem, *rest):
        outs = rest[1:] if with_scalar else rest
        gs_gmem = rest[0] if with_scalar else None
        @functools.partial(
            pl.run_scoped,
            x_smem=plgpu.SMEM((M, K), jnp.bfloat16),
            q_smem=plgpu.SMEM((M, K), FP8),
            sf_smem=plgpu.SMEM((M,), sf_dtype),
            barrier=plgpu.Barrier(),
        )
        def scoped(x_smem, q_smem, sf_smem, barrier):
            plgpu.copy_gmem_to_smem(x_gmem.at[:], x_smem, barrier)
            plgpu.barrier_wait(barrier)
            if with_scalar:
                compute(loader(x_smem), q_smem, sf_smem, gs_gmem[0])
            else:
                compute(loader(x_smem), q_smem, sf_smem)
            plgpu.commit_smem()
            for ref, out in zip((q_smem, sf_smem), outs):
                plgpu.copy_smem_to_gmem(ref, out.at[:])
            plgpu.wait_smem_to_gmem(0)

    try:
        f = plgpu.kernel(body, out_shape=out_shapes, grid=(1,), grid_names=("i",))
        args = (jnp.ones((M, K), jnp.bfloat16),) + ((jnp.ones((1,), jnp.float32),) if with_scalar else ())
        jax.block_until_ready(f(*args))
        print(f"PASS  {name}")
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()
        print(f"FAIL  {name}: {msg[-1][:120] if msg else type(e).__name__}")
        return False


SF_ONLY = (jax.ShapeDtypeStruct((M, K), FP8), jax.ShapeDtypeStruct((M,), jnp.float32))

cast_load = lambda ref: plgpu.layout_cast(ref[...].astype(jnp.float32), plgpu.Layout.WGMMA)
plgpu_load = lambda ref: plgpu.load(ref, layout=plgpu.Layout.WGMMA).astype(jnp.float32)


def only_reduce(vals, q_smem, sf_smem):
    sf_smem[...] = plgpu.layout_cast(jnp.max(jnp.abs(vals), axis=-1), plgpu.Layout.WGMMA.reduce(1))
    q_smem[...] = vals.astype(FP8)


def reduce_and_broadcast(vals, q_smem, sf_smem):
    amax = plgpu.layout_cast(jnp.max(jnp.abs(vals), axis=-1), plgpu.Layout.WGMMA.reduce(1))
    sf_smem[...] = amax
    q_smem[...] = (vals / amax[:, None]).astype(FP8)


def broadcast_f32_out(vals, q_smem, sf_smem):
    amax = plgpu.layout_cast(jnp.max(jnp.abs(vals), axis=-1), plgpu.Layout.WGMMA.reduce(1))
    sf_smem[...] = amax
    q_smem[...] = (vals / amax[:, None]).astype(jnp.bfloat16).astype(FP8)


FP8_SF = (jax.ShapeDtypeStruct((M, K), FP8), jax.ShapeDtypeStruct((M,), FP8))


def fp8_scale_store(vals, q_smem, sf_smem):
    amax = plgpu.layout_cast(jnp.max(jnp.abs(vals), axis=-1), plgpu.Layout.WGMMA.reduce(1))
    sf_smem[...] = amax.astype(FP8)  # 8-bit reduced-layout store
    q_smem[...] = (vals / amax[:, None]).astype(FP8)


def with_gmem_scalar(vals, q_smem, sf_smem, gs):
    amax = plgpu.layout_cast(jnp.max(jnp.abs(vals), axis=-1), plgpu.Layout.WGMMA.reduce(1))
    sf_smem[...] = amax * gs  # scalar read from a GMEM ref
    q_smem[...] = (vals / amax[:, None]).astype(FP8)


def real_quantize_tile(vals, q_smem, sf_smem, gs):
    from quantize_kernels import _quantize_tile
    q, sf = _quantize_tile(vals, gs)
    q_smem[...] = q
    sf_smem[...] = sf


def clip_only(vals, q_smem, sf_smem, gs):
    amax = plgpu.layout_cast(jnp.max(jnp.abs(vals), axis=-1), plgpu.Layout.WGMMA.reduce(1))
    sf = jnp.clip(amax / 448.0 * gs, 0.0, 448.0).astype(FP8)
    sf_smem[...] = sf
    q_smem[...] = (vals / sf.astype(jnp.float32)[:, None]).astype(FP8)


def where_on_reduced(vals, q_smem, sf_smem, gs):
    amax = plgpu.layout_cast(jnp.max(jnp.abs(vals), axis=-1), plgpu.Layout.WGMMA.reduce(1))
    step = jnp.where(amax == 0.0, 1.0, amax)  # select on a reduced-layout vector
    sf_smem[...] = step
    q_smem[...] = (vals / step[:, None]).astype(FP8)


def where_broadcast_cond(vals, q_smem, sf_smem, gs):
    amax = plgpu.layout_cast(jnp.max(jnp.abs(vals), axis=-1), plgpu.Layout.WGMMA.reduce(1))
    sf_smem[...] = amax
    # condition broadcast from reduced layout to the full tile
    q_smem[...] = jnp.where(amax[:, None] > 0, vals / amax[:, None], 0.0).astype(FP8)


ok = True
ok &= run("clip + fp8 scale", clip_only, FP8_SF, cast_load, sf_dtype=FP8, with_scalar=True)
ok &= run("where on reduced vector", where_on_reduced, SF_ONLY, cast_load, with_scalar=True)
ok &= run("where with broadcast condition", where_broadcast_cond, SF_ONLY, cast_load, with_scalar=True)
ok &= run("real _quantize_tile", real_quantize_tile, FP8_SF, cast_load, sf_dtype=FP8, with_scalar=True)
ok &= run("fp8 scale-vector store", fp8_scale_store, FP8_SF, cast_load, sf_dtype=FP8)
ok &= run("scalar from gmem ref", with_gmem_scalar, SF_ONLY, cast_load, with_scalar=True)
for name, loader in (("ref[...] + cast", cast_load), ("plgpu.load", plgpu_load)):
    ok &= run(f"{name}: store fp8 tile + row max", only_reduce, SF_ONLY, loader)
    ok &= run(f"{name}: + broadcast divide", reduce_and_broadcast, SF_ONLY, loader)
    ok &= run(f"{name}: + via bf16", broadcast_f32_out, SF_ONLY, loader)
sys.exit(0 if ok else 1)
