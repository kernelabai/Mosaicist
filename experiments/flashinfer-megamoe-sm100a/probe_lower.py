"""Which formulation of the NVFP4 quantize tile actually lowers for sm_100a?

Same bisect-by-variant approach as the sm_90a experiment's probe_layout.py, but the
oracle here is Mosaic's lowering rather than a run: no Blackwell device is available,
so "does it compile for sm_100a" is the strongest signal on offer.
"""
import functools
import sys

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.mosaic.gpu import core as mgpu_core

mgpu_core._infer_arch = lambda: (10, 0)

from nvfp4 import FP4_DTYPE, FP4_MAX, SF_DTYPE, SF_MAX, SF_VEC_SIZE  # noqa: E402

M = 128


SF_ROW = 64  # scale columns in the gmem tensor, so its row stride is TMA-legal (>=16B)


def run(name, tile_k, compute, sf_cols=None):
    nb = tile_k // SF_VEC_SIZE
    sf_cols = nb if sf_cols is None else sf_cols

    def body(x_gmem, q_gmem, sf_gmem):
        @functools.partial(
            pl.run_scoped,
            x_smem=plgpu.SMEM((M, tile_k), jnp.bfloat16),
            q_smem=plgpu.SMEM((M, tile_k), FP4_DTYPE),
            sf_smem=plgpu.SMEM((M, sf_cols), SF_DTYPE),
            barrier=plgpu.Barrier(),
        )
        def scoped(x_smem, q_smem, sf_smem, barrier):
            plgpu.copy_gmem_to_smem(x_gmem.at[:], x_smem, barrier)
            plgpu.barrier_wait(barrier)
            q, sf = compute(x_smem[...].astype(jnp.float32), jnp.float32(8.0), tile_k, nb)
            q_smem[...] = q
            sf_smem[...] = sf
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(q_smem, q_gmem.at[:])
            plgpu.copy_smem_to_gmem(sf_smem, sf_gmem.at[:, pl.ds(0, sf_cols)])
            plgpu.wait_smem_to_gmem(0)

    try:
        f = plgpu.kernel(body, out_shape=(jax.ShapeDtypeStruct((M, tile_k), FP4_DTYPE),
                                          jax.ShapeDtypeStruct((M, nb), SF_DTYPE)),
                         grid=(1,), grid_names=("i",))
        jax.jit(f).lower(jax.ShapeDtypeStruct((M, tile_k), jnp.bfloat16))
        print(f"PASS  {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name}: {' '.join(str(e).split())[:110]}")
        return False


def _finish(blocked, sf, gs):
    step = sf.astype(jnp.float32) / gs
    safe = jnp.where(step == 0.0, 1.0, step)
    keep = jnp.where(step > 0.0, 1.0, 0.0)
    return jnp.clip(blocked / safe[..., None] * keep[..., None], -FP4_MAX, FP4_MAX)


def v_reshape_cast(vals, gs, tk, nb):
    vals = plgpu.layout_cast(vals, plgpu.Layout.WGMMA)
    blocked = vals.reshape(M, nb, SF_VEC_SIZE)
    sf = jnp.clip(jnp.max(jnp.abs(blocked), -1) / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE)
    return _finish(blocked, sf, gs).reshape(M, tk).astype(FP4_DTYPE), sf


def v_reshape_nocast(vals, gs, tk, nb):
    blocked = vals.reshape(M, nb, SF_VEC_SIZE)
    sf = jnp.clip(jnp.max(jnp.abs(blocked), -1) / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE)
    return _finish(blocked, sf, gs).reshape(M, tk).astype(FP4_DTYPE), sf


def v_rowmax_cast(vals, gs, tk, nb):
    """tile_k == SF_VEC_SIZE: the block scale is just the row max (the sm_90a shape)."""
    vals = plgpu.layout_cast(vals, plgpu.Layout.WGMMA)
    amax = plgpu.layout_cast(jnp.max(jnp.abs(vals), -1), plgpu.Layout.WGMMA.reduce(1))
    sf = jnp.clip(amax / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE)
    step = sf.astype(jnp.float32) / gs
    safe, keep = jnp.where(step == 0.0, 1.0, step), jnp.where(step > 0.0, 1.0, 0.0)
    q = jnp.clip(vals / safe[:, None] * keep[:, None], -FP4_MAX, FP4_MAX)
    return q.astype(FP4_DTYPE), sf[:, None]


def v_rowmax_nocast(vals, gs, tk, nb):
    amax = jnp.max(jnp.abs(vals), -1)
    sf = jnp.clip(amax / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE)
    step = sf.astype(jnp.float32) / gs
    safe, keep = jnp.where(step == 0.0, 1.0, step), jnp.where(step > 0.0, 1.0, 0.0)
    q = jnp.clip(vals / safe[:, None] * keep[:, None], -FP4_MAX, FP4_MAX)
    return q.astype(FP4_DTYPE), sf[:, None]


def v_reshape_cast_result(vals, gs, tk, nb):
    """Annotate the reduced result rather than the input tile."""
    blocked = vals.reshape(M, nb, SF_VEC_SIZE)
    amax = plgpu.layout_cast(jnp.max(jnp.abs(blocked), -1), plgpu.Layout.WGMMA)
    sf = jnp.clip(amax / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE)
    return _finish(blocked, sf, gs).reshape(M, tk).astype(FP4_DTYPE), sf


def v_reshape_strided(vals, gs, tk, nb):
    """WG_STRIDED is the layout for plain elementwise work, which is all this is."""
    vals = plgpu.layout_cast(vals, plgpu.Layout.WG_STRIDED((M, tk), vec_size=8))
    blocked = vals.reshape(M, nb, SF_VEC_SIZE)
    sf = jnp.clip(jnp.max(jnp.abs(blocked), -1) / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE)
    return _finish(blocked, sf, gs).reshape(M, tk).astype(FP4_DTYPE), sf


def _assemble(vals, gs, tk, nb, amax_of):
    """Build the full-width divisor and the scale row without slicing either output.

    Neither a column-slice store into swizzled smem nor a concatenate of the per-block
    results survives lowering, so each block's contribution is added into a full-width
    array through a constant one-hot mask instead.
    """
    inv_full = jnp.zeros((M, tk), jnp.float32)
    sf_full = jnp.zeros((M, nb), jnp.float32)
    for b in range(nb):
        col = (jnp.arange(tk) // SF_VEC_SIZE == b).astype(jnp.float32)  # (tk,)
        slot = (jnp.arange(nb) == b).astype(jnp.float32)  # (nb,)
        amax = plgpu.layout_cast(amax_of(b, col), plgpu.Layout.WGMMA.reduce(1))
        sf = jnp.clip(amax / FP4_MAX * gs, 0.0, SF_MAX).astype(SF_DTYPE).astype(jnp.float32)
        step = sf / gs
        inv = jnp.where(step > 0.0, 1.0, 0.0) / jnp.where(step == 0.0, 1.0, step)
        inv_full += inv[:, None] * col[None, :]
        sf_full += sf[:, None] * slot[None, :]
    q = jnp.clip(vals * inv_full, -FP4_MAX, FP4_MAX).astype(FP4_DTYPE)
    return q, sf_full.astype(SF_DTYPE)


def v_masked_reduce(vals, gs, tk, nb):
    """No slicing anywhere: a block's max is a full-row max of the masked tile."""
    vals = plgpu.layout_cast(vals, plgpu.Layout.WGMMA)
    absv = jnp.abs(vals)
    return _assemble(vals, gs, tk, nb, lambda b, col: jnp.max(absv * col[None, :], axis=-1))


def v_slice_reduce(vals, gs, tk, nb):
    """Reduce a 2D column slice -- cheaper, if slicing a register tile survives."""
    vals = plgpu.layout_cast(vals, plgpu.Layout.WGMMA)
    return _assemble(vals, gs, tk, nb, lambda b, col: jnp.max(
        jnp.abs(vals[:, b * SF_VEC_SIZE:(b + 1) * SF_VEC_SIZE]), axis=-1))


def v_no_scale_at_all(vals, gs, tk, nb):
    """Control: does an fp4 tile store plus a small e4m3 store lower at all?"""
    return vals.astype(FP4_DTYPE), jnp.zeros((M, nb), SF_DTYPE)


# tk=256 gives 16 scale columns; a narrower scale tile is rejected by TMA, which needs
# at least 128 bits along the last dimension.
ok = True
ok &= run("control: fp4 tile store, no reduction  (tk=256)", 256, v_no_scale_at_all)
ok &= run("reshape (M,nb,16) reduce, layout_cast  (tk=256)", 256, v_reshape_cast)
ok &= run("reshape (M,nb,16) reduce, no cast      (tk=256)", 256, v_reshape_nocast)
ok &= run("reshape reduce, cast result to WGMMA   (tk=256)", 256, v_reshape_cast_result)
ok &= run("reshape reduce, WG_STRIDED on tile     (tk=256)", 256, v_reshape_strided)
ok &= run("masked full-row reduce per block       (tk=256)", 256, v_masked_reduce)
ok &= run("sliced row reduce per block            (tk=256)", 256, v_slice_reduce)
sys.exit(0 if ok else 1)
