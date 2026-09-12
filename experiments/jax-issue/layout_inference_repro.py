"""Minimal repro: Mosaic GPU layout inference cannot express block-wise quantization.

Block-scaled formats (NVFP4, MXFP4, fp8 with per-block scales) all need the same two
steps on a register tile:

    1. reduce over contiguous groups of V columns   -> one scale per group
    2. broadcast those scales back over their groups -> rescale the tile

Every natural spelling of that pair fails layout inference. The only formulation that
lowers reduces each group as a separate 2-D slice and then reassembles the full-width
result by accumulating through constant one-hot masks, which costs a full-tile operation
per group.

Run:  python layout_inference_repro.py
"""

import functools
import sys

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

M, K, VEC = 128, 256, 16  # tile rows, tile columns, elements per scale block
NB = K // VEC


def run(name, compute, out_dtype=jnp.bfloat16):
    def body(x_gmem, o_gmem, x_smem, o_smem, barrier):
        plgpu.copy_gmem_to_smem(x_gmem.at[:], x_smem, barrier)
        plgpu.barrier_wait(barrier)
        vals = plgpu.layout_cast(x_smem[...].astype(jnp.float32), plgpu.Layout.WGMMA)
        o_smem[...] = compute(vals).astype(out_dtype)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(o_smem, o_gmem.at[:])
        plgpu.wait_smem_to_gmem(0)

    f = plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((M, K), out_dtype),
        grid=(1,), grid_names=("i",),
        scratch_shapes=[plgpu.SMEM((M, K), jnp.bfloat16),
                        plgpu.SMEM((M, K), out_dtype),
                        plgpu.Barrier()],
    )
    try:
        jax.block_until_ready(jax.jit(f)(jnp.ones((M, K), jnp.bfloat16)))
        print(f"  OK    {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL  {name}\n          {' '.join(str(e).split())[:150]}")
        return False


def _safe(x):
    return jnp.where(x == 0.0, 1.0, x)


# --- the natural spellings, all of which fail --------------------------------------

def v_reshape_reduce(vals):
    """Split the minor axis into (blocks, VEC) and reduce the last one."""
    blocked = vals.reshape(M, NB, VEC)
    amax = jnp.max(jnp.abs(blocked), axis=-1)
    return (blocked / _safe(amax)[..., None]).reshape(M, K)


def v_reshape_reduce_annotated(vals):
    """As above, but annotating the reduced result."""
    blocked = vals.reshape(M, NB, VEC)
    amax = plgpu.layout_cast(jnp.max(jnp.abs(blocked), axis=-1), plgpu.Layout.WGMMA)
    return (blocked / _safe(amax)[..., None]).reshape(M, K)


def v_repeat(vals):
    """Reduce per slice (which works), then widen with jnp.repeat."""
    amax = jnp.concatenate(
        [plgpu.layout_cast(jnp.max(jnp.abs(vals[:, b * VEC:(b + 1) * VEC]), axis=-1),
                           plgpu.Layout.WGMMA.reduce(1))[:, None] for b in range(NB)],
        axis=-1)
    return vals / jnp.repeat(_safe(amax), VEC, axis=-1)


def v_broadcast(vals):
    """Widen with broadcast_to + reshape instead."""
    cols = [plgpu.layout_cast(jnp.max(jnp.abs(vals[:, b * VEC:(b + 1) * VEC]), axis=-1),
                              plgpu.Layout.WGMMA.reduce(1))[:, None] for b in range(NB)]
    amax = jnp.concatenate(cols, axis=-1)
    wide = jnp.broadcast_to(_safe(amax)[:, :, None], (M, NB, VEC)).reshape(M, K)
    return vals / wide


def v_concat(vals):
    """Never build a full-width divisor: scale each slice, concatenate the results."""
    outs = []
    for b in range(NB):
        blk = vals[:, b * VEC:(b + 1) * VEC]
        amax = plgpu.layout_cast(jnp.max(jnp.abs(blk), axis=-1),
                                 plgpu.Layout.WGMMA.reduce(1))
        outs.append(blk / _safe(amax)[:, None])
    return jnp.concatenate(outs, axis=-1)


def v_where_broadcast_cond(vals):
    """A select whose *condition* broadcasts from the reduced layout to the tile."""
    amax = plgpu.layout_cast(jnp.max(jnp.abs(vals), axis=-1), plgpu.Layout.WGMMA.reduce(1))
    return jnp.where(amax[:, None] > 0, vals / _safe(amax)[:, None], 0.0)


# --- the workaround that does lower -------------------------------------------------

def v_onehot_workaround(vals):
    """Reduce each slice separately, then accumulate the full-width divisor through
    constant one-hot masks. One full-tile operation per block: O(K/VEC * K) work to
    express what is logically a broadcast."""
    wide = jnp.zeros((M, K), jnp.float32)
    for b in range(NB):
        amax = plgpu.layout_cast(jnp.max(jnp.abs(vals[:, b * VEC:(b + 1) * VEC]), axis=-1),
                                 plgpu.Layout.WGMMA.reduce(1))
        mask = (jnp.arange(K) // VEC == b).astype(jnp.float32)
        wide += _safe(amax)[:, None] * mask[None, :]
    return vals / wide


# --- what a real block-scaled quantizer needs: values *and* the scale vector ----------

def _run2(name, compute):
    """Like run(), but the kernel also writes the (M, K//VEC) scale array."""
    def body(x_gmem, q_gmem, sf_gmem, x_smem, q_smem, sf_smem, barrier):
        plgpu.copy_gmem_to_smem(x_gmem.at[:], x_smem, barrier)
        plgpu.barrier_wait(barrier)
        vals = plgpu.layout_cast(x_smem[...].astype(jnp.float32), plgpu.Layout.WGMMA)
        q, sf = compute(vals)
        q_smem[...] = q.astype(jnp.float4_e2m1fn)
        sf_smem[...] = sf.astype(jnp.float8_e4m3fn)
        plgpu.commit_smem()
        plgpu.copy_smem_to_gmem(q_smem, q_gmem.at[:])
        plgpu.copy_smem_to_gmem(sf_smem, sf_gmem.at[:])
        plgpu.wait_smem_to_gmem(0)

    f = plgpu.kernel(
        body,
        out_shape=(jax.ShapeDtypeStruct((M, K), jnp.float4_e2m1fn),
                   jax.ShapeDtypeStruct((M, NB), jnp.float8_e4m3fn)),
        grid=(1,), grid_names=("i",),
        scratch_shapes=[plgpu.SMEM((M, K), jnp.bfloat16),
                        plgpu.SMEM((M, K), jnp.float4_e2m1fn),
                        plgpu.SMEM((M, NB), jnp.float8_e4m3fn),
                        plgpu.Barrier()])
    try:
        jax.block_until_ready(jax.jit(f)(jnp.ones((M, K), jnp.bfloat16)))
        print(f"  OK    {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL  {name}\n          {' '.join(str(e).split())[:150]}")
        return False


def _blocks(vals):
    out = []
    for b in range(NB):
        blk = vals[:, b * VEC:(b + 1) * VEC]
        amax = plgpu.layout_cast(jnp.max(jnp.abs(blk), axis=-1),
                                 plgpu.Layout.WGMMA.reduce(1))
        out.append((blk, amax))
    return out


def q_concat_sf_concat(vals):
    """Both outputs assembled by concatenation."""
    bs = _blocks(vals)
    q = jnp.concatenate([blk / _safe(a)[:, None] for blk, a in bs], axis=-1)
    sf = jnp.concatenate([a[:, None] for _, a in bs], axis=-1)
    return q, sf


def q_concat_sf_onehot(vals):
    """Values by concatenation; the narrow scale vector by one-hot accumulation."""
    bs = _blocks(vals)
    q = jnp.concatenate([blk / _safe(a)[:, None] for blk, a in bs], axis=-1)
    sf = jnp.zeros((M, NB), jnp.float32)
    for b, (_, a) in enumerate(bs):
        sf += a[:, None] * (jnp.arange(NB) == b).astype(jnp.float32)[None, :]
    return q, sf


def q_onehot_sf_onehot(vals):
    """What the port does today: a full-width divisor built from one-hot masks."""
    bs = _blocks(vals)
    wide = jnp.zeros((M, K), jnp.float32)
    sf = jnp.zeros((M, NB), jnp.float32)
    for b, (_, a) in enumerate(bs):
        wide += _safe(a)[:, None] * (jnp.arange(K) // VEC == b).astype(jnp.float32)[None, :]
        sf += a[:, None] * (jnp.arange(NB) == b).astype(jnp.float32)[None, :]
    return vals / wide, sf


dev = jax.devices()[0]
print(f"jax {jax.__version__} on {dev.device_kind} (sm_{dev.compute_capability})\n")
print("natural spellings:")
natural = [run("reshape to (M, K//VEC, VEC), reduce axis -1", v_reshape_reduce),
           run("... with layout_cast on the reduced result", v_reshape_reduce_annotated),
           run("per-slice reduce + jnp.repeat", v_repeat),
           run("per-slice reduce + broadcast_to + reshape", v_broadcast),
           run("per-slice reduce + scale + jnp.concatenate", v_concat),
           run("... same, but the e2m1 output the format needs", v_concat,
               jnp.float4_e2m1fn),
           run("reshape-reduce, e2m1 output", v_reshape_reduce, jnp.float4_e2m1fn),
           run("jnp.where with a broadcast condition", v_where_broadcast_cond)]
print("\nworkaround:")
work = run("per-slice reduce + one-hot mask accumulation", v_onehot_workaround,
           jnp.float4_e2m1fn)
print("\nvalues + scale vector together (what a quantizer actually emits):")
_run2("values by concatenate, scales by concatenate", q_concat_sf_concat)
_run2("values by concatenate, scales by one-hot", q_concat_sf_onehot)
_run2("both by one-hot (what the port does today)", q_onehot_sf_onehot)
print(f"\n{sum(natural)}/{len(natural)} natural spellings lower; workaround lowers: {work}")
sys.exit(0)
