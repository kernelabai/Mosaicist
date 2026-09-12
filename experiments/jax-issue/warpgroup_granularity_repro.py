"""Mosaic GPU block sizes come in whole warpgroups, so a 6-warp kernel is unreachable.

CUTLASS/CuTeDSL Blackwell GEMMs commonly launch 192 threads: four warps for the
epilogue, one issuing TMAs, one issuing MMAs. `plgpu.kernel` takes `num_threads`, but
that counts *warpgroups*, so the only reachable block sizes are 128, 256, 384 ...

This prints the block size Mosaic actually launches for each `num_threads`.
"""

import sys

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

sys.path.insert(0, "../../src")
from mosaicist.bench.cupti_trace import KernelTrace  # noqa: E402

M = 128


def launched_block(num_threads):
    """Run a trivial kernel and report the block size the driver actually saw."""

    def body(x_gmem, o_gmem, smem, barrier):
        # with several warpgroups only one may issue the copies
        lead = (lax.axis_index("wg") == 0) if num_threads else True

        @pl.when(lead)
        def _():
            plgpu.copy_gmem_to_smem(x_gmem.at[:], smem, barrier)
            plgpu.barrier_wait(barrier)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(smem, o_gmem.at[:])
            plgpu.wait_smem_to_gmem(0)

    kwargs = {}
    if num_threads is not None:
        kwargs = dict(num_threads=num_threads, thread_name="wg")
    f = plgpu.kernel(body, out_shape=jax.ShapeDtypeStruct((M, M), jnp.float32),
                     grid=(1,), grid_names=("i",),
                     scratch_shapes=[plgpu.SMEM((M, M), jnp.float32), plgpu.Barrier()],
                     **kwargs)
    x = jnp.ones((M, M), jnp.float32)
    jax.block_until_ready(jax.jit(f)(x))
    with KernelTrace() as tr:
        jax.block_until_ready(jax.jit(f)(x))
    recs = [r for r in tr.records if "mosaic" in r.name]
    return recs[0].block[0] if recs else None


print(f"jax {jax.__version__} on {jax.devices()[0].device_kind}\n")
print(f"{'num_threads':>12}  {'threads launched':>17}")
for nt in (None, 1, 2, 3):
    try:
        print(f"{str(nt):>12}  {str(launched_block(nt)):>17}")
    except Exception as e:  # noqa: BLE001
        print(f"{str(nt):>12}  {' '.join(str(e).split())[:60]}")
print("\nreachable block sizes are multiples of 128; 192 is not among them")
