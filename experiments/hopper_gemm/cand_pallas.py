"""Candidate capture: Pallas Mosaic GPU Hopper matmul (jax.experimental.pallas.ops.gpu.hopper_matmul_mgpu).

    python cand_pallas.py --runs ~/runs --name cand --tile-m 64 --tile-n 256 --wg-dim M --cluster-dim M

Writes runs/<name>/: kernel.ptx, ptxas.log (pinned assembler), sass.txt,
out.npy, and bundle.json with the CUPTI launch record and device-time samples.
"""

import argparse
import glob
import os
import shutil
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=Path, required=True)
ap.add_argument("--name", default="cand")
ap.add_argument("--tile-m", type=int, default=64)
ap.add_argument("--tile-n", type=int, default=256)
ap.add_argument("--tile-k", type=int, default=64)
ap.add_argument("--steps", type=int, default=4, help="max_concurrent_steps")
ap.add_argument("--wg-dim", choices=["M", "N"], default="M")
ap.add_argument("--cluster-dim", choices=["M", "N", "none"], default="M")
ap.add_argument("--grid-minor", choices=["M", "N"], default="N")
ap.add_argument("--grid-tile-width", type=int, default=1)
ap.add_argument("--delay-release", type=int, default=0,
                help="wgmma groups kept in flight (the kernel's delay_release; matmul() doesn't expose it)")
ap.add_argument("--ptxas", default="ptxas", help="pinned assembler used for both kernels")
ap.add_argument("--nvdisasm", default="nvdisasm")
ap.add_argument("--reps", type=int, default=50)
args = ap.parse_args()

out = args.runs / args.name
dump = out / "dump"
shutil.rmtree(dump, ignore_errors=True)
dump.mkdir(parents=True)
os.environ.update(MOSAIC_GPU_DUMP_PTX="1", MOSAIC_GPU_DUMP_PTXAS="1", MOSAIC_GPU_DUMP_TO=str(dump))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import functools  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental.pallas.ops.gpu import hopper_matmul_mgpu as hm  # noqa: E402

from mosaicist.bench.cupti_trace import time_kernel  # noqa: E402
from mosaicist.bundle import Bundle, Launch  # noqa: E402

if args.delay_release:
    # matmul() binds the module-level `kernel` at call time; thread delay_release through it.
    hm.kernel = functools.partial(hm.kernel, delay_release=args.delay_release)

D = hm.MatmulDimension
config = hm.TuningConfig(
    tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k, max_concurrent_steps=args.steps,
    epi_tile_n=64, epi_tile_m=64, grid_minor_dim=D[args.grid_minor], grid_tile_width=args.grid_tile_width,
    wg_dimension=D[args.wg_dim], cluster_dimension=None if args.cluster_dim == "none" else D[args.cluster_dim],
)
a = jnp.asarray(np.load(args.runs / "inputs" / "a.npy"))
b = jnp.asarray(np.load(args.runs / "inputs" / "b.npy"))
m, k = a.shape
n = b.shape[1]
f = jax.jit(functools.partial(hm.matmul, config=config), static_argnums=())
result = f(a, b, None).block_until_ready()
np.save(out / "out.npy", np.asarray(result))

held = {}
flush_buf = jnp.zeros(128 << 20, dtype=jnp.uint8)
flush_fn = jax.jit(lambda x: x + 1)


def run():
    held["c"] = f(a, b, None)


def flush():
    held["f"] = flush_fn(flush_buf)


def sync():
    for v in held.values():
        v.block_until_ready()


times, rec = time_kernel(run, sync, flush, reps=args.reps)

ptxs = glob.glob(str(dump / "**" / "*.ptx"), recursive=True)
if not ptxs:
    raise SystemExit(f"no PTX found under {dump}: {os.listdir(dump)}")
shutil.copy(max(ptxs, key=os.path.getsize), out / "kernel.ptx")
log = subprocess.run([args.ptxas, "-arch=sm_90a", "-v", str(out / "kernel.ptx"), "-o", str(out / "pinned.cubin")],
                     capture_output=True, text=True)
(out / "ptxas.log").write_text(log.stdout + log.stderr)
sass = subprocess.run([args.nvdisasm, "-c", str(out / "pinned.cubin")], capture_output=True, text=True)
(out / "sass.txt").write_text(sass.stdout)

Bundle(
    kind="candidate", name=f"pallas_hopper_matmul_{args.name}", arch="sm_90a", ptx="kernel.ptx",
    ptxas_log="ptxas.log", sass="sass.txt" if sass.returncode == 0 else None, source=hm.__file__,
    launch=Launch(**rec.launch()), timings=times,
    resources={"registers": rec.registers, "static_smem": rec.static_smem,
               "local_mem_per_thread": rec.local_mem_per_thread},
    versions={"jax": jax.__version__, "config": repr(config), "delay_release": str(args.delay_release)},
).save(out)
t = sorted(times)[len(times) // 2]
print(f"candidate {args.name}: {rec.name[:60]} grid={rec.grid} block={rec.block} cluster={rec.cluster} "
      f"regs={rec.registers} dyn_smem={rec.dynamic_smem}  median {t:.1f} us  "
      f"{2 * m * n * k / t / 1e6:.0f} TFLOP/s")
