"""Reference capture: CuTeDSL Hopper dense GEMM (examples/python/CuTeDSL/.../dense_gemm.py).

    python ref_cutedsl.py --runs ~/runs --example-dir ~/cutlass/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm \
        --tile 128,256 --cluster 2,1 --ptxas /path/to/pinned/ptxas

Writes runs/ref/: kernel.ptx, ptxas.log (pinned assembler), out.npy,
oracle.npy (fp64, computed on the GPU), and bundle.json with the CUPTI launch
record and device-time samples.
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=Path, required=True)
ap.add_argument("--example-dir", type=Path, required=True)
ap.add_argument("--name", default="ref", help="output directory under --runs (use e.g. ref_aa for an A/A run)")
ap.add_argument("--tile", default="128,256")
ap.add_argument("--cluster", default="2,1")
ap.add_argument("--ptxas", default="ptxas", help="pinned assembler used for both kernels")
ap.add_argument("--nvdisasm", default="nvdisasm")
ap.add_argument("--reps", type=int, default=50)
args = ap.parse_args()

out = args.runs / args.name
out.mkdir(parents=True, exist_ok=True)
for old in glob.glob(str(out / "*.ptx")) + glob.glob(str(out / "*.cubin")):
    os.remove(old)
os.environ["CUTE_DSL_KEEP_PTX"] = "1"
os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
os.environ["CUTE_DSL_DUMP_DIR"] = str(out)

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

from mosaicist.bench.cupti_trace import time_kernel  # noqa: E402
from mosaicist.bundle import Bundle, Launch  # noqa: E402

sys.path.insert(0, str(args.example_dir))
from dense_gemm import HopperWgmmaGemmKernel  # noqa: E402

a = torch.from_numpy(np.load(args.runs / "inputs" / "a.npy")).cuda()  # (M, K)
b = torch.from_numpy(np.load(args.runs / "inputs" / "b.npy")).cuda()  # (K, N)
m, k = a.shape
n = b.shape[1]
c = torch.empty((m, n), dtype=torch.float16, device="cuda")

# The example's (mode0, mode1, L) convention: A (m,k,l) k-major, B (n,k,l) n-major, C (m,n,l) n-major.
mA = from_dlpack(a.unsqueeze(0).permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)
mB = from_dlpack(b.unsqueeze(0).permute(2, 1, 0), assumed_align=16).mark_layout_dynamic(leading_dim=0)
mC = from_dlpack(c.unsqueeze(0).permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)

tile = tuple(int(x) for x in args.tile.split(","))
cluster = tuple(int(x) for x in args.cluster.split(","))
gemm = HopperWgmmaGemmKernel(cutlass.Float32, tile, cluster)
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
compiled = cute.compile(gemm, mA, mB, mC, stream)

compiled(mA, mB, mC, stream)
torch.cuda.synchronize()
np.save(out / "out.npy", c.cpu().numpy())
np.save(out / "oracle.npy", (a.double() @ b.double()).cpu().numpy())

flush_buf = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")  # > 2x H100 L2
times, rec = time_kernel(lambda: compiled(mA, mB, mC, stream), torch.cuda.synchronize,
                         lambda: flush_buf.zero_(), reps=args.reps)

ptx = max(glob.glob(str(out / "*.ptx")), key=os.path.getsize)
os.replace(ptx, out / "kernel.ptx")
log = subprocess.run([args.ptxas, "-arch=sm_90a", "-v", str(out / "kernel.ptx"), "-o", str(out / "pinned.cubin")],
                     capture_output=True, text=True)
(out / "ptxas.log").write_text(log.stdout + log.stderr)
sass = subprocess.run([args.nvdisasm, "-c", str(out / "pinned.cubin")], capture_output=True, text=True)
(out / "sass.txt").write_text(sass.stdout)

Bundle(
    kind="reference", name="cutedsl_hopper_dense_gemm", arch="sm_90a", ptx="kernel.ptx", ptxas_log="ptxas.log",
    sass="sass.txt" if sass.returncode == 0 else None,
    source=str(args.example_dir / "dense_gemm.py"), launch=Launch(**rec.launch()), timings=times,
    resources={"registers": rec.registers, "static_smem": rec.static_smem,
               "local_mem_per_thread": rec.local_mem_per_thread},
    versions={"cutlass_dsl": getattr(cutlass, "__version__", "?"), "torch": torch.__version__,
              "ptxas": log.stderr.strip().splitlines()[-1] if log.returncode else "ok",
              "config": f"tile={tile} cluster={cluster} mnk={(m, n, k)}"},
).save(out)
t = sorted(times)[len(times) // 2]
print(f"reference: {rec.name[:60]}... grid={rec.grid} block={rec.block} cluster={rec.cluster} "
      f"regs={rec.registers} dyn_smem={rec.dynamic_smem}  median {t:.1f} us  "
      f"{2 * m * n * k / t / 1e6:.0f} TFLOP/s")
