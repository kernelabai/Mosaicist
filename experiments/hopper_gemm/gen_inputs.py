"""Generate the shared fp16 GEMM inputs both harnesses read.

    python gen_inputs.py --runs ~/runs --mnk 8192,8192,8192 --dist uniform --seed 0

Writes runs/inputs/a.npy (M, K) and b.npy (K, N), both float16 row-major, so
the CuTeDSL and Pallas kernels see bit-identical operands.
"""

import argparse
from pathlib import Path

import numpy as np

from mosaicist.verify import generate

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=Path, required=True)
ap.add_argument("--mnk", default="8192,8192,8192")
ap.add_argument("--dist", default="uniform")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

m, n, k = (int(x) for x in args.mnk.split(","))
out = args.runs / "inputs"
out.mkdir(parents=True, exist_ok=True)
np.save(out / "a.npy", generate((m, k), args.dist, args.seed, "f16", arg_index=0).astype(np.float16))
np.save(out / "b.npy", generate((k, n), args.dist, args.seed, "f16", arg_index=1).astype(np.float16))
print(f"wrote {out}/a.npy {(m, k)} and b.npy {(k, n)} ({args.dist}, seed {args.seed})")
