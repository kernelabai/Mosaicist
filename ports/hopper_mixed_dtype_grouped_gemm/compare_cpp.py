"""Interleaved CuTeDSL-vs-C++ comparison on identical problem sets.

    python compare_cpp.py --cpp ~/cutlass-main/build69/69_hopper_mixed_dtype_grouped_gemm --rounds 3

For each configuration a shared problem file (C++ `--benchmark` format) is written, both
implementations run on it alternately `--rounds` times, and median runtimes are compared.
Both use alpha=1, beta=0.5 so every group reads C (the C++ default randomizes them).
"""

import argparse
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# name, shape spec ("fixed" m n k | "random" n k), groups, mode (0 convert-only / 1 scale), c
CONFIGS = [
    ("fixed 16x 2048x5120x8192 convert", ("fixed", 2048, 5120, 8192), 16, 0, 512),
    ("fixed 16x 2048x5120x8192 c=512", ("fixed", 2048, 5120, 8192), 16, 1, 512),
    ("fixed 16x 4096x5120x8192 c=8192", ("fixed", 4096, 5120, 8192), 16, 1, 8192),
    ("random 6x (M<=1024)x2048x512", ("random", 2048, 512), 6, 1, 512),
    ("random 100x (M<=1024)x2048x512", ("random", 2048, 512), 100, 1, 512),
    ("fixed 100x 2048x512x512", ("fixed", 2048, 512, 512), 100, 1, 512),
    ("fixed 100x 128x128x512", ("fixed", 128, 128, 512), 100, 1, 512),
]


def problem_file(spec, groups, seed, path):
    rnd = random.Random(seed)
    lines = []
    for i in range(groups):
        if spec[0] == "fixed":
            m, n, k = spec[1:]
        else:
            m, n, k = 16 * rnd.randint(1, 64), spec[1], spec[2]
        lines.append(f"{i} {m}x{n}x{k}")
    path.write_text("\n".join(lines) + "\n")


def run(cmd, pattern):
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    m = re.search(pattern, text)
    ok = ("BIT-EXACT" in text) or ("Disposition: Passed" in text)
    return (float(m.group(1)) if m else None), ok, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--only", default="", help="substring filter on config names")
    args = ap.parse_args()
    tmp = HERE / "problems_tmp"
    tmp.mkdir(exist_ok=True)
    print(f"{'config':<36} {'C++ ms':>9} {'CuTeDSL ms':>11} {'ratio':>7}  correctness")
    for name, spec, groups, mode, c in CONFIGS:
        if args.only and args.only not in name:
            continue
        pf = tmp / (re.sub(r"[^a-z0-9]+", "_", name.lower()) + ".txt")
        problem_file(spec, groups, 2020, pf)
        cpp_cmd = [args.cpp, f"--benchmark={pf}", f"--mode={mode}", f"--c={c}", "--alpha=1", "--beta=0.5",
                   f"--iterations={args.iterations}"]
        cute_cmd = [sys.executable, str(HERE / "run_grouped.py"), "--benchmark", str(pf), "--c",
                    str(0 if mode == 0 else c), "--alpha", "1", "--beta", "0.5", "--iterations", str(args.iterations)]
        t_cpp, t_cute, oks = [], [], []
        for _ in range(args.rounds):
            t, ok, txt = run(cpp_cmd, r"Avg runtime\s*:\s*([0-9.]+) ms")
            t_cpp.append(t); oks.append(("C++", ok))
            t, ok, txt2 = run(cute_cmd, r"CuTeDSL avg runtime:\s*([0-9.]+) ms")
            t_cute.append(t); oks.append(("CuTeDSL", ok))
        a = statistics.median([t for t in t_cpp if t is not None]) if any(t_cpp) else float("nan")
        b = statistics.median([t for t in t_cute if t is not None]) if any(t_cute) else float("nan")
        good = all(ok for _, ok in oks)
        print(f"{name:<36} {a:>9.4f} {b:>11.4f} {b / a:>7.3f}  {'both verified' if good else oks}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
