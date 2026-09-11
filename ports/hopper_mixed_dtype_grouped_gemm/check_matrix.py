"""Correctness matrix for the grouped port (run on an sm_90a GPU).

Each case compiles the grouped kernel for its configuration and checks every group's D
bit-exactly against problems.reference.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CASES = [
    # (label, args)
    ("e5m2 scale c=512, 7 random M, random alpha/beta", ["--groups", "7"]),
    ("e5m2 convert-only, 5 random M", ["--groups", "5", "--c", "0"]),
    ("e4m3 scale c=512, 5 random M", ["--groups", "5", "--quant", "e4m3"]),
    ("e5m2 per-column scale c=K=1024", ["--groups", "4", "--k", "1024", "--c", "1024"]),
    ("e5m2 c=128 (many scale groups)", ["--groups", "3", "--k", "1024", "--c", "128"]),
    ("ragged M (not a tile multiple)", ["--benchmark", str(HERE / "problems_tmp" / "ragged.txt")]),
    ("beta=0 everywhere", ["--groups", "6", "--beta", "0"]),
    ("single group, big", ["--groups", "1", "--m", "1024", "--n", "4096", "--k", "4096"]),
]


def main():
    (HERE / "problems_tmp").mkdir(exist_ok=True)
    (HERE / "problems_tmp" / "ragged.txt").write_text("0 250x2048x512\n1 17x512x512\n2 1000x256x1024\n3 1x128x512\n")
    failures = 0
    for label, args in CASES:
        out = subprocess.run([sys.executable, str(HERE / "run_grouped.py"), *args, "--iterations", "1", "--warmup", "0"],
                             capture_output=True, text=True)
        text = out.stdout + out.stderr
        line = next((ln for ln in text.splitlines() if ln.startswith("correctness")), text.strip().splitlines()[-1])
        ok = "BIT-EXACT" in line
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {label:<52} {line}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
