"""Command-line entry point.

    mosaicist fingerprint KERNEL.ptx|BUNDLE_DIR [--json]
    mosaicist diff REF CAND [--ref-log F] [--cand-log F] [--json]
    mosaicist check REF.npy CAND.npy ORACLE.npy --fmt bf16 [--json]

REF / CAND are PTX files or bundle directories (containing bundle.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path

from .bundle import BUNDLE_FILE, Bundle
from .diagnose import PHASES, diagnose
from .ptx.diff import diff


def _load(path: str, kind: str, log: str | None, entry: str | None) -> Bundle:
    p = Path(path)
    if p.is_dir():
        if not (p / BUNDLE_FILE).exists():
            raise SystemExit(f"error: {p} is a directory without {BUNDLE_FILE}")
        b = Bundle.load(p)
        if log:
            b.ptxas_log = str(Path(log).resolve())
        if entry:
            b.entry = entry
        return b
    if not p.exists():
        raise SystemExit(f"error: {p} does not exist")
    return Bundle.from_ptx(p, kind, ptxas_log=log, entry=entry)


def _cmd_fingerprint(args) -> int:
    fp = _load(args.kernel, "reference", args.log, args.entry).fingerprint()
    if args.json:
        print(json.dumps(fp.to_dict(), indent=2))
        return 0
    print(f"{fp.entry}  ({fp.target}, PTX {fp.ptx_version})")
    print("L0 skeleton  ", json.dumps(fp.skeleton))
    print("L1 inventory ", json.dumps(dict(sorted(fp.inventory.items()))))
    print("   mma       ", json.dumps(fp.mma_signatures))
    print("   tma       ", json.dumps(fp.tma_signatures))
    print("L2 structure ", json.dumps(fp.structure))
    for lp in fp.loops:
        print(f"L3 loop {lp.id:<3} role={lp.role:<9} depth={lp.depth} line={lp.header_line} stages={lp.stages}")
        print("   " + " ".join(lp.tokens))
    if fp.fp_flavor:
        print("L4 fp flavor ", json.dumps(dict(sorted(fp.fp_flavor.items()))))
    if fp.machine:
        print("L5 machine   ", json.dumps(fp.machine))
    return 0


def _cmd_diff(args) -> int:
    ref = _load(args.ref, "reference", args.ref_log, args.ref_entry).fingerprint()
    cand = _load(args.cand, "candidate", args.cand_log, args.cand_entry).fingerprint()
    report = diff(ref, cand)
    fixes = diagnose(report)
    if args.json:
        print(json.dumps({**report.to_dict(), "fixes": [asdict(f) for f in fixes]}, indent=2, default=str))
        return 0
    layers = "  ".join(f"{k} {v:.2f}" for k, v in report.layer_distance.items())
    print(f"reference {ref.entry}  vs  candidate {cand.entry}")
    print(f"fingerprint distance D = {report.distance:.3f}    {layers}\n")
    print(f"Discrepancies ({len(report.discrepancies)})")
    for d in report.discrepancies:
        print(f"  {d.key:<24} {d.message}")
    print(f"\nRanked fixes ({len(fixes)})")
    wrap = textwrap.TextWrapper(width=100, initial_indent=" " * 7, subsequent_indent=" " * 7)
    for i, f in enumerate(fixes, 1):
        print(f"  {i:>2}. {f.phase} {PHASES[f.phase]:<13} {f.tag:<11} {f.title}")
        print(wrap.fill(f.detail))
        print(f"       closes: {', '.join(f.keys)}")
    return 0


def _cmd_check(args) -> int:
    import numpy as np

    from .verify.numerics import GateConfig, compare_outputs

    ref, cand, oracle = (np.load(p) for p in (args.ref, args.cand, args.oracle))
    report = compare_outputs(ref, cand, oracle, args.fmt, GateConfig(delta=args.delta, tau_ulp=args.tau))
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(report.summary())
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mosaicist", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fingerprint", help="print the L0-L5 fingerprint of one kernel")
    p.add_argument("kernel", help="PTX file or bundle directory")
    p.add_argument("--log", help="ptxas -v output for L5")
    p.add_argument("--entry", help=".entry name when the module has several")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_fingerprint)

    p = sub.add_parser("diff", help="diff a candidate against the reference and rank fixes")
    p.add_argument("ref", help="reference PTX file or bundle directory")
    p.add_argument("cand", help="candidate PTX file or bundle directory")
    p.add_argument("--ref-log", help="reference ptxas -v output")
    p.add_argument("--cand-log", help="candidate ptxas -v output")
    p.add_argument("--ref-entry")
    p.add_argument("--cand-entry")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_diff)

    p = sub.add_parser("check", help="run the numerics gate on saved outputs (.npy)")
    p.add_argument("ref")
    p.add_argument("cand")
    p.add_argument("oracle", help="float64 oracle output")
    p.add_argument("--fmt", required=True, help="output format: f32, f16, bf16, e4m3, e5m2")
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--tau", type=float, default=1.0, help="absolute slack in ULPs")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_check)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
