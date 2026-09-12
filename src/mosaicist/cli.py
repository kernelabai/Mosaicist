"""Command-line entry point.

    mosaicist fingerprint KERNEL.ptx|BUNDLE_DIR [--json]
    mosaicist diff REF CAND [--ref-log F] [--cand-log F] [--json]
    mosaicist check REF.npy CAND.npy ORACLE.npy --fmt bf16 [--json]
    mosaicist time REF_BUNDLE CAND_BUNDLE... [--aa REF_BUNDLE2 ...]

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


def _cmd_time(args) -> int:
    from .bench.stats import MIN_NOISE, noise_floor, summarize, verdict

    ref = Bundle.load(args.ref)
    batches = [ref.timings] + [Bundle.load(p).timings for p in args.aa]
    noise = noise_floor(batches) if len(batches) > 1 else MIN_NOISE
    t_ref = summarize(ref.timings)
    src = f"A/A over {len(batches)} reference runs" if len(batches) > 1 else "default floor; pass --aa for a measured one"
    print(f"noise floor {noise:.1%} ({src})")
    print(f"  {'reference':<28} {t_ref.median:9.1f} us  [{t_ref.ci_low:.1f}, {t_ref.ci_high:.1f}]  n={t_ref.n}")
    for p in args.cands:
        b = Bundle.load(p)
        t = summarize(b.timings)
        print(f"  {Path(p).name:<28} {t.median:9.1f} us  [{t.ci_low:.1f}, {t.ci_high:.1f}]  n={t.n}  "
              f"{t.median / t_ref.median:5.3f}x  {verdict(t.median, t_ref.median, noise)}")
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


def _cmd_capture(args) -> int:
    from .capture import capture, capture_subprocess

    if args.in_process:
        b = capture(args.entry, args.compiler, args.out, reps=args.reps, arch=args.arch,
                    prefer=args.prefer, name=args.name)
    else:
        b = capture_subprocess(args.entry, args.compiler, args.out,
                               python=args.python, reps=args.reps)
    times = sorted(b.timings)
    med = times[len(times) // 2] if times else float("nan")
    print(f"{b.kind} bundle {args.out}: ptx={b.ptx} sass={b.sass} "
          f"median {med:.1f} us over {len(times)} reps")
    return 0


def _cmd_translate(args) -> int:
    from .contract import Contract
    from .translate import emit_v0

    contract = Contract.load(args.contract)
    path = emit_v0(contract, args.out)
    print(f"wrote {path} for {contract.name} ({contract.arch}, {contract.mma})")
    if contract.deferred:
        print("deferred from the reference: " + ", ".join(contract.deferred))
    return 0


def _cmd_converge(args) -> int:
    from .bundle import Bundle
    from .converge.loop import LoopConfig, converge

    result = converge(
        reference=Bundle.load(args.ref),
        candidate_entry=args.entry,
        knobs_module=args.knobs,
        outdir=args.out,
        config=LoopConfig(max_steps=args.steps, reps=args.reps, python=args.python,
                          gate_numerics=not args.no_gate),
    )
    print(result.summary())
    return 0 if result.converged else 1


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

    p = sub.add_parser("time", help="compare device times across bundles against a measured noise floor")
    p.add_argument("ref", help="reference bundle directory")
    p.add_argument("cands", nargs="+", help="candidate bundle directories")
    p.add_argument("--aa", nargs="*", default=[], help="extra reference runs (A/A) for the noise floor")
    p.set_defaults(func=_cmd_time)

    p = sub.add_parser("check", help="run the numerics gate on saved outputs (.npy)")
    p.add_argument("ref")
    p.add_argument("cand")
    p.add_argument("oracle", help="float64 oracle output")
    p.add_argument("--fmt", required=True, help="output format: f32, f16, bf16, e4m3, e5m2")
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--tau", type=float, default=1.0, help="absolute slack in ULPs")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("capture", help="run one kernel with dumping on and write a bundle")
    p.add_argument("entry", help="module:function returning (callable, args); default fn 'build'")
    p.add_argument("--compiler", choices=("pallas", "cutedsl"), required=True)
    p.add_argument("--out", required=True, help="bundle directory to write")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--arch", help="sm_90a / sm_100a, recorded in the bundle")
    p.add_argument("--prefer", help="substring picking the kernel when several are dumped")
    p.add_argument("--name")
    p.add_argument("--python", help="interpreter for the capture subprocess")
    p.add_argument("--in-process", action="store_true",
                   help="assume the dump environment is already set (used by the driver)")
    p.set_defaults(func=_cmd_capture)

    p = sub.add_parser("translate", help="emit a naive v0 Pallas kernel from a contract")
    p.add_argument("contract", help="contract JSON")
    p.add_argument("--out", required=True, help="path for the generated module")
    p.set_defaults(func=_cmd_translate)

    p = sub.add_parser("converge", help="tune a candidate toward a reference bundle")
    p.add_argument("ref", help="reference bundle directory")
    p.add_argument("entry", help="candidate module:function exposing build()/knobs")
    p.add_argument("--knobs", help="module declaring the knob space (default: the entry module)")
    p.add_argument("--out", required=True, help="directory for per-step bundles")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--python", help="interpreter for capture subprocesses")
    p.add_argument("--no-gate", action="store_true",
                   help="skip the numerics gate: use when the reference and candidate "
                        "cannot be fed identical inputs (different frameworks), and "
                        "check numerics separately")
    p.set_defaults(func=_cmd_converge)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
