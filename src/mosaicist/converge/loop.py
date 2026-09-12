"""The convergence loop (DESIGN §8): capture, diff, diagnose, turn one knob, repeat.

Runtime is the objective, the numerics gate decides pass/fail, and the fingerprint
distance only breaks ties -- that rule lives in `accept.py`, and this module is the
driver around it. Each step captures the candidate at one knob setting, diffs its PTX
against the reference, asks Diagnose what to change, and lets the tuner turn exactly one
knob so an accepted step can be attributed to a specific edit.

The loop stops when the candidate is within the reference's own run-to-run noise, when
the knob space is exhausted, or after `max_steps`. Whatever is left over is reported as
named gaps rather than silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ..bench.stats import noise_floor, summarize
from ..bundle import Bundle
from ..diagnose import diagnose
from ..ptx.diff import diff as diff_fingerprints
from .accept import Beam, Candidate, Decision
from .knobs import KnobSpace, Setting, key, to_env
from .rewrite import Rewriter
from .tune import Tuner


@dataclass
class LoopConfig:
    max_steps: int = 12
    reps: int = 30
    python: str | None = None
    #: fallback noise floor when the reference has too few samples to measure one
    default_noise: float = 0.05
    #: how many batches to split the reference's samples into when measuring it
    noise_batches: int = 4
    gate_numerics: bool = True


@dataclass
class Step:
    index: int
    setting: Setting
    time: float
    distance: float
    numerics_pass: bool
    accepted: bool
    reason: str
    fix: str | None = None
    top_rows: list[str] = field(default_factory=list)


@dataclass
class Result:
    steps: list[Step]
    best: Candidate | None
    reference_time: float
    noise: float
    converged: bool
    remaining: list[str] = field(default_factory=list)
    #: structural proposals the loop took, in order
    rewrites: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"reference {self.reference_time:.1f} us, noise floor {self.noise:.1%}", ""]
        for s in self.steps:
            mark = "accept" if s.accepted else "  --  "
            if s.time == float("inf"):
                lines.append(f"  {s.index:>2}  ----   {s.reason}  "
                             f"{json.dumps(s.setting, sort_keys=True)}")
                continue
            lines.append(f"  {s.index:>2} {mark}  {s.time:8.1f} us  D={s.distance:.3f}  "
                         f"{'numerics ok' if s.numerics_pass else 'NUMERICS FAIL'}  "
                         f"{json.dumps(s.setting, sort_keys=True)}")
            if s.fix:
                lines.append(f"          next: {s.fix}")
        if self.best is not None:
            ratio = self.best.time / self.reference_time if self.reference_time else float("nan")
            lines += ["", f"best {self.best.time:.1f} us ({ratio:.2f}x reference), "
                          f"D={self.best.distance:.3f}, setting {self.best.id}"]
        if self.rewrites:
            lines += ["", "structural steps taken:"] + [f"  - {r}" for r in self.rewrites]
        lines.append("converged" if self.converged
                     else "did not reach the reference within its noise floor")
        if self.remaining:
            lines += ["", "remaining differences with no knob to turn:"] + \
                     [f"  - {r}" for r in self.remaining]
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps({"steps": [asdict(s) for s in self.steps],
                                 "best": asdict(self.best) if self.best else None,
                                 "reference_time": self.reference_time, "noise": self.noise,
                                 "converged": self.converged, "remaining": self.remaining,
                                 "rewrites": self.rewrites},
                                indent=2))
        return p


def _reference_noise(samples: list[float], cfg: LoopConfig) -> float:
    """The reference's own run-to-run spread, which is what "equal" has to mean.

    The design measures this with repeated A/A runs; a single capture's samples split
    into batches is the same statistic from data already in hand, and it beats asserting
    a constant. Too few samples to split and we fall back to the configured default.
    """
    n = len(samples or [])
    if n < 2 * cfg.noise_batches:
        return cfg.default_noise
    size = n // cfg.noise_batches
    batches = [samples[i * size:(i + 1) * size] for i in range(cfg.noise_batches)]
    return max(noise_floor(batches), cfg.default_noise * 0.2)


#: A runner captures the candidate at one setting and returns its bundle.
Runner = Callable[[Setting, Path], Bundle]


def subprocess_runner(entry: str, config: LoopConfig) -> Runner:
    from ..capture import capture_subprocess

    def run(setting: Setting, outdir: Path) -> Bundle:
        return capture_subprocess(entry, "pallas", outdir, python=config.python,
                                  reps=config.reps, extra_env=to_env(setting))

    return run


def _numerics_ok(ref: Bundle, cand: Bundle, fmt: str) -> bool:
    """Gate the candidate against the reference using the saved outputs and oracle.

    Returns True when there is nothing to compare: a missing oracle means the gate was
    not configured, and silently failing every candidate would be worse than skipping.
    """
    import numpy as np

    from ..verify.numerics import GateConfig, compare_outputs

    r, c = ref.path("out0.npy"), cand.path("out0.npy")
    o = ref.path("oracle0.npy")
    if not (r and c and o and r.exists() and c.exists() and o.exists()):
        return True
    report = compare_outputs(np.load(r), np.load(c), np.load(o), fmt, GateConfig())
    return report.passed


def converge(reference: Bundle, candidate_entry: str, outdir: str | Path,
             knobs_module: str | None = None, config: LoopConfig | None = None,
             runner: Runner | None = None, space: KnobSpace | None = None,
             out_fmt: str = "bf16", rewriter: Rewriter | None = None) -> Result:
    """Turn one knob at a time until the candidate matches the reference's runtime."""
    import importlib

    cfg = config or LoopConfig()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    module = None
    if space is None or rewriter is None:
        mod_name = knobs_module or candidate_entry.partition(":")[0]
        try:
            module = importlib.import_module(mod_name)
        except ImportError:
            module = None
    if space is None:
        space = KnobSpace.from_module(module) if module else KnobSpace()
    if rewriter is None and module is not None:
        from .rewrite import VariantRewriter

        rewriter = VariantRewriter.from_module(module, space)
    run = runner or subprocess_runner(candidate_entry, cfg)

    ref_fp = reference.fingerprint()
    ref_times = summarize(reference.timings) if reference.timings else None
    t_ref = ref_times.median if ref_times else float("nan")
    noise = _reference_noise(reference.timings, cfg)

    structural = tuple(getattr(module, "STRUCTURAL", ()) or ()) if module else ()
    if rewriter is not None and hasattr(rewriter, "structural"):
        structural = tuple(rewriter.structural)
    beam, tuner = Beam(), Tuner(space, structural=structural)
    by_id: dict[str, Setting] = {}
    rewrites: list[str] = []
    setting: Setting | None = space.default()
    steps: list[Step] = []
    last_fixes: list = []

    for i in range(cfg.max_steps):
        if setting is None:
            break
        tuner.mark(setting)
        try:
            bundle = run(setting, out / f"step{i:02d}")
        except Exception as exc:  # noqa: BLE001
            # a knob space always contains combinations that cannot be built -- too much
            # shared memory, an illegal tile. That is a result about the space, not a
            # reason to abandon the search.
            steps.append(Step(index=i, setting=dict(setting), time=float("inf"),
                              distance=float("nan"), numerics_pass=False, accepted=False,
                              reason=f"capture failed: {' '.join(str(exc).split())[:120]}"))
            base = by_id.get(beam.best.id, setting) if beam.best else setting
            setting = tuner.propose(base, last_fixes)
            if setting is None and rewriter is not None:
                proposal = rewriter.propose(base, last_fixes, tuner.tried)
                if proposal is not None:
                    setting, _ = proposal.setting, rewrites.append(proposal.rationale)
            if setting is None:
                break
            continue
        fp = bundle.fingerprint()
        report = diff_fingerprints(ref_fp, fp)
        fixes = diagnose(report)
        last_fixes = fixes
        times = summarize(bundle.timings) if bundle.timings else None
        t = times.median if times else float("inf")
        ok = _numerics_ok(reference, bundle, out_fmt) if cfg.gate_numerics else True

        by_id[key(space.clamp(setting))] = dict(setting)
        cand = Candidate(id=key(space.clamp(setting)), time=t, distance=report.distance,
                         numerics_pass=ok, fix=str(fixes[0]) if fixes else None)
        decision: Decision = beam.offer(cand, noise)
        steps.append(Step(index=i, setting=dict(setting), time=t, distance=report.distance,
                          numerics_pass=ok, accepted=decision.accepted, reason=decision.reason,
                          fix=str(fixes[0]) if fixes else None,
                          top_rows=[d.key for d in report.discrepancies[:4]]))

        if beam.converged(t_ref, noise):
            break
        # explore from the best candidate so far, not from the last one tried: a greedy
        # search that walks away from its best result stops being greedy
        base = by_id.get(beam.best.id, setting) if beam.best else setting
        setting = tuner.propose(base, fixes)
        if setting is None and rewriter is not None:
            # knobs around the best are spent: ask for the structural step Diagnose has
            # been asking for. The beam keeps it even if it lands briefly slower.
            proposal = rewriter.propose(base, fixes, tuner.tried)
            if proposal is not None:
                setting = proposal.setting
                rewrites.append(proposal.rationale)
        if setting is None:
            break

    remaining = [f"{f.phase} {f.tag}: {f.title}" for f in last_fixes
                 if f.tag in ("rewrite", "gap", "investigate")]
    result = Result(steps=steps, best=beam.best, reference_time=t_ref, noise=noise,
                    converged=beam.converged(t_ref, noise), remaining=remaining,
                    rewrites=rewrites)
    result.save(out / "converge.json")
    return result
