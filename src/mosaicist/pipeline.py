"""The whole pipeline (DESIGN §3): capture -> contract -> v0 -> converge.

Each stage already stands alone behind its own CLI command; this is the orchestration
that runs them in order and leaves every intermediate on disk, so a port can be resumed
or inspected at any stage rather than only at the end.

    workdir/
      ref/            the reference bundle: PTX, SASS, timings, outputs, oracle
      contract.json   what the translation must preserve
      v0.py           the naive Pallas kernel, importable and capturable
      run/            one bundle per convergence step, plus converge.json
      report.md       what converged, and what is left
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .bundle import Bundle
from .contract import Contract, Operand, from_capture
from .converge.loop import LoopConfig, Result, converge
from .translate import emit_v0


@dataclass
class Spec:
    """What the contract needs that a capture cannot tell you: the problem's shape."""

    dims: dict[str, int]
    operands: list[Operand]
    out_shape: tuple[int, ...]
    out_dtype: str = "bfloat16"

    @classmethod
    def load(cls, path: str | Path) -> "Spec":
        d = json.loads(Path(path).read_text())
        return cls(dims=d["dims"],
                   operands=[Operand(**o) for o in d["operands"]],
                   out_shape=tuple(d["out_shape"]),
                   out_dtype=d.get("out_dtype", "bfloat16"))


def report(result: Result, contract: Contract, ref: Bundle) -> str:
    """The report the design asks for: what converged, and what is named as a gap."""
    lines = [f"# Port of {contract.name}", "",
             f"- reference: `{ref.source}` on {contract.arch}, "
             f"{result.reference_time:.1f} us",
             f"- tile {contract.tile}, mma {contract.mma} ({contract.mma_kind})",
             f"- deferred from the reference at v0: "
             f"{', '.join(contract.deferred) or 'nothing'}", ""]
    if result.best is not None:
        ratio = result.best.time / result.reference_time if result.reference_time else float("nan")
        lines += [f"Best candidate {result.best.time:.1f} us ({ratio:.2f}x reference), "
                  f"fingerprint distance {result.best.distance:.3f}, "
                  f"setting `{result.best.id}`.", ""]
    lines += ["```", result.summary(), "```"]
    if result.remaining:
        lines += ["", "## Gaps", "",
                  "Differences with no knob left to turn. These are the candidates for",
                  "an upstream report:", ""]
        lines += [f"- {r}" for r in result.remaining]
    return "\n".join(lines) + "\n"


def port(ref_entry: str, ref_compiler: str, spec: Spec, workdir: str | Path, *,
         config: LoopConfig | None = None, ref_python: str | None = None,
         candidate_entry: str | None = None, gate: bool = True) -> Result:
    """Capture the reference, translate a v0, and converge it.

    `candidate_entry` overrides the generated v0 with a hand-written candidate, which is
    how an existing port gets tuned against a reference it was not generated from.
    """
    from .capture import capture_subprocess

    cfg = config or LoopConfig()
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)

    ref = capture_subprocess(ref_entry, ref_compiler, work / "ref",
                             python=ref_python, reps=cfg.reps)
    contract = from_capture(ref, spec.dims, spec.operands, spec.out_shape, spec.out_dtype,
                            name=ref_entry)
    contract.save(work / "contract.json")

    entry = candidate_entry
    if entry is None:
        v0 = emit_v0(contract, work / "v0.py")
        entry = v0.stem  # importable because the loop runs with workdir on sys.path

    result = converge(ref, entry, work / "run", config=cfg,
                      out_fmt=spec.out_dtype.replace("bfloat", "bf").replace("float", "f"))
    (work / "report.md").write_text(report(result, contract, ref))
    return result
