"""Structural rewrites (DESIGN §8, M4): the step a knob cannot reach.

Knob fixes are cheap and deterministic, so the tuner tries them first. What is left --
warp specialization, persistent scheduling, a different epilogue -- changes the shape of
the kernel, and Diagnose tags those fixes ``rewrite`` rather than ``knob``. This module
is where such a proposal comes from, and the loop only asks for one once the knobs
around the current best are spent.

Two backends:

  ``VariantRewriter``  the candidate module declares named structural variants it
                       already implements (``VARIANTS``), and the rewriter picks the one
                       whose catalog construct matches the fix. Deterministic, and the
                       only backend that runs here.
  ``LLMRewriter``      the seam from the design: hand a model the source, the fix, and
                       the diff rows, get source back. It takes the model call as an
                       injected callable, so nothing in this package depends on having
                       one.

A structural step is expected to be *briefly* slower than the best knob setting, which
is why `Beam` keeps runners-up: rejecting it immediately would hide the setting it
unlocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

from .. import catalog
from .knobs import KnobSpace, Setting, key


@dataclass
class Proposal:
    """A structural change to try, expressed as a setting plus why."""

    setting: Setting
    rationale: str
    source: str | None = None  # set by a rewriter that emits code


class Rewriter(Protocol):
    def propose(self, current: Setting, fixes: Iterable, tried: set[str]) -> Proposal | None: ...


def structural_knobs(fixes: Iterable) -> list[str]:
    """Knobs named by fixes Diagnose tagged as rewrites, best-ranked first."""
    out: list[str] = []
    for fix in fixes:
        if getattr(fix, "tag", "") != "rewrite":
            continue
        for row in getattr(fix, "keys", ()) or ():
            for construct in catalog.for_row(row):
                for k in construct.knobs:
                    if k not in out:
                        out.append(k)
    return out


@dataclass
class VariantRewriter:
    """Turns a structural variant the candidate already implements.

    `space` holds every knob; `structural` names the subset that changes the shape of
    the kernel rather than a parameter of it. The candidate declares that subset, since
    only it knows which of its knobs swap in a different kernel.
    """

    space: KnobSpace
    structural: tuple[str, ...] = ()

    @classmethod
    def from_module(cls, module, space: KnobSpace | None = None) -> "VariantRewriter":
        sp = space or KnobSpace.from_module(module)
        return cls(space=sp, structural=tuple(getattr(module, "STRUCTURAL", ()) or ()))

    def propose(self, current: Setting, fixes: Iterable, tried: set[str]) -> Proposal | None:
        named = [k for k in structural_knobs(fixes) if k in self.structural]
        order = named + [k for k in self.structural if k not in named]
        for knob in order:
            for cand in self.space.neighbours(current, only=[knob]):
                if key(self.space.clamp(cand)) not in tried:
                    why = (f"structural: {knob} -> {cand[knob]}"
                           + (f", named by {named[0]}" if named and knob == named[0] else ""))
                    return Proposal(setting=cand, rationale=why)
        return None


@dataclass
class LLMRewriter:
    """The design's rewriter, with the model call injected.

    `emit(source, fix, rows) -> source` is whatever calls a model. This class only owns
    the contract around it: one fix per edit, the diff rows as evidence, and the source
    that came back recorded on the proposal so an accepted step stays attributable.
    """

    emit: Callable[[str, object, list[str]], str]
    source: str
    setting: Setting = field(default_factory=dict)

    def propose(self, current: Setting, fixes: Iterable, tried: set[str]) -> Proposal | None:
        ranked = [f for f in fixes if getattr(f, "tag", "") == "rewrite"]
        if not ranked:
            return None
        fix = ranked[0]
        rows = list(getattr(fix, "keys", ()) or ())
        new_source = self.emit(self.source, fix, rows)
        if not new_source or new_source == self.source:
            return None
        return Proposal(setting=dict(current), rationale=f"rewrite: {fix}", source=new_source)
