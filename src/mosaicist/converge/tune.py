"""The tuner: which knob to turn next, given the diff.

Diagnose ranks the fingerprint discrepancies and names a fix for each; the catalog says
which knobs each construct exposes. The tuner walks that ranking and proposes the
nearest untried setting of a knob the top-ranked fix actually implicates, so the search
follows the PTX diff rather than sweeping blindly. When the ranking is exhausted -- or
every knob it names has been tried -- it falls back to unexplored neighbours, in
declaration order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .. import catalog
from .knobs import KnobSpace, Setting, key


def knobs_for_fixes(fixes: Iterable) -> list[str]:
    """Knob names implicated by ranked fixes, best-ranked first, without duplicates.

    A fix carries the diff row keys it closes; each maps through the catalog to the
    constructs that produce it, and each construct declares its knobs.
    """
    out: list[str] = []
    for fix in fixes:
        for row in getattr(fix, "keys", ()) or ():
            for construct in catalog.for_row(row):
                for k in construct.knobs:
                    if k not in out:
                        out.append(k)
    return out


@dataclass
class Tuner:
    space: KnobSpace
    tried: set[str] = field(default_factory=set)
    #: knobs that swap in a different kernel rather than retune this one. The design
    #: orders the phases knobs-before-rewrites, so the tuner leaves these to the
    #: rewriter instead of reaching them incidentally.
    structural: tuple[str, ...] = ()

    def mark(self, setting: Setting) -> None:
        self.tried.add(key(self.space.clamp(setting)))

    def propose(self, current: Setting, fixes: Iterable = ()) -> Setting | None:
        """The next setting to try, or None when the space is exhausted."""
        current = self.space.clamp(current)
        tunable = [k for k in self.space.knobs if k not in self.structural]
        named = [k for k in knobs_for_fixes(fixes) if k in tunable]

        # never fall back to `only=None`: that means "no restriction" and would let
        # structural knobs back in through the fallback, which is the rewriter's job
        for only in ([named, tunable] if named else [tunable]):
            for candidate in self.space.neighbours(current, only=only):
                if key(self.space.clamp(candidate)) not in self.tried:
                    return candidate
        return None
