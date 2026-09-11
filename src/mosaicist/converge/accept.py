"""The acceptance rule and the candidate beam.

    accept(c) <=> numerics_pass(c)
                  and ( t(c) < t(best) - noise
                        or ( |t(c) - t(best)| <= noise and D(c) < D(best) ) )

Runtime is the objective, numerics are a gate, and the fingerprint distance D
only breaks ties. Candidates that get closer in D but slower are rejected and
logged: they usually reveal a hidden dependency between fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..bench.stats import verdict


@dataclass
class Candidate:
    id: str
    time: float  # median device time
    distance: float  # fingerprint distance D to the reference
    numerics_pass: bool
    parent: str | None = None
    fix: str | None = None  # which fix produced it


@dataclass
class Decision:
    accepted: bool
    reason: str
    closer_but_slower: bool = False


def accept(c: Candidate, best: Candidate | None, noise: float) -> Decision:
    if not c.numerics_pass:
        return Decision(False, "numerics gate failed")
    if best is None:
        return Decision(True, "first passing candidate")
    v = verdict(c.time, best.time, noise)
    if v == "faster":
        return Decision(True, f"faster than best by more than noise ({noise:.1%})")
    if v == "equal" and c.distance < best.distance:
        return Decision(True, f"same speed within noise, closer to reference (D {c.distance:.3f} < {best.distance:.3f})")
    if v == "slower" and c.distance < best.distance:
        return Decision(False, "closer to reference but slower", closer_but_slower=True)
    return Decision(False, "slower" if v == "slower" else "same speed, not closer")


@dataclass
class Beam:
    """Keeps the k best passing candidates so a briefly-slower structural step isn't lost."""

    k: int = 4
    members: list[Candidate] = field(default_factory=list)
    rejected_closer_but_slower: list[Candidate] = field(default_factory=list)

    @property
    def best(self) -> Candidate | None:
        return self.members[0] if self.members else None

    def offer(self, c: Candidate, noise: float) -> Decision:
        d = accept(c, self.best, noise)
        if d.closer_but_slower:
            self.rejected_closer_but_slower.append(c)
        if d.accepted:
            self.members.insert(0, c)  # the acceptance rule alone decides the best
        elif c.numerics_pass:
            # runners-up stay in the beam by (time, D) so a briefly-slower step isn't lost
            rest = sorted(self.members[1:] + [c], key=lambda m: (m.time, m.distance))
            self.members = self.members[:1] + rest
        del self.members[self.k :]
        return d

    def converged(self, t_ref: float, noise: float) -> bool:
        b = self.best
        return b is not None and verdict(b.time, t_ref, noise) in ("equal", "faster")
