"""Needleman-Wunsch global alignment of critical-op token sequences (L3)."""

from __future__ import annotations

from dataclasses import dataclass

from .ops import token_kind

MATCH = 2.0
SUBST = 0.5  # same kind, different detail (e.g. wgmma.wait:1 vs wgmma.wait:0)
MISMATCH = -1.0
GAP = -1.0


@dataclass
class Step:
    op: str  # "match" | "subst" | "del" (ref only) | "ins" (cand only) | "mismatch"
    ref: int | None
    cand: int | None


@dataclass
class Alignment:
    steps: list[Step]
    score: float

    @property
    def similarity(self) -> float:
        """Fraction of positions that match exactly, over the longer sequence."""
        n = sum(1 for s in self.steps if s.ref is not None)
        m = sum(1 for s in self.steps if s.cand is not None)
        if max(n, m) == 0:
            return 1.0
        return sum(1 for s in self.steps if s.op == "match") / max(n, m)

    def differences(self) -> list[Step]:
        return [s for s in self.steps if s.op != "match"]


def _score(a: str, b: str) -> float:
    if a == b:
        return MATCH
    if token_kind(a) == token_kind(b):
        return SUBST
    return MISMATCH


def align(ref: list[str], cand: list[str]) -> Alignment:
    n, m = len(ref), len(cand)
    # dp[i][j]: best score aligning ref[:i] with cand[:j]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * GAP
    for j in range(1, m + 1):
        dp[0][j] = j * GAP
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = max(prev[j - 1] + _score(ri, cand[j - 1]), prev[j] + GAP, row[j - 1] + GAP)

    steps: list[Step] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + _score(ref[i - 1], cand[j - 1]):
            s = _score(ref[i - 1], cand[j - 1])
            op = "match" if s == MATCH else "subst" if s == SUBST else "mismatch"
            steps.append(Step(op, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + GAP:
            steps.append(Step("del", i - 1, None))
            i -= 1
        else:
            steps.append(Step("ins", None, j - 1))
            j -= 1
    steps.reverse()
    # A mismatch is reported as a deletion plus an insertion: it is clearer to read.
    out: list[Step] = []
    for s in steps:
        if s.op == "mismatch":
            out.append(Step("del", s.ref, None))
            out.append(Step("ins", None, s.cand))
        else:
            out.append(s)
    return Alignment(out, dp[n][m])
