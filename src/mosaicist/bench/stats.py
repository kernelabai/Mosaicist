"""Timing statistics: robust summaries, bootstrap intervals, and the noise floor.

Samples are device times (CUPTI activity records or equivalent) in one unit,
collected under locked clocks with an L2 flush between repetitions. Nothing
here decides "faster" with a fixed percentage: the noise floor measured by
running the reference against itself sets that threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_NOISE = 0.005  # never claim resolution finer than 0.5%


@dataclass
class Timing:
    median: float
    p10: float
    p90: float
    ci_low: float  # bootstrap CI of the median
    ci_high: float
    n: int

    @property
    def rel_halfwidth(self) -> float:
        return (self.ci_high - self.ci_low) / (2 * self.median) if self.median else float("inf")


def summarize(samples, confidence: float = 0.95, resamples: int = 2000, seed: int = 0) -> Timing:
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no timing samples")
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(x, size=(resamples, x.size), replace=True), axis=1)
    lo, hi = np.quantile(boots, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return Timing(float(np.median(x)), float(np.quantile(x, 0.1)), float(np.quantile(x, 0.9)),
                  float(lo), float(hi), int(x.size))


def noise_floor(batches, floor: float = MIN_NOISE) -> float:
    """Relative noise from repeated batches of the *same* kernel (an A/A run).

    The largest relative deviation of any batch median from the pooled median,
    widened by the typical bootstrap half-width, and never below `floor`.
    """
    batches = [np.asarray(b, dtype=np.float64) for b in batches]
    if len(batches) < 2:
        raise ValueError("noise_floor needs at least two batches of the same kernel")
    pooled = float(np.median(np.concatenate(batches)))
    meds = np.array([np.median(b) for b in batches])
    spread = float(np.max(np.abs(meds - pooled)) / pooled)
    halfwidth = float(np.median([summarize(b).rel_halfwidth for b in batches]))
    return max(floor, spread + halfwidth)


def verdict(t_cand: float, t_ref: float, noise: float) -> str:
    """'faster', 'slower', or 'equal' within the relative noise floor."""
    rel = (t_cand - t_ref) / t_ref
    if rel < -noise:
        return "faster"
    if rel > noise:
        return "slower"
    return "equal"
