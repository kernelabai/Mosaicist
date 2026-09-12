"""The numerics gate: is the candidate as accurate as the reference?

With e = |y - y64| in ULPs of the output format, a candidate passes when for
each quantile q:

    Q_q(e_cand) <= (1 + delta) * Q_q(e_ref) + tau

and its NaN/Inf pattern matches the reference's exactly. This ties accuracy
to what the reference actually achieves rather than to a fixed tolerance.
The fraction of bit-identical outputs is reported as a convergence signal,
not gated on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .formats import FloatFormat, get_format, ulp, ulp_distance

_HIST_EDGES = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 8, "5-8"), (9, 16, "9-16"),
               (17, None, ">16")]


@dataclass(frozen=True)
class GateConfig:
    quantiles: tuple[float, ...] = (0.5, 0.99, 1.0)
    delta: float = 0.10
    tau_ulp: float = 1.0


@dataclass
class QuantileRow:
    q: float
    ref_err: float
    cand_err: float
    bound: float
    ok: bool


@dataclass
class NumericsReport:
    passed: bool
    fmt: str
    n: int
    quantiles: list[QuantileRow]
    bitwise_equal_fraction: float
    ulp_hist: dict[str, int]  # candidate vs reference, finite positions
    max_ulp_ref_vs_cand: int
    special_mismatch: int
    failures: list[str] = field(default_factory=list)
    worst_index: tuple[int, ...] | None = None

    def summary(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        q = ", ".join(f"q{r.q:g}: cand {r.cand_err:.3g} vs ref {r.ref_err:.3g} ulp" for r in self.quantiles)
        # not .1%: 99.996% rounds to "100.0%", which reads as bit-identical right next
        # to a non-zero max|ref-cand|
        frac = self.bitwise_equal_fraction
        shown = "100%" if frac == 1.0 else f"{min(frac, 0.99999):.3%}"
        return (f"{head} [{self.fmt}] n={self.n} bitwise={shown} "
                f"max|ref-cand|={self.max_ulp_ref_vs_cand} ulp; {q}"
                + ("" if self.passed else " :: " + "; ".join(self.failures)))


def _special_mismatch(ref: np.ndarray, cand: np.ndarray) -> np.ndarray:
    nan_r, nan_c = np.isnan(ref), np.isnan(cand)
    inf_r = np.where(np.isinf(ref), np.sign(ref), 0)
    inf_c = np.where(np.isinf(cand), np.sign(cand), 0)
    return (nan_r != nan_c) | (inf_r != inf_c)


def bitwise_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise bit equality for values upcast to float64 (NaN == NaN, -0 != +0)."""
    both_nan = np.isnan(a) & np.isnan(b)
    same = (a == b) & (np.signbit(a) == np.signbit(b))
    return both_nan | same


def compare_outputs(
    ref: np.ndarray,
    cand: np.ndarray,
    oracle: np.ndarray,
    fmt: str | FloatFormat,
    config: GateConfig = GateConfig(),
) -> NumericsReport:
    """Gate one output tensor. Arrays may be any float dtype; they are upcast to float64."""
    f = get_format(fmt)
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    failures: list[str] = []
    if not (ref.shape == cand.shape == oracle.shape):
        return NumericsReport(False, f.name, int(cand.size), [], 0.0, {}, 0, 0,
                              [f"shape mismatch: ref {ref.shape}, cand {cand.shape}, oracle {oracle.shape}"])

    special = _special_mismatch(ref, cand)
    n_special = int(special.sum())
    if n_special:
        first = tuple(int(i) for i in np.argwhere(special)[0])
        failures.append(f"NaN/Inf pattern differs at {n_special} positions (first at {first})")

    eq = bitwise_equal(ref, cand)
    finite = np.isfinite(ref) & np.isfinite(cand) & np.isfinite(oracle)
    u = ulp(oracle, f)
    with np.errstate(invalid="ignore"):
        e_ref = np.abs(ref - oracle) / u
        e_cand = np.abs(cand - oracle) / u

    rows: list[QuantileRow] = []
    worst = None
    if finite.any():
        er, ec = e_ref[finite], e_cand[finite]
        for q in config.quantiles:
            qr, qc = float(np.quantile(er, q)), float(np.quantile(ec, q))
            bound = (1 + config.delta) * qr + config.tau_ulp
            ok = qc <= bound
            rows.append(QuantileRow(q, qr, qc, bound, ok))
            if not ok:
                failures.append(f"q{q:g} error {qc:.3g} ulp exceeds bound {bound:.3g} (reference {qr:.3g})")
        masked = np.where(finite, e_cand - e_ref, -np.inf)
        worst = tuple(int(i) for i in np.unravel_index(int(np.argmax(masked)), masked.shape))
        d = ulp_distance(ref[finite], cand[finite], f)
        hist = {label: int(((d >= lo) & (d <= (hi if hi is not None else np.iinfo(np.int64).max))).sum())
                for lo, hi, label in _HIST_EDGES}
        max_ulp = int(d.max())
    else:
        hist, max_ulp = {}, 0

    return NumericsReport(
        passed=not failures, fmt=f.name, n=int(cand.size), quantiles=rows,
        bitwise_equal_fraction=float(eq.mean()) if eq.size else 1.0,
        ulp_hist=hist, max_ulp_ref_vs_cand=max_ulp, special_mismatch=n_special,
        failures=failures, worst_index=worst,
    )


def tile_error_summary(bad: np.ndarray, tile: tuple[int, int]) -> dict:
    """Where the failing elements of a 2-D output sit, in tile coordinates.

    Feeds the translator's repair loop: errors confined to the last tile row
    or column point at boundary masking; errors everywhere point at math.
    """
    bad = np.asarray(bad, dtype=bool)
    if bad.ndim != 2:
        raise ValueError("tile_error_summary expects a 2-D mask")
    tm, tn = tile
    rows, cols = -(-bad.shape[0] // tm), -(-bad.shape[1] // tn)
    grid = np.zeros((rows, cols), dtype=bool)
    for i, j in np.argwhere(bad):
        grid[i // tm, j // tn] = True
    hit = np.argwhere(grid)
    return {
        "tiles": (rows, cols),
        "bad_tiles": int(grid.sum()),
        "bad_fraction": float(grid.mean()) if grid.size else 0.0,
        "only_last_tile_row": bool(hit.size) and bool((hit[:, 0] == rows - 1).all()),
        "only_last_tile_col": bool(hit.size) and bool((hit[:, 1] == cols - 1).all()),
        "boundary_only": bool(hit.size) and bool(((hit[:, 0] == rows - 1) | (hit[:, 1] == cols - 1)).all()),
    }
