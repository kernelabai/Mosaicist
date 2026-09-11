"""Floating-point formats and value-based ULP arithmetic.

Everything works on float64 arrays holding values that are exactly
representable in the target format (true for f32, f16, bf16, and fp8
outputs upcast to float64), so bf16 and fp8 need no special dtypes: a value's
position in the format is computed from its exponent and mantissa directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FloatFormat:
    name: str
    mant: int  # explicit mantissa bits
    emin: int  # exponent of the smallest normal number
    max_finite: float
    has_inf: bool = True

    @property
    def min_normal(self) -> float:
        return 2.0 ** self.emin

    @property
    def min_subnormal(self) -> float:
        return 2.0 ** (self.emin - self.mant)


FORMATS: dict[str, FloatFormat] = {
    "float64": FloatFormat("float64", 52, -1022, float(np.finfo(np.float64).max)),
    "float32": FloatFormat("float32", 23, -126, float(np.finfo(np.float32).max)),
    "float16": FloatFormat("float16", 10, -14, 65504.0),
    "bfloat16": FloatFormat("bfloat16", 7, -126, float.fromhex("0x1.fep127")),
    "float8_e4m3fn": FloatFormat("float8_e4m3fn", 3, -6, 448.0, has_inf=False),
    "float8_e5m2": FloatFormat("float8_e5m2", 2, -14, 57344.0),
}
ALIASES = {"f64": "float64", "f32": "float32", "f16": "float16", "bf16": "bfloat16",
           "e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2", "half": "float16"}


def get_format(fmt: str | FloatFormat) -> FloatFormat:
    if isinstance(fmt, FloatFormat):
        return fmt
    name = ALIASES.get(fmt, fmt)
    if name not in FORMATS:
        raise KeyError(f"unknown float format {fmt!r}; known: {sorted(FORMATS)}")
    return FORMATS[name]


def _exponent(ax: np.ndarray, f: FloatFormat) -> np.ndarray:
    """floor(log2 |x|) clamped to emin (subnormals share emin's spacing)."""
    _, e = np.frexp(ax)  # ax = m * 2**e, m in [0.5, 1); frexp(0) gives e = 0
    e = np.maximum(e.astype(np.int64) - 1, f.emin)
    return np.where(ax == 0, f.emin, e)


def ulp(x: np.ndarray, fmt: str | FloatFormat) -> np.ndarray:
    """Spacing of the format at |x| (the ULP of the binade containing x)."""
    f = get_format(fmt)
    ax = np.abs(np.asarray(x, dtype=np.float64))
    return np.ldexp(1.0, _exponent(ax, f) - f.mant)


def round_to_format(x: np.ndarray, fmt: str | FloatFormat) -> np.ndarray:
    """Round-to-nearest-even into the format, returned as float64.

    Overflow goes to +-inf (or saturates to max_finite for formats without
    inf, matching the `satfinite` conversions fp8 kernels use).
    """
    f = get_format(fmt)
    x = np.asarray(x, dtype=np.float64)
    if f.name == "float64":
        return x.copy()
    u = ulp(x, f)
    with np.errstate(invalid="ignore", over="ignore"):
        r = np.rint(x / u) * u  # np.rint rounds half to even
    finite = np.isfinite(x)
    over = finite & (np.abs(r) > f.max_finite)
    # values that round above max_finite become inf, or saturate when the format has none
    r = np.where(over, np.copysign(np.inf if f.has_inf else f.max_finite, x), r)
    return np.where(finite, r, x)


def ordinal(x: np.ndarray, fmt: str | FloatFormat) -> np.ndarray:
    """Signed position of each value in the format's ordered value set.

    Adjacent representable values differ by 1, +0 and -0 both map to 0, so
    |ordinal(a) - ordinal(b)| is the ULP distance between a and b. Values must
    be finite and representable in the format.
    """
    f = get_format(fmt)
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    e = _exponent(ax, f)
    u = np.ldexp(1.0, e - f.mant)
    normal = ax >= f.min_normal
    # steps within the binade (subnormals: steps above zero)
    within = np.rint((ax - np.where(normal, np.ldexp(1.0, e), 0.0)) / u).astype(np.int64)
    # normals: skip the 2**mant subnormal values plus every full binade below this one
    idx = np.where(normal, (e - f.emin + 1) * (1 << f.mant) + within, within)
    return np.where(np.signbit(x) & (ax > 0), -idx, idx)


def ulp_distance(a: np.ndarray, b: np.ndarray, fmt: str | FloatFormat) -> np.ndarray:
    """ULP distance between two arrays of representable values (finite only)."""
    return np.abs(ordinal(a, fmt) - ordinal(b, fmt))
