"""Deterministic input suites: shapes x distributions x seeds.

Inputs are generated in float64 and rounded into the kernel's input format,
so the same suite can be fed to CuTeDSL (via torch/DLPack) and to JAX and
stored alongside the reference bundle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .formats import FloatFormat, get_format, round_to_format

DISTRIBUTIONS = ("normal", "uniform", "wide", "cancel", "special")


@dataclass(frozen=True)
class Case:
    name: str
    dims: tuple[tuple[str, int], ...]  # e.g. (("M", 8192), ("N", 8192), ("K", 8192))
    dist: str
    seed: int

    def dim(self, name: str) -> int:
        return dict(self.dims)[name]


def shape_variants(dims: dict[str, int], tile: dict[str, int]) -> dict[str, dict[str, int]]:
    """Tile-aligned, ragged-per-dimension, and minimal problem sizes."""
    out = {"aligned": dict(dims)}
    for d, t in tile.items():
        ragged = dict(dims)
        ragged[d] = max(1, dims[d] - t // 2 - 1)  # leaves a partial tile in dimension d
        out[f"ragged_{d}"] = ragged
    out["minimal"] = {d: tile.get(d, v) for d, v in dims.items()}
    return out


def make_suite(dims: dict[str, int], tile: dict[str, int], dists=DISTRIBUTIONS, seeds=(0, 1, 2, 3, 4)) -> list[Case]:
    cases = []
    for shape_name, shape in shape_variants(dims, tile).items():
        for dist in dists:
            for seed in seeds:
                cases.append(Case(f"{shape_name}/{dist}/s{seed}", tuple(shape.items()), dist, seed))
    return cases


def generate(shape: tuple[int, ...], dist: str, seed: int, fmt: str | FloatFormat, arg_index: int = 0) -> np.ndarray:
    """One input array (float64, exactly representable in `fmt`)."""
    f = get_format(fmt)
    rng = np.random.default_rng([seed, arg_index])
    if dist == "normal":
        x = rng.standard_normal(shape)
    elif dist == "uniform":
        x = rng.uniform(-1.0, 1.0, shape)
    elif dist == "wide":
        emax = int(np.log2(f.max_finite))
        k = max(2, min(20, emax // 2))
        x = rng.choice([-1.0, 1.0], shape) * np.exp2(rng.uniform(-k, k, shape))
    elif dist == "cancel":
        # large alternating-sign values along the last axis: reductions cancel heavily
        big = np.abs(rng.standard_normal(shape)) * 64.0
        sign = np.where(np.arange(shape[-1]) % 2 == 0, 1.0, -1.0)
        x = big * sign + rng.standard_normal(shape) * 1e-2
    elif dist == "special":
        x = rng.standard_normal(shape)
        flat = x.reshape(-1)
        n = flat.size
        picks = rng.choice(n, size=min(n, max(5, n // 1000)), replace=False)
        specials = [np.nan, np.inf, -np.inf, -0.0, f.min_subnormal]
        if not f.has_inf:
            specials = [np.nan, -0.0, f.min_subnormal, f.max_finite, -f.max_finite]
        for i, p in enumerate(picks):
            flat[p] = specials[i % len(specials)]
        x = flat.reshape(shape)
    else:
        raise ValueError(f"unknown distribution {dist!r}; expected one of {DISTRIBUTIONS}")
    return round_to_format(x, f)
