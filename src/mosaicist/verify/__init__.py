"""Numerical equivalence: formats, input suites, the oracle-relative gate, guard bands."""

from .formats import FORMATS, FloatFormat, get_format, ordinal, round_to_format, ulp, ulp_distance
from .numerics import GateConfig, NumericsReport, bitwise_equal, compare_outputs, tile_error_summary
from .suites import Case, generate, make_suite, shape_variants

__all__ = [
    "FORMATS", "FloatFormat", "get_format", "ordinal", "round_to_format", "ulp", "ulp_distance",
    "GateConfig", "NumericsReport", "bitwise_equal", "compare_outputs", "tile_error_summary",
    "Case", "generate", "make_suite", "shape_variants",
]
