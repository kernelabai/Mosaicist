import numpy as np
import pytest

from mosaicist.verify import (
    compare_outputs, generate, get_format, make_suite, ordinal, round_to_format, tile_error_summary,
    ulp, ulp_distance,
)
from mosaicist.verify import guards


def test_ordinal_adjacent_values_differ_by_one():
    one_next_f32 = float(np.nextafter(np.float32(1), np.float32(2)))
    assert ulp_distance(np.array([1.0]), np.array([one_next_f32]), "f32")[0] == 1
    assert ulp_distance(np.array([1.0]), np.array([1.0078125]), "bf16")[0] == 1
    assert ulp_distance(np.array([1.0]), np.array([1.125]), "e4m3")[0] == 1
    # +0 and -0 share position 0; crossing zero counts both sides
    assert ordinal(np.array([-0.0]), "f16")[0] == 0
    tiny = get_format("f16").min_subnormal
    assert ulp_distance(np.array([-tiny]), np.array([tiny]), "f16")[0] == 2
    # the boundary between subnormals and normals is one step
    f = get_format("f16")
    largest_sub = f.min_normal - f.min_subnormal
    assert ulp_distance(np.array([largest_sub]), np.array([f.min_normal]), "f16")[0] == 1


def test_ordinal_matches_bit_patterns_for_float16():
    x = np.random.default_rng(0).standard_normal(10_000).astype(np.float16)
    x = x[np.isfinite(x) & (x > 0)]
    bits = x.view(np.uint16).astype(np.int64)
    assert np.array_equal(ordinal(x.astype(np.float64), "f16"), bits)


@pytest.mark.parametrize("fmt,np_dtype", [("f16", np.float16), ("f32", np.float32)])
def test_round_to_format_matches_numpy_casts(fmt, np_dtype):
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.standard_normal(20_000) * 10.0 ** rng.integers(-6, 6, 20_000),
                        [0.0, -0.0, 1e-9, 70000.0, -70000.0]])
    with np.errstate(over="ignore"):  # 70000 overflows float16 to inf on purpose
        expected = x.astype(np_dtype).astype(np.float64)
    assert np.array_equal(round_to_format(x, fmt), expected)


def test_round_to_format_bf16_and_fp8_saturation():
    # bf16 keeps 8 significant bits: 1 + 2**-8 is a tie that rounds to even (1.0)
    assert round_to_format(np.array([1 + 2**-8, 1 + 3 * 2**-8]), "bf16").tolist() == [1.0, 1.015625]
    assert round_to_format(np.array([1000.0, -470.0]), "e4m3").tolist() == [448.0, -448.0]
    assert np.isinf(round_to_format(np.array([1e6]), "e5m2"))[0]


def test_ulp_scale():
    assert ulp(np.array([1.0]), "bf16")[0] == 2**-7
    assert ulp(np.array([0.0]), "f32")[0] == 2**-149


def _case(seed=0, n=4096, fmt="bf16", noise_seed=None):
    """An fp64 oracle and a kernel output ~0.6 ulp of noise away from it, rounded to fmt."""
    oracle = np.random.default_rng(seed).standard_normal(n) * 3
    noise = np.random.default_rng(seed if noise_seed is None else noise_seed + 1000).standard_normal(n)
    return oracle, round_to_format(oracle + noise * ulp(oracle, fmt) * 0.6, fmt)


def test_gate_passes_identical_and_reports_bitwise():
    oracle, ref = _case()
    r = compare_outputs(ref, ref.copy(), oracle, "bf16")
    assert r.passed and r.bitwise_equal_fraction == 1.0 and r.max_ulp_ref_vs_cand == 0
    assert r.ulp_hist["0"] == ref.size


def test_gate_fails_less_accurate_candidate():
    oracle, ref = _case()
    worse = round_to_format(ref + 8 * ulp(ref, "bf16"), "bf16")
    r = compare_outputs(ref, worse, oracle, "bf16")
    assert not r.passed
    assert any("q0.5" in f for f in r.failures)
    assert r.worst_index is not None


def test_gate_allows_equally_accurate_different_rounding():
    oracle, ref = _case(seed=0)
    _, other = _case(seed=0, noise_seed=1)  # same oracle and error statistics, different bits
    r = compare_outputs(ref, other, oracle, "bf16")
    assert r.passed, r.summary()
    assert r.bitwise_equal_fraction < 1.0


def test_gate_catches_special_value_mismatch_and_shape():
    oracle, ref = _case(n=16)
    cand = ref.copy()
    ref = ref.copy()
    ref[3] = np.nan
    r = compare_outputs(ref, cand, oracle, "bf16")
    assert not r.passed and r.special_mismatch == 1
    r = compare_outputs(ref, cand[:8], oracle, "bf16")
    assert not r.passed and "shape" in r.failures[0]


def test_tile_error_summary_points_at_boundary():
    bad = np.zeros((256, 512), dtype=bool)
    bad[250:, 10] = True  # last tile row only
    s = tile_error_summary(bad, (128, 256))
    assert s["tiles"] == (2, 2) and s["bad_tiles"] == 1
    assert s["only_last_tile_row"] and s["boundary_only"]


def test_suite_generation_is_deterministic_and_representable():
    cases = make_suite({"M": 512, "N": 512, "K": 256}, {"M": 128, "N": 256, "K": 64}, seeds=(0,))
    names = {c.name.split("/")[0] for c in cases}
    assert names == {"aligned", "ragged_M", "ragged_N", "ragged_K", "minimal"}
    for dist in ("normal", "uniform", "wide", "cancel", "special"):
        a = generate((64, 32), dist, 7, "bf16")
        b = generate((64, 32), dist, 7, "bf16")
        assert np.array_equal(a, b, equal_nan=True)
        fin = a[np.isfinite(a)]
        assert np.array_equal(round_to_format(fin, "bf16"), fin)
    special = generate((64, 32), "special", 0, "f16")
    assert np.isnan(special).any() and np.isinf(special).any()


def test_guard_bands():
    total, off = guards.layout(1000, pad_nbytes=512)
    raw = guards.fill(total)
    raw[off : off + 1000] = 0  # in-bounds writes are fine
    assert guards.violations(raw, off, 1000) == []
    raw[off + 1000] = 7  # one byte past the end
    assert guards.violations(raw, off, 1000) == [(1000, 7)]
