"""M3: the knob space, the tuner, and the convergence loop driver.

The loop is exercised with a fake runner that synthesises bundles, so the control flow
-- accept, propose, stop -- is tested without a GPU. What the fake cannot check (that a
capture really produces those bundles) is covered on hardware by experiments/loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaicist.bundle import Bundle
from mosaicist.converge.knobs import KnobSpace, from_env, key, to_env
from mosaicist.converge.loop import LoopConfig, converge
from mosaicist.converge.tune import Tuner, knobs_for_fixes
from mosaicist.diagnose import Fix

REF_PTX = """//
.version 8.7
.target sm_90a
.address_size 64
.visible .entry ref(.param .u64 p)
{
	.reg .b32 %r<4>;
	wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%r1}, %r2, %r3;
	ret;
}
"""


def _bundle(tmp: Path, name: str, times: list[float]) -> Bundle:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "kernel.ptx").write_text(REF_PTX)
    b = Bundle(kind="candidate", name=name, arch="sm_90a", ptx="kernel.ptx",
               timings=times, root=d)
    b.save(d)
    return b


def test_setting_round_trips_through_the_environment(monkeypatch):
    monkeypatch.setenv(*next(iter(to_env({"stages": 4}).items())))
    assert from_env({"stages": 2, "tile_m": 128}) == {"stages": 4, "tile_m": 128}


def test_knob_space_walks_one_knob_at_a_time():
    sp = KnobSpace({"stages": [2, 4], "tile_m": [128, 64]})
    assert sp.default() == {"stages": 2, "tile_m": 128}
    ns = list(sp.neighbours(sp.default()))
    assert {"stages": 4, "tile_m": 128} in ns and {"stages": 2, "tile_m": 64} in ns
    for n in ns:
        differing = [k for k in n if n[k] != sp.default()[k]]
        assert len(differing) == 1, "each neighbour must differ in exactly one knob"


def test_knob_space_clamps_settings_it_does_not_declare():
    sp = KnobSpace({"stages": [1]})
    assert sp.clamp({"stages": 1, "gone": 7}) == {"stages": 1}


def test_tuner_follows_the_ranked_fix_before_sweeping():
    sp = KnobSpace({"tile_m": [128, 64], "stages": [2, 4, 8]})
    t = Tuner(sp)
    start = sp.default()
    t.mark(start)
    depth = Fix(phase="P2", tag="knob", title="pipeline depth differs", detail="",
                keys=["L2.pipeline_barriers"], weight=1.0)
    assert "stages" in knobs_for_fixes([depth])
    proposal = t.propose(start, [depth])
    assert proposal["stages"] != start["stages"], "should turn the implicated knob"
    assert proposal["tile_m"] == start["tile_m"]


def test_tuner_exhausts_the_space_and_stops():
    sp = KnobSpace({"stages": [1, 2]})
    t = Tuner(sp)
    seen = []
    setting = sp.default()
    for _ in range(5):
        t.mark(setting)
        seen.append(key(setting))
        nxt = t.propose(setting)
        if nxt is None:
            break
        setting = nxt
    assert len(set(seen)) == 2 and nxt is None


def test_loop_accepts_a_faster_candidate_and_reports_the_ratio(tmp_path):
    ref = _bundle(tmp_path, "ref", [100.0] * 5)
    # the first setting must be slower than the reference or the loop stops immediately,
    # which is the designed behaviour: reaching the reference's noise floor is the goal
    times = {1: 150.0, 2: 60.0, 4: 61.0}

    def runner(setting, outdir):
        return _bundle(tmp_path, Path(outdir).name, [times[setting["stages"]]] * 5)

    res = converge(ref, "unused:build", tmp_path / "run",
                   space=KnobSpace({"stages": [1, 2, 4]}),
                   config=LoopConfig(max_steps=5, gate_numerics=False), runner=runner)
    assert [s.time for s in res.steps][0] == 150.0
    assert res.best is not None and res.best.time == 60.0
    assert res.converged, "60 us against a 100 us reference beats the noise floor"
    assert "0.60x reference" in res.summary()
    saved = json.loads((tmp_path / "run" / "converge.json").read_text())
    assert saved["converged"] is True and len(saved["steps"]) == len(res.steps)


def test_loop_rejects_candidates_that_fail_the_numerics_gate(tmp_path):
    ref = _bundle(tmp_path, "ref", [100.0] * 5)

    def runner(setting, outdir):
        return _bundle(tmp_path, Path(outdir).name, [10.0] * 5)

    # every candidate is fast but the gate says no, so nothing may be accepted
    import mosaicist.converge.loop as loop_mod

    orig = loop_mod._numerics_ok
    loop_mod._numerics_ok = lambda *a, **k: False
    try:
        res = converge(ref, "unused:build", tmp_path / "run2",
                       space=KnobSpace({"stages": [1, 2]}),
                       config=LoopConfig(max_steps=3), runner=runner)
    finally:
        loop_mod._numerics_ok = orig
    assert res.best is None and not res.converged
    assert all(not s.accepted for s in res.steps)
    assert all(not s.numerics_pass for s in res.steps)


def test_loop_stops_when_the_knob_space_runs_out(tmp_path):
    ref = _bundle(tmp_path, "ref", [1.0] * 5)  # unreachably fast

    def runner(setting, outdir):
        return _bundle(tmp_path, Path(outdir).name, [500.0] * 5)

    res = converge(ref, "unused:build", tmp_path / "run3",
                   space=KnobSpace({"stages": [1, 2, 4]}),
                   config=LoopConfig(max_steps=99, gate_numerics=False), runner=runner)
    assert len(res.steps) == 3, "one step per distinct setting, then the space is spent"
    assert not res.converged


def test_loop_survives_settings_that_cannot_be_built(tmp_path):
    """A knob space always contains invalid combinations; one must not end the run."""
    ref = _bundle(tmp_path, "ref", [1.0] * 5)
    seen = []

    def runner(setting, outdir):
        seen.append(setting["stages"])
        if setting["stages"] == 2:
            raise ValueError("needs 300000 bytes of smem")
        return _bundle(tmp_path, Path(outdir).name, [50.0] * 5)

    res = converge(ref, "unused:build", tmp_path / "run4",
                   space=KnobSpace({"stages": [1, 2, 4]}),
                   config=LoopConfig(max_steps=9, gate_numerics=False), runner=runner)
    assert 2 in seen and len(seen) == 3, "the bad setting is tried once, then passed over"
    failed = [s for s in res.steps if s.time == float("inf")]
    assert len(failed) == 1 and "smem" in failed[0].reason
    assert res.best is not None and res.best.time == 50.0
    assert "----" in res.summary()
