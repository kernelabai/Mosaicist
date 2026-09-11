from mosaicist.diagnose import diagnose
from mosaicist.ptx import align, diff, fingerprint


def _pair(ref_ptx, cand_ptx, fixtures, logs=True):
    kw = lambda name: {"ptxas_log": (fixtures / f"{name}.ptxas.log").read_text()} if logs else {}
    return fingerprint(ref_ptx, **kw("hopper_gemm_ref")), fingerprint(cand_ptx, **kw("hopper_gemm_v0"))


def test_alignment():
    a = align(["w", "x", "wait:1"], ["w", "x", "x", "wait:0"])
    # either x may be the inserted one; both alignments are optimal
    assert sorted(s.op for s in a.steps) == ["ins", "match", "match", "subst"]
    assert a.steps[-1].op == "subst"
    assert a.similarity == 0.5
    assert align([], []).similarity == 1.0
    b = align(["a"], ["b"])
    assert [s.op for s in b.steps] == ["del", "ins"]  # unrelated ops are not substitutions


def test_identical_kernels_have_zero_distance(ref_ptx):
    fp = fingerprint(ref_ptx)
    report = diff(fp, fp)
    assert report.discrepancies == []
    assert report.distance == 0.0
    assert diagnose(report) == []


def test_design_doc_example_rows(ref_ptx, cand_ptx, fixtures):
    """The worked example in DESIGN.md §7: naive v0 vs the warp-specialized reference."""
    report = diff(*_pair(ref_ptx, cand_ptx, fixtures))
    keys = report.keys()
    for expected in ("L0.warpgroups", "L0.cluster", "L0.setmaxnreg", "L1.tma_multicast",
                     "L2.warp_specialized", "L2.mma_wait_depth", "L5.spills"):
        assert expected in keys, expected
    assert report.get("L2.mma_wait_depth").ref == 1 and report.get("L2.mma_wait_depth").cand == 0
    per_stage = next(d for d in report.discrepancies if d.key.startswith("L1.mma_per_stage"))
    assert (per_stage.ref, per_stage.cand) == (4, 8)
    # the fused candidate loop is projected per role: producer ops don't pollute the mainloop alignment
    mma_order = next(d for d in report.discrepancies if d.key.startswith("L3.order") and d.context["ref_role"] == "mma")
    assert mma_order.context["projected"]
    assert not any(t.startswith("tma.load") for t in mma_order.cand)
    assert 0 < report.distance < 1


def test_design_doc_example_fixes(ref_ptx, cand_ptx, fixtures):
    fixes = diagnose(diff(*_pair(ref_ptx, cand_ptx, fixtures)))
    phases = [f.phase for f in fixes]
    assert phases == sorted(phases)  # converge in dependency order
    first = fixes[0]
    assert (first.phase, first.tag) == ("P1", "rewrite")
    assert {"L0.warpgroups", "L2.warp_specialized", "L5.spills"} <= set(first.keys)
    assert "num_compute_wgs=2" in first.detail

    by_key = {k: f for f in fixes for k in f.keys}
    assert by_key["L1.tma_multicast"] is by_key["L0.cluster"]  # one fix closes both
    assert "collective_axes" in by_key["L1.tma_multicast"].detail
    assert by_key["L2.mma_wait_depth"].tag == "knob"
    assert "wgmma_wait(1)" in by_key["L2.mma_wait_depth"].detail
    assert "memory_registers=40" in by_key["L0.setmaxnreg"].detail
    # every discrepancy is closed by exactly one fix
    closed = [k for f in fixes for k in f.keys]
    assert len(closed) == len(set(closed))


def test_without_machine_data_no_l5_rows(ref_ptx, cand_ptx, fixtures):
    report = diff(*_pair(ref_ptx, cand_ptx, fixtures, logs=False))
    assert not any(d.layer == "L5" for d in report.discrepancies)
    fixes = diagnose(report)
    assert fixes[0].phase == "P1" and "spills" not in fixes[0].title


def test_fp_flavor_rows():
    tmpl = """.version 8.5
.target sm_90a
.address_size 64
.visible .entry k()
{{
	{op} %f1, %f2;
	ret;
}}
"""
    ref = fingerprint(tmpl.format(op="ex2.approx.ftz.f32"))
    cand = fingerprint(tmpl.format(op="ex2.approx.f32"))
    report = diff(ref, cand)
    row = report.get("L4.ex2")
    assert row is not None and row.weight == 2.0
    fix = next(f for f in diagnose(report) if "L4.ex2" in f.keys)
    assert fix.phase == "P5"
