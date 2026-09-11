"""Regression tests on real compiler output (captured on an H100 PCIe, 2026-09-11).

- cutedsl_dense_gemm_sm90.ptx: CUTLASS 4.7.1 CuTeDSL examples/.../hopper/kernel/dense_gemm/dense_gemm.py,
  fp16, tile 128x256, cluster 2x1. Non-persistent; warp 0 prefetches, then the same warps run a
  fused MMA + next-TMA mainloop.
- pallas_hopper_matmul_sm90.ptx: JAX 0.11.1 pallas/ops/gpu/hopper_matmul_mgpu.py, tile_m=64, tile_n=256,
  wg_dimension=M, cluster_dimension=M. Persistent and warp-specialized (2 compute + 1 memory warpgroup).
- pallas_hopper_matmul_dr1_sm90.ptx: the same with delay_release=1 (the P3 fix Diagnose proposed).

Launch records come from CUPTI (grid/cluster are not in PTX).
"""

from pathlib import Path

import pytest

from mosaicist.diagnose import diagnose
from mosaicist.ptx import diff, fingerprint, parse

REAL = Path(__file__).parent / "fixtures" / "real"
REF_LAUNCH = {"grid": [64, 32, 1], "block": [256, 1, 1], "cluster": [2, 1, 1], "dynamic_smem": 197632}
CAND_LAUNCH = {"grid": [114, 1, 1], "block": [384, 1, 1], "cluster": [2, 1, 1], "dynamic_smem": 229440}


def _fp(name: str, launch: dict, log: str | None = None):
    text = (REAL / f"{name}.ptx").read_text()
    log_text = (REAL / f"{log}.ptxas.log").read_text() if log else None
    return fingerprint(text, launch=launch, ptxas_log=log_text)


@pytest.fixture(scope="module")
def ref():
    return _fp("cutedsl_dense_gemm_sm90", REF_LAUNCH, "cutedsl_dense_gemm_sm90")


@pytest.fixture(scope="module")
def cand():
    return _fp("pallas_hopper_matmul_sm90", CAND_LAUNCH, "pallas_hopper_matmul_sm90")


def test_cutlass_scoped_wait_labels_parse():
    fn = parse((REAL / "cutedsl_dense_gemm_sm90.ptx").read_text()).entry()
    labels = [s.name for s in fn.body if not hasattr(s, "opcode")]
    assert any(name.startswith("LAB_WAIT@") for name in labels)  # scope-qualified
    assert len(labels) == len(set(labels))


def test_reference_structure(ref):
    sk, st = ref.skeleton, ref.structure
    assert (sk["warpgroups"], sk["cluster"], sk["grid"]) == (2, [2, 1, 1], [64, 32, 1])
    assert not st["warp_specialized"] and st["prefetch_prologue"] and not st["persistent"]
    assert st["wgmma_wait_depths"] == [1]
    assert set(st["roles"]) == {"load", "mma+load"}
    assert ref.mma_signatures == {"wgmma:m64n256k16.f32.f16.f16/ss": 8}
    assert ref.loop("mma+load")[0].mma_per_stage() == 4


def test_candidate_structure(cand):
    st = cand.structure
    assert st["warp_specialized"] and st["persistent"] and not st["prefetch_prologue"]
    assert st["wgmma_wait_depths"] == [0]
    assert set(st["roles"]) == {"outer", "load", "mma"}  # tile loop around producer / consumer
    assert cand.skeleton["setmaxnreg"] == [["dec", 40], ["inc", 232]]
    assert any(a.startswith("C7519") for a in cand.machine["ptxas_advisories"])


def test_real_diff_and_diagnosis(ref, cand):
    report = diff(ref, cand)
    keys = report.keys()
    assert {"L2.warp_specialized", "L2.persistent", "L2.mma_wait_depth", "L5.ptxas_C7519"} <= keys
    assert "L0.cluster" not in keys  # cluster comes from the launch record and matches
    assert not any(k.startswith("L1.mma_per_stage") for k in keys)  # same MMAs per stage
    fixes = diagnose(report)
    first = fixes[0]
    assert (first.phase, first.tag) == ("P1", "investigate") and "producer warpgroup" in first.title
    assert not any("Reduce num_compute_wgs" in f.detail for f in fixes)
    wait = next(f for f in fixes if "L2.mma_wait_depth" in f.keys)
    assert wait.tag == "knob" and "wgmma_wait(1)" in wait.detail
    persist = next(f for f in fixes if "L2.persistent" in f.keys)
    assert "L0.grid" in persist.keys


def test_delay_release_fix_closes_the_row(ref, cand):
    fixed = _fp("pallas_hopper_matmul_dr1_sm90", CAND_LAUNCH)
    before, after = diff(ref, cand), diff(ref, fixed)
    assert "L2.mma_wait_depth" in before.keys() and "L2.mma_wait_depth" not in after.keys()
    assert fixed.structure["wgmma_wait_depths"] == [1]
    mainloop = lambda r: next(p for p in r.loop_pairs if p.ref_role == "mma+load").alignment.similarity
    assert mainloop(after) > mainloop(before)
