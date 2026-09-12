"""M0 capture workers, M2 contract extraction, the catalog, and the v0 translator.

None of these need a GPU: the pieces that do (running a kernel, reading CUPTI) are
exercised on hardware by experiments/loop, and what is unit-tested here is the
bookkeeping around them, which is where the mistakes actually live.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaicist import catalog
from mosaicist.bundle import Bundle
from mosaicist.capture import KIND_OF, collect_artifacts, dump_env, _pick
from mosaicist.contract import Contract, Operand, block_k_for, from_capture, stages_for
from mosaicist.translate import TemplateTranslator, emit_v0, repair_hints
from mosaicist.verify.numerics import NumericsReport


def test_dump_env_points_both_compilers_at_the_directory(tmp_path):
    p = dump_env("pallas", tmp_path / "d")
    assert p["MOSAIC_GPU_DUMP_TO"] == str(tmp_path / "d")
    assert p["MOSAIC_GPU_DUMP_SASS"] == "1"
    c = dump_env("cutedsl", tmp_path / "d")
    assert c["CUTE_DSL_DUMP_DIR"] == str(tmp_path / "d")
    assert set(p) & set(c) == set(), "the two compilers must not share dump switches"
    with pytest.raises(ValueError):
        dump_env("nvcc", tmp_path)


def test_kind_maps_compiler_to_bundle_role():
    assert KIND_OF == {"pallas": "candidate", "cutedsl": "reference"}


def test_pick_prefers_the_named_then_the_largest(tmp_path):
    small, big, named = (tmp_path / n for n in ("a.ptx", "b.ptx", "wanted.ptx"))
    small.write_text("x")
    big.write_text("x" * 100)
    named.write_text("xx")
    assert _pick([small, big, named]) == big
    assert _pick([small, big, named], prefer="wanted") == named
    assert _pick([]) is None


def test_collect_artifacts_gathers_dumps_into_the_bundle(tmp_path):
    dump = tmp_path / "dump"
    (dump / "nested").mkdir(parents=True)
    (dump / "nested" / "k.ptx").write_text(".version 8.7\n" + "x" * 50)
    (dump / "small.ptx").write_text("tiny")
    (dump / "k.sass").write_text("sass text")
    (dump / "ptxas.log").write_text("ptxas info : Used 200 registers")
    out = tmp_path / "bundle"

    found = collect_artifacts("pallas", dump, out)
    assert found == {"ptx": "kernel.ptx", "sass": "kernel.sass", "ptxas_log": "ptxas.log"}
    assert "8.7" in (out / "kernel.ptx").read_text(), "should pick the larger PTX"
    assert (out / "kernel.sass").exists() and (out / "ptxas.log").exists()


def test_collect_artifacts_tolerates_an_empty_dump(tmp_path):
    (tmp_path / "dump").mkdir()
    assert collect_artifacts("pallas", tmp_path / "dump", tmp_path / "b") == {
        "ptx": None, "sass": None, "ptxas_log": None}


def _fake_bundle(tmp_path, ptx_text, grid=(4, 8, 1)):
    d = tmp_path / "ref"
    d.mkdir()
    (d / "kernel.ptx").write_text(ptx_text)
    b = Bundle(kind="reference", name="ref", arch="sm_100a", ptx="kernel.ptx", root=d)
    b.launch.grid = list(grid)
    b.launch.block = [128, 1, 1]
    return b


PTX = """//
.version 8.7
.target sm_100a
.address_size 64
.visible .entry k(.param .u64 p0)
{
	.reg .b32 %r<4>;
	tcgen05.mma.cta_group::1.kind::mxf4nvf4 [%r1], [%r2], [%r3];
	ret;
}
"""


def test_contract_from_capture_infers_tile_from_grid(tmp_path):
    b = _fake_bundle(tmp_path, PTX, grid=(4, 8, 1))
    c = from_capture(b, dims={"m": 512, "n": 1024, "k": 2048},
                     operands=[Operand("a", (512, 2048), "float4_e2m1fn")],
                     out_shape=(512, 1024))
    assert c.mma == "tcgen05" and c.mma_kind == "mxf4nvf4"
    assert c.tile == {"m": 128, "n": 128}, "512/4 and 1024/8"
    assert c.arch == "sm_100a"


def test_contract_round_trips_through_json(tmp_path):
    c = Contract(name="x", arch="sm_90a", operands=[Operand("a", (8, 8), "bfloat16")],
                 dims={"m": 8}, tile={"m": 8})
    p = c.save(tmp_path / "c.json")
    back = Contract.load(p)
    assert back.operands[0].dtype == "bfloat16" and back.dims == {"m": 8}
    assert json.loads(p.read_text())["name"] == "x"


def test_block_k_and_stages_stay_inside_the_occupancy_budget():
    c = Contract(name="g", arch="sm_100a",
                 operands=[Operand("a", (512, 2048), "bfloat16")],
                 dims={"m": 512, "n": 512, "k": 2048}, tile={"m": 128, "n": 128})
    bk = block_k_for(c)
    st = stages_for(c, bk)
    assert bk in (64, 128, 256, 512) and c.dims["k"] % bk == 0
    per_stage = (128 + 128) * bk * 16 // 8
    assert st >= 1 and st * per_stage + 128 * 128 * 2 <= 227 * 1024 // 2


@pytest.mark.parametrize("mma,needle", [("wgmma", "plgpu.wgmma("),
                                        ("tcgen05", "plgpu.tcgen05_mma(")])
def test_v0_emits_compilable_source_for_both_mma_families(tmp_path, mma, needle):
    c = Contract(name="gemm", arch="sm_90a" if mma == "wgmma" else "sm_100a",
                 operands=[Operand("a", (256, 256), "bfloat16"),
                           Operand("b", (256, 256), "bfloat16")],
                 out_shape=(256, 256), dims={"m": 256, "n": 256, "k": 256},
                 tile={"m": 128, "n": 128}, mma=mma)
    path = emit_v0(c, tmp_path / "v0.py")
    src = path.read_text()
    compile(src, str(path), "exec")
    assert needle in src
    assert "def build()" in src and "def reference(" in src, "capture needs both"


def test_v0_records_what_it_deliberately_left_out(tmp_path):
    c = Contract(name="g", arch="sm_100a", operands=[Operand("a", (8, 8), "bfloat16")],
                 dims={"m": 128, "n": 128, "k": 128}, tile={"m": 128, "n": 128},
                 mma="tcgen05", deferred=["warp specialization", "persistent scheduling"])
    src = emit_v0(c, tmp_path / "v0.py").read_text()
    assert "warp specialization" in src and "persistent scheduling" in src


def test_catalog_covers_every_knob_and_maps_diff_rows():
    assert {"stages", "tile_m", "num_threads"} <= set(catalog.knobs())
    assert any(c.concept == "warp roles" for c in catalog.for_row("L0.warpgroups"))
    assert any(c.concept == "TMA load" for c in catalog.for_row("L1.tma"))
    assert all(c.tag in ("core", "low-level", "gap") for c in catalog.CONSTRUCTS)
    assert catalog.gaps(), "the report needs at least one known gap"


def test_repair_hints_separate_boundary_failures_from_arithmetic():
    localized = NumericsReport(False, "bf16", 100, [], 1.0, {}, 2, 0, ["q1 exceeded"])
    localized.worse_fraction = 0.02
    assert "boundary" in " ".join(repair_hints(localized))

    spread = NumericsReport(False, "bf16", 100, [], 0.1, {}, 9, 0, ["q1 exceeded"])
    spread.worse_fraction = 0.9
    assert "accumulator" in " ".join(repair_hints(spread))

    ok = NumericsReport(True, "bf16", 100, [], 1.0, {}, 0, 0, [])
    assert repair_hints(ok) == []
