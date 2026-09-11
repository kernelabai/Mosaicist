from mosaicist.ptx import classify, fingerprint, parse, parse_ptxas_log, parse_sass


def op(text: str):
    fn = parse(f".version 8.5\n.target sm_90a\n.address_size 64\n.visible .entry k()\n{{\n\t{text}\n}}\n").entry()
    return classify(fn.instrs[0])


def test_classify_tensor_core_ops():
    o = op("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%f1}, %rd1, %rd2, p, 1, 1, 0, 1;")
    assert (o.family, o.token, o.attrs["dtypes"], o.attrs["a_src"]) == ("mma", "wgmma:m64n256k16", "f32.bf16.bf16", "ss")
    o = op("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%f1}, {%r1, %r2}, %rd2, p, 1, 1, 1;")
    assert o.token == "wgmma:m64n128k16.rs"
    assert op("wgmma.wait_group.sync.aligned 1;").token == "wgmma.wait:1"
    o = op("tcgen05.mma.cta_group::2.kind::f16 [%r1], %rd1, %rd2, %r3, %p1;")
    assert (o.family, o.token) == ("mma", "tcgen05.mma:2.f16")
    assert op("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f1}, {%r1}, {%r2}, {%f2};").token == "mma.sync:m16n8k16"


def test_classify_data_movement_and_sync():
    o = op("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
           "[%r1], [%rd1, {%r2, %r3, %r4}], [%r5], %rs1;")
    assert (o.family, o.token, o.attrs["multicast"]) == ("tma.load", "tma.load:3d.mc", True)
    o = op("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd1, {%r1, %r2}], [%r3];")
    assert (o.family, o.token) == ("tma.store", "tma.store:2d")
    assert op("cp.async.bulk.wait_group.read 1;").token == "bulk.wait:1.read"
    assert op("cp.async.cg.shared.global [%r1], [%rd1], 16;").family == "cp.async"
    assert op("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r1], 1024;").token == "mbar.arrive_tx"
    assert op("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%r1];").token == "mbar.arrive:remote"
    assert op("setmaxnreg.inc.sync.aligned.u32 232;").token == "setmaxnreg:inc.232"
    assert op("fence.proxy.async.shared::cta;").token == "fence.proxy.async"
    assert op("barrier.cluster.wait.aligned;").token == "cluster.wait"
    o = op("st.global.v4.b32 [%rd1], {%r1, %r2, %r3, %r4};")
    assert (o.family, o.attrs["bits"]) == ("st.global", 128)
    assert op("add.s32 %r1, %r2, %r3;") is None


def test_reference_fingerprint(ref_ptx, fixtures):
    fp = fingerprint(ref_ptx, ptxas_log=(fixtures / "hopper_gemm_ref.ptxas.log").read_text())
    sk, st = fp.skeleton, fp.structure
    assert (sk["threads"], sk["warpgroups"], sk["cluster"]) == (384, 3, [2, 1, 1])
    assert sk["setmaxnreg"] == [["dec", 40], ["inc", 240]]
    assert st["warp_specialized"] and not st["persistent"]
    assert st["roles"] == ["load", "mma"]
    assert st["wgmma_wait_depths"] == [1]  # the drain after the loop is not in the mainloop
    assert st["mbarrier_inits"] == 8 and st["epilogue"] == "tma_store"
    mma_loop = fp.loop("mma")[0]
    assert mma_loop.mma_per_stage() == 4
    assert mma_loop.tokens[0] == "mbar.wait"  # spin-wait folded into the loop
    assert fp.mma_signatures == {"wgmma:m64n256k16.f32.bf16.bf16/ss": 4}
    assert fp.machine == {"registers": 168, "stack_frame": 0, "spill_stores": 0, "spill_loads": 0,
                          "static_smem": 0}
    # .loc maps tokens back to CuTeDSL source lines
    assert "examples/python/CuTeDSL/hopper/dense_gemm.py:292" in mma_loop.token_locs


def test_candidate_fingerprint(cand_ptx):
    fp = fingerprint(cand_ptx)
    assert fp.skeleton["warpgroups"] == 1 and fp.skeleton["cluster"] == [1, 1, 1]
    assert fp.structure["roles"] == ["mma+load"]
    assert not fp.structure["warp_specialized"]
    assert fp.structure["wgmma_wait_depths"] == [0]
    assert fp.loops[0].mma_per_stage() == 8


def test_persistent_detection():
    text = """.version 8.5
.target sm_90a
.address_size 64
.visible .entry k()
{
$TILE:
	mov.u32 %r2, 0;
$K:
	wgmma.fence.sync.aligned;
	wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%f1}, %rd1, %rd2, p, 1, 1, 0, 1;
	wgmma.commit_group.sync.aligned;
	add.s32 %r2, %r2, 1;
	@%p2 bra $K;
	st.global.v4.b32 [%rd3], {%r1, %r2, %r3, %r4};
	@%p3 bra $TILE;
	ret;
}
"""
    fp = fingerprint(text)
    assert fp.structure["persistent"] and fp.structure["mma_loop_depth"] == 2


def test_fully_unrolled_kernel_falls_back_to_kernel_region():
    text = """.version 8.5
.target sm_90a
.address_size 64
.visible .entry k()
{
	wgmma.fence.sync.aligned;
	wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%f1}, %rd1, %rd2, p, 1, 1, 0, 1;
	wgmma.commit_group.sync.aligned;
	wgmma.wait_group.sync.aligned 0;
	ret;
}
"""
    fp = fingerprint(text)
    assert [lp.role for lp in fp.loops] == ["kernel"]


def test_fp_flavor_and_logs():
    text = """.version 8.5
.target sm_90a
.address_size 64
.visible .entry k()
{
	ex2.approx.ftz.f32 %f1, %f2;
	fma.rn.f32 %f3, %f1, %f2, %f4;
	cvt.rn.bf16x2.f32 %r1, %f1, %f2;
	add.s32 %r2, %r3, 1;
	ret;
}
"""
    fp = fingerprint(text)
    assert fp.fp_flavor == {"ex2.approx.ftz.f32": 1, "fma.rn.f32": 1, "cvt.rn.bf16x2.f32": 1}
    log = ("ptxas info    : Compiling entry function 'a' for 'sm_90a'\n"
           "ptxas info    : Used 32 registers\n"
           "ptxas info    : Compiling entry function 'b' for 'sm_90a'\n"
           "    8 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads\n"
           "ptxas info    : Used 255 registers, 1024 bytes smem\n")
    assert parse_ptxas_log(log, "b") == {"registers": 255, "stack_frame": 8, "spill_stores": 8,
                                         "spill_loads": 8, "static_smem": 1024}
    sass = "        /*0010*/                   STL [R1], R2 ;\n        /*0020*/                   LDL R3, [R1] ;\n"
    assert parse_sass(sass) == {"sass_instrs": 2, "sass_stl": 1, "sass_ldl": 1}
