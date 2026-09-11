from mosaicist.ptx import build_cfg, parse

NESTED = """.version 8.5
.target sm_90a
.address_size 64
.visible .entry k()
{
	mov.u32 %r1, 0;
$TILE:
	mov.u32 %r2, 0;
$K:
	add.s32 %r5, %r2, 1;
$SPIN:
	mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r3], %r4;
	@!%p1 bra $SPIN;
	wgmma.fence.sync.aligned;
	add.s32 %r2, %r2, 1;
	setp.lt.s32 %p2, %r2, 16;
	@%p2 bra $K;
	add.s32 %r1, %r1, 1;
	setp.lt.s32 %p3, %r1, 4;
	@%p3 bra $TILE;
	ret;
}
"""


def test_nested_loops_and_spin_detection():
    cfg = build_cfg(parse(NESTED).entry())
    spins = [lp for lp in cfg.loops if lp.spin]
    structural = cfg.structural_loops()
    assert len(spins) == 1
    assert len(structural) == 2
    outer, inner = sorted(structural, key=lambda lp: len(lp.blocks), reverse=True)
    assert inner.parent == outer.id
    assert (outer.depth, inner.depth) == (1, 2)
    # the spin loop's poll is folded into the K loop's own body
    own = [i.opcode for i in cfg.own_instrs(inner)]
    assert any(op.startswith("mbarrier.try_wait") for op in own)
    assert not any(op.startswith("mbarrier") for op in (i.opcode for i in cfg.own_instrs(outer)))


def test_dominators_and_unconditional_branches():
    cfg = build_cfg(parse(NESTED).entry())
    assert cfg.idom[0] == 0
    for b in cfg.blocks:
        last = b.instrs[-1] if b.instrs else None
        if last is not None and last.opcode == "ret":
            assert b.succs == []
