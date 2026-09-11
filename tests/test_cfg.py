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


CUTLASS_WAITS = """.version 8.8
.target sm_90a
.address_size 64
.visible .entry k()
{
	mov.u32 %r1, 0;
$L__BB0_12:
	add.s32 %r5, %r1, 1;
	{
	.reg .pred P1;
	LAB_WAIT:
	mbarrier.try_wait.parity.shared::cta.b64 P1, [%r3], %r4;
	@P1 bra.uni DONE;
	bra.uni     LAB_WAIT;
	DONE:
	}
	wgmma.fence.sync.aligned;
	add.s32 %r1, %r1, 1;
	setp.lt.s32 %p2, %r1, 16;
	@%p2 bra $L__BB0_12;
	{
	.reg .pred P1;
	LAB_WAIT:
	mbarrier.try_wait.parity.shared::cta.b64 P1, [%r6], %r4;
	@P1 bra.uni DONE;
	bra.uni     LAB_WAIT;
	DONE:
	}
	ret;
}
"""


def test_scoped_inline_asm_labels_resolve_to_their_own_scope():
    """CUTLASS reuses LAB_WAIT/DONE in every wait scope; each branch must stay in its scope."""
    fn = parse(CUTLASS_WAITS).entry()
    bras = [i for i in fn.instrs if i.base == "bra"]
    first_scope_targets = {bras[0].operands[0], bras[1].operands[0]}
    last_scope_targets = {bras[3].operands[0], bras[4].operands[0]}
    assert first_scope_targets.isdisjoint(last_scope_targets)
    assert bras[2].operands[0] == "$L__BB0_12"  # function-scope labels keep their names
    cfg = build_cfg(fn)
    structural = cfg.structural_loops()
    assert len(structural) == 1 and len([lp for lp in cfg.loops if lp.spin]) == 2
    own = [i.opcode for i in cfg.own_instrs(structural[0])]
    assert any(op.startswith("mbarrier.try_wait") for op in own)


def test_dominators_and_unconditional_branches():
    cfg = build_cfg(parse(NESTED).entry())
    assert cfg.idom[0] == 0
    for b in cfg.blocks:
        last = b.instrs[-1] if b.instrs else None
        if last is not None and last.opcode == "ret":
            assert b.succs == []
