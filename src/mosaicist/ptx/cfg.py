"""Basic blocks, dominators, and natural loops over a parsed PTX function."""

from __future__ import annotations

from dataclasses import dataclass, field

from .parse import Function, Instr, Label

_TERMINATORS = {"bra", "ret", "exit", "brx", "trap"}


@dataclass
class Block:
    id: int
    labels: list[str] = field(default_factory=list)
    instrs: list[Instr] = field(default_factory=list)
    succs: list[int] = field(default_factory=list)
    preds: list[int] = field(default_factory=list)


@dataclass
class Loop:
    id: int
    header: int
    blocks: set[int]
    parent: int | None = None
    children: list[int] = field(default_factory=list)
    spin: bool = False  # a busy-wait on an mbarrier, not a structural loop
    depth: int = 0  # structural nesting depth (spin loops don't count)


@dataclass
class CFG:
    blocks: list[Block]
    loops: list[Loop]
    idom: dict[int, int]

    def loop_of(self, block_id: int) -> Loop | None:
        """Innermost loop containing the block."""
        best = None
        for lp in self.loops:
            if block_id in lp.blocks and (best is None or len(lp.blocks) < len(best.blocks)):
                best = lp
        return best

    def structural_loops(self) -> list[Loop]:
        return [lp for lp in self.loops if not lp.spin]

    def own_instrs(self, loop: Loop) -> list[Instr]:
        """Instructions of `loop` excluding its structural (non-spin) sub-loops.

        Spin loops are folded into their parent, so an mbarrier wait written as
        a try_wait/branch spin still shows up in the enclosing loop's body.
        """
        excluded: set[int] = set()
        for cid in loop.children:
            child = self.loops[cid]
            if not child.spin:
                excluded |= child.blocks
        out = []
        for b in sorted(loop.blocks - excluded):
            out.extend(self.blocks[b].instrs)
        return out


def _is_terminator(ins: Instr) -> bool:
    return ins.base in _TERMINATORS


def build_cfg(fn: Function) -> CFG:
    blocks: list[Block] = [Block(0)]
    for item in fn.body:
        cur = blocks[-1]
        if isinstance(item, Label):
            if cur.instrs:
                blocks.append(Block(len(blocks)))
                cur = blocks[-1]
            cur.labels.append(item.name)
        else:
            cur.instrs.append(item)
            if _is_terminator(item):
                blocks.append(Block(len(blocks)))
    if not blocks[-1].instrs and not blocks[-1].labels and len(blocks) > 1:
        blocks.pop()

    label_block = {lab: b.id for b in blocks for lab in b.labels}
    for b in blocks:
        last = b.instrs[-1] if b.instrs else None
        falls_through = True
        if last is not None and last.base == "bra" and last.operands:
            target = label_block.get(last.operands[-1])
            if target is not None:
                b.succs.append(target)
            falls_through = last.pred is not None
        elif last is not None and last.base in ("ret", "exit", "trap", "brx"):
            falls_through = last.pred is not None
        if falls_through and b.id + 1 < len(blocks):
            if b.id + 1 not in b.succs:
                b.succs.append(b.id + 1)
    for b in blocks:
        for s in b.succs:
            blocks[s].preds.append(b.id)

    idom = _dominators(blocks)
    loops = _natural_loops(blocks, idom)
    return CFG(blocks, loops, idom)


def _reachable(blocks: list[Block]) -> list[int]:
    seen, order, stack = set(), [], [0]
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        seen.add(b)
        order.append(b)
        stack.extend(reversed(blocks[b].succs))
    return order


def _dominators(blocks: list[Block]) -> dict[int, int]:
    """Cooper-Harvey-Kennedy iterative immediate-dominator computation."""
    # reverse postorder over reachable blocks
    post, seen = [], set()

    def dfs(start: int) -> None:
        stack = [(start, iter(blocks[start].succs))]
        seen.add(start)
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                post.append(node)
                stack.pop()
            elif nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, iter(blocks[nxt].succs)))

    if not blocks:
        return {}
    dfs(0)
    rpo = list(reversed(post))
    order = {b: i for i, b in enumerate(rpo)}
    idom: dict[int, int] = {0: 0}

    def intersect(a: int, b: int) -> int:
        while a != b:
            while order[a] > order[b]:
                a = idom[a]
            while order[b] > order[a]:
                b = idom[b]
        return a

    changed = True
    while changed:
        changed = False
        for b in rpo[1:]:
            preds = [p for p in blocks[b].preds if p in idom]
            if not preds:
                continue
            new = preds[0]
            for p in preds[1:]:
                new = intersect(p, new)
            if idom.get(b) != new:
                idom[b] = new
                changed = True
    return idom


def dominates(idom: dict[int, int], a: int, b: int) -> bool:
    while True:
        if a == b:
            return True
        if b not in idom or idom[b] == b:
            return False
        b = idom[b]


def _natural_loops(blocks: list[Block], idom: dict[int, int]) -> list[Loop]:
    by_header: dict[int, set[int]] = {}
    for b in blocks:
        if b.id not in idom:
            continue
        for s in b.succs:
            if s in idom and dominates(idom, s, b.id):  # back edge b -> s
                body = by_header.setdefault(s, {s})
                stack = [b.id]
                while stack:
                    x = stack.pop()
                    if x in body:
                        continue
                    body.add(x)
                    stack.extend(p for p in blocks[x].preds if p in idom)
    loops = [Loop(i, h, body) for i, (h, body) in enumerate(sorted(by_header.items()))]

    # nesting: parent is the smallest strictly-enclosing loop
    for lp in loops:
        enclosing = [o for o in loops if o is not lp and lp.blocks < o.blocks]
        if enclosing:
            parent = min(enclosing, key=lambda o: len(o.blocks))
            lp.parent = parent.id
            parent.children.append(lp.id)

    for lp in loops:
        lp.spin = _is_spin(lp, blocks)

    for lp in loops:
        depth, p = 0 if lp.spin else 1, lp.parent
        while p is not None:
            if not loops[p].spin:
                depth += 1
            p = loops[p].parent
        lp.depth = depth
    return loops


_SPIN_OK = {"mbarrier", "bra", "nanosleep", "setp", "selp", "mov", "and", "xor", "add", "not"}


def _is_spin(lp: Loop, blocks: list[Block]) -> bool:
    """A small loop whose only real work is polling an mbarrier."""
    instrs = [i for b in lp.blocks for i in blocks[b].instrs]
    if not instrs or len(instrs) > 12 or lp.children:
        return False
    polls = [
        i for i in instrs
        if i.base == "mbarrier" and ("try_wait" in i.parts or "test_wait" in i.parts)
    ]
    return bool(polls) and all(i.base in _SPIN_OK for i in instrs)
