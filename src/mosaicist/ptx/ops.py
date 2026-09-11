"""Classify PTX instructions into the families the fingerprint cares about.

Each classified instruction gets a *family* (counted in L1), optionally a
*token* (a critical op that participates in L3 order alignment), and a few
attributes. Tokens have the form ``kind`` or ``kind:detail``; two tokens with
the same kind but different detail align as a substitution (e.g.
``wgmma.wait:1`` vs ``wgmma.wait:0``) rather than as unrelated ops.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parse import Instr

SHAPE_RE = re.compile(r"m\d+n\d+k\d+")
_DIMS_RE = re.compile(r"\d+d")
_VEC_RE = re.compile(r"v(\d+)")
_BITS_RE = re.compile(r"[bufs](\d+)")

FLOAT_TYPES = {
    "f16", "f16x2", "bf16", "bf16x2", "f32", "f32x2", "f64", "tf32",
    "e4m3", "e5m2", "e4m3x2", "e5m2x2", "e2m1x2", "e2m3x2", "e3m2x2", "ue8m0x2",
}
FP_BASES = {
    "ex2", "lg2", "rcp", "div", "sqrt", "rsqrt", "tanh", "sin", "cos",
    "fma", "mad", "mul", "add", "sub", "cvt", "max", "min", "neg", "abs",
}
# FP flavors that change results, not just speed; diffs on these weigh more.
FP_SENSITIVE_BASES = {"ex2", "lg2", "rcp", "div", "sqrt", "rsqrt", "tanh", "sin", "cos", "cvt"}

MMA_FAMILIES = {"mma"}
LOAD_FAMILIES = {"tma.load", "cp.async", "bulk.copy"}


@dataclass
class Op:
    family: str
    token: str | None = None
    attrs: dict = field(default_factory=dict)

    @property
    def kind(self) -> str | None:
        return None if self.token is None else self.token.split(":", 1)[0]


def token_kind(token: str) -> str:
    return token.split(":", 1)[0]


def _int_operand(ins: Instr, i: int = 0) -> int | None:
    try:
        return int(ins.operands[i])
    except (IndexError, ValueError):
        return None


def _space(parts: list[str]) -> str:
    for p in parts:
        if p == "global":
            return "global"
        if p.startswith("shared"):
            return "shared"
        if p in ("local", "param", "const"):
            return p
    return "generic"


def classify(ins: Instr) -> Op | None:
    p = ins.parts
    b = p[0]
    sub = p[1] if len(p) > 1 else ""

    if b == "wgmma":
        if sub == "mma_async":
            shape = next((x for x in p if SHAPE_RE.fullmatch(x)), "?")
            dtypes = ".".join(p[p.index(shape) + 1 :]) if shape in p else ""
            a_src = "rs" if len(ins.operands) > 1 and ins.operands[1].startswith("{") else "ss"
            detail = shape + (".rs" if a_src == "rs" else "")
            return Op("mma", f"wgmma:{detail}", {
                "kind": "wgmma", "shape": shape, "dtypes": dtypes, "a_src": a_src, "sparse": "sp" in p,
            })
        if sub == "fence":
            return Op("wgmma.sync", "wgmma.fence")
        if sub == "commit_group":
            return Op("wgmma.sync", "wgmma.commit")
        if sub == "wait_group":
            n = _int_operand(ins)
            return Op("wgmma.sync", f"wgmma.wait:{n}", {"n": n})

    if b == "tcgen05":
        if sub == "mma":
            cta = next((x for x in p if x.startswith("cta_group::")), "cta_group::1")
            kind = next((x for x in p if x.startswith("kind::")), "kind::?")
            ws = ".ws" if "ws" in p else ""
            return Op("mma", f"tcgen05.mma:{cta.split('::')[1]}.{kind.split('::')[1]}{ws}", {
                "kind": "tcgen05", "cta_group": cta, "mma_kind": kind, "shape": None,
            })
        if sub == "commit":
            return Op("tcgen05.sync", "tcgen05.commit")
        if sub in ("alloc", "dealloc", "relinquish_alloc_permit"):
            return Op("tcgen05.alloc", f"tcgen05.{sub}")
        if sub in ("ld", "st", "cp", "shift"):
            return Op(f"tcgen05.{sub}", f"tcgen05.{sub}")
        if sub in ("wait", "fence"):
            return Op("tcgen05.sync", f"tcgen05.{sub}:{p[2] if len(p) > 2 else ''}")

    if b == "mma":
        shape = next((x for x in p if SHAPE_RE.fullmatch(x)), "?")
        return Op("mma", f"mma.sync:{shape}", {"kind": "mma.sync", "shape": shape,
                                              "dtypes": ".".join(p[p.index(shape) + 1 :]) if shape in p else ""})

    if b == "cp" and sub == "async":
        if len(p) > 2 and p[2] == "bulk":
            if "tensor" in p:
                dims = next((x for x in p if _DIMS_RE.fullmatch(x)), "?d")
                if "prefetch" in p:
                    return Op("tma.prefetch", None, {"dims": dims})
                i = p.index(dims)
                dst = p[i + 1] if i + 1 < len(p) else ""
                direction = "load" if dst.startswith("shared") else "store"
                mc = "multicast::cluster" in p
                cta_group = next((x for x in p if x.startswith("cta_group::")), None)
                detail = dims + (".mc" if mc else "")
                return Op(f"tma.{direction}", f"tma.{direction}:{detail}", {
                    "dims": dims, "multicast": mc, "cache_hint": "L2::cache_hint" in p,
                    "mode": "im2col" if "im2col" in p else "tile", "cta_group": cta_group,
                })
            if "commit_group" in p:
                return Op("bulk.sync", "bulk.commit")
            if "wait_group" in p:
                n = _int_operand(ins)
                read = "read" in p
                return Op("bulk.sync", f"bulk.wait:{n}{'.read' if read else ''}", {"n": n, "read": read})
            if "prefetch" in p:
                return Op("bulk.prefetch", None)
            return Op("bulk.copy", "bulk.copy")
        if "commit_group" in p:
            return Op("cp.async.sync", "cp.async.commit")
        if "wait_group" in p or "wait_all" in p:
            n = _int_operand(ins) if "wait_group" in p else 0
            return Op("cp.async.sync", f"cp.async.wait:{n}", {"n": n})
        if "mbarrier" in p:
            return Op("cp.async.sync", "cp.async.mbarrier_arrive")
        return Op("cp.async", "cp.async")

    if b == "cp" and sub == "reduce" and "tensor" in p:
        return Op("tma.reduce", "tma.reduce")

    if b == "mbarrier":
        if sub == "init":
            return Op("mbarrier.init", "mbar.init")
        if sub in ("try_wait", "test_wait"):
            return Op("mbarrier.wait", "mbar.wait", {"parity": "parity" in p})
        if sub in ("arrive", "arrive_drop"):
            if "expect_tx" in p:
                return Op("mbarrier.arrive", "mbar.arrive_tx")
            remote = "shared::cluster" in p
            return Op("mbarrier.arrive", "mbar.arrive:remote" if remote else "mbar.arrive")
        if sub == "expect_tx":
            return Op("mbarrier.arrive", "mbar.expect_tx")
        return Op(f"mbarrier.{sub}", None)

    if b == "fence":
        if "proxy" in p and "async" in p:
            return Op("fence", "fence.proxy.async")
        if "mbarrier_init" in p:
            return Op("fence", "fence.mbar_init")
        return Op("fence", None)

    if b in ("bar", "barrier"):
        if "cluster" in p:
            what = "arrive" if "arrive" in p else "wait"
            return Op("barrier.cluster", f"cluster.{what}")
        what = "arrive" if "arrive" in p else "sync"
        return Op("bar", f"bar.{what}", {"id": ins.operands[0] if ins.operands else None})

    if b == "setmaxnreg":
        n = _int_operand(ins)
        return Op("setmaxnreg", f"setmaxnreg:{sub}.{n}", {"dir": sub, "n": n})

    if b == "elect":
        return Op("elect")

    if b in ("ldmatrix", "stmatrix"):
        num = next((x for x in p if x in ("x1", "x2", "x4")), "x?")
        return Op(b, None, {"num": num, "trans": "trans" in p})

    if b in ("ld", "ldu", "st"):
        space = _space(p)
        vec = next((int(m.group(1)) for x in p if (m := _VEC_RE.fullmatch(x))), 1)
        elem = next((int(m.group(1)) for x in reversed(p) if (m := _BITS_RE.fullmatch(x))), 0)
        fam = f"{'st' if b == 'st' else 'ld'}.{space}"
        return Op(fam, None, {"vec": vec, "bits": vec * elem})

    if b in ("atom", "red"):
        return Op(f"{b}.{_space(p)}")

    if b == "griddepcontrol":
        return Op("pdl", f"pdl:{sub}")

    return None


def fp_flavor_key(ins: Instr) -> str | None:
    """Opcode key for floating-point ops (L4); None for integer/other ops."""
    p = ins.parts
    if p[0] not in FP_BASES:
        return None
    if p[0] == "cvt":
        return ins.opcode if any(x in FLOAT_TYPES for x in p[1:]) else None
    return ins.opcode if p[-1] in FLOAT_TYPES else None
