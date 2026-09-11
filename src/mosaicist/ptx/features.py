"""Layered kernel fingerprints (L0-L5) lifted from PTX, ptxas logs, and SASS.

L0 skeleton    launch geometry, warpgroups, cluster, smem, register directives
L1 inventory   tensor-core / data-movement instruction families and signatures
L2 structure   loops, warp roles, pipeline depth, persistence, epilogue form
L3 order       per-loop critical-op token sequences (aligned in diff.py)
L4 FP flavor   rounding / approximation / contraction choices
L5 machine     registers and spills from ptxas, spill instructions from SASS

Instruction counts that feed comparisons are taken per loop body and per
pipeline stage, because whole-kernel counts mostly measure unrolling.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from .cfg import CFG, Loop, build_cfg
from .ops import LOAD_FAMILIES, classify, fp_flavor_key
from .parse import Instr, Module, parse


@dataclass
class LoopFP:
    id: int
    depth: int
    role: str  # mma | mma+load | load | store | outer | other | kernel
    header_line: int
    parent: int | None
    stages: int  # pipeline stages issued per iteration (commit groups, else waits)
    counts: dict[str, int]
    mma: dict[str, int]  # mma token -> count
    tokens: list[str]
    token_lines: list[int]
    token_locs: list[str | None]

    def per_stage(self, family: str) -> float:
        return self.counts.get(family, 0) / max(1, self.stages)

    def mma_per_stage(self) -> float:
        return self.per_stage("mma")


@dataclass
class Fingerprint:
    entry: str
    target: str | None
    ptx_version: str | None
    skeleton: dict  # L0
    inventory: dict[str, int]  # L1: family -> count (whole kernel)
    mma_signatures: dict[str, int]  # L1
    tma_signatures: dict[str, int]  # L1
    global_access_bits: dict[str, dict[str, int]]  # L1: {"ld": {bits: n}, "st": {...}}
    structure: dict  # L2
    loops: list[LoopFP]  # L2/L3
    fp_flavor: dict[str, int]  # L4
    machine: dict = field(default_factory=dict)  # L5

    def to_dict(self) -> dict:
        return asdict(self)

    def loop(self, role: str) -> list[LoopFP]:
        return [lp for lp in self.loops if lp.role == role]

    def mma_loops(self) -> list[LoopFP]:
        return [lp for lp in self.loops if lp.role in ("mma", "mma+load", "kernel") and lp.mma]


def _role(counts: Counter, has_structural_children: bool) -> str:
    has_mma = counts.get("mma", 0) > 0
    has_load = any(counts.get(f, 0) for f in LOAD_FAMILIES)
    has_store = counts.get("tma.store", 0) > 0 or counts.get("st.global", 0) > 0
    if has_mma and has_load:
        return "mma+load"
    if has_mma:
        return "mma"
    if has_load:
        return "load"
    if has_store:
        return "store"
    if has_structural_children:
        return "outer"
    return "other"


def _region_fp(lid: int, depth: int, parent: int | None, instrs: list[Instr], role: str | None,
               has_children: bool) -> LoopFP:
    counts: Counter = Counter()
    mma: Counter = Counter()
    tokens, lines, locs = [], [], []
    for ins in instrs:
        op = classify(ins)
        if op is None:
            continue
        counts[op.family] += 1
        if op.family == "mma":
            mma[op.token] += 1
        if op.token is not None:
            tokens.append(op.token)
            lines.append(ins.line)
            locs.append(str(ins.loc) if ins.loc else None)
    commits = sum(1 for t in tokens if t in ("wgmma.commit", "tcgen05.commit"))
    waits = sum(1 for t in tokens if t == "mbar.wait")
    stages = commits or waits or 1
    return LoopFP(
        id=lid, depth=depth, role=role or _role(counts, has_children),
        header_line=instrs[0].line if instrs else 0, parent=parent, stages=stages,
        counts=dict(counts), mma=dict(mma), tokens=tokens, token_lines=lines, token_locs=locs,
    )


def parse_ptxas_log(text: str, entry: str | None = None) -> dict:
    """Extract registers / spills / smem from `ptxas -v` output."""
    section = text
    if entry:
        idx = text.find(f"'{entry}'")
        if idx != -1:
            nxt = text.find("Compiling entry function", idx + 1)
            section = text[idx : nxt if nxt != -1 else len(text)]
    out: dict = {}
    if m := re.search(r"Used (\d+) registers", section):
        out["registers"] = int(m.group(1))
    if m := re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", section):
        out["stack_frame"], out["spill_stores"], out["spill_loads"] = map(int, m.groups())
    if m := re.search(r"(\d+) bytes smem", section):
        out["static_smem"] = int(m.group(1))
    return out


def parse_sass(text: str) -> dict:
    """Instruction and local-memory (spill) counts from a SASS listing."""
    instr_lines = [ln for ln in text.splitlines() if re.search(r"/\*[0-9a-f]{4,}\*/", ln)]
    return {
        "sass_instrs": len(instr_lines),
        "sass_stl": sum(1 for ln in instr_lines if re.search(r"\bSTL\b", ln)),
        "sass_ldl": sum(1 for ln in instr_lines if re.search(r"\bLDL\b", ln)),
    }


def fingerprint(
    ptx: str | Module,
    entry: str | None = None,
    launch: dict | None = None,
    ptxas_log: str | None = None,
    sass: str | None = None,
) -> Fingerprint:
    """Build the fingerprint of one entry in a PTX module.

    `launch` optionally carries {"grid", "block", "cluster", "dynamic_smem"}
    recorded at launch time (grid size is not in PTX).
    """
    mod = parse(ptx) if isinstance(ptx, str) else ptx
    fn = mod.entry(entry)
    cfg = build_cfg(fn)
    instrs = fn.instrs

    # ---- L1 inventory (whole kernel) ----
    inventory: Counter = Counter()
    mma_sigs: Counter = Counter()
    tma_sigs: Counter = Counter()
    gbits: dict[str, Counter] = {"ld": Counter(), "st": Counter()}
    setmaxnreg: set[tuple[str, int]] = set()
    fp: Counter = Counter()
    for ins in instrs:
        op = classify(ins)
        if key := fp_flavor_key(ins):
            fp[key] += 1
        if op is None:
            continue
        inventory[op.family] += 1
        a = op.attrs
        if op.family == "mma":
            sig = a["kind"] + ":" + (a.get("shape") or a.get("cta_group", "")) + "." + a.get(
                "dtypes", a.get("mma_kind", "")) + ("/" + a["a_src"] if "a_src" in a else "")
            mma_sigs[sig] += 1
        elif op.family in ("tma.load", "tma.store"):
            tma_sigs[op.token] += 1
        elif op.family in ("ld.global", "st.global"):
            gbits[op.family[:2]][str(a["bits"])] += 1
        elif op.family == "setmaxnreg":
            setmaxnreg.add((a["dir"], a["n"]))

    # ---- L2/L3 loops ----
    loops: list[LoopFP] = []
    structural = cfg.structural_loops()
    for lp in structural:
        has_children = any(not cfg.loops[c].spin for c in lp.children)
        parent = _structural_parent(cfg, lp)
        loops.append(_region_fp(lp.id, lp.depth, parent, cfg.own_instrs(lp), None, has_children))
    if not any(lp.mma for lp in loops):
        # fully unrolled kernels: treat the whole kernel as one region
        loops.append(_region_fp(-1, 0, None, instrs, "kernel", bool(structural)))

    by_id = {lp.id: lp for lp in loops}

    def ancestors(lp: LoopFP):
        p = lp.parent
        while p is not None and p in by_id:
            yield by_id[p]
            p = by_id[p].parent

    mma_loops = [lp for lp in loops if lp.role in ("mma", "mma+load")]
    load_loops = [lp for lp in loops if lp.role == "load"]
    warp_specialized = any(
        ld.id != mm.id and mm not in ancestors(ld) and ld not in ancestors(mm)
        for ld in load_loops for mm in mma_loops
    )
    persistent = any(any(True for _ in ancestors(lp)) for lp in mma_loops)
    wait_depths = sorted({
        int(t.split(":")[1]) for lp in mma_loops for t in lp.tokens
        if t.startswith("wgmma.wait:") and t.split(":")[1].isdigit()
    })
    bulk_waits = sorted({
        t.split(":", 1)[1] for t in _kernel_tokens(instrs) if t.startswith("bulk.wait:")
    })
    epilogue = "tma_store" if inventory.get("tma.store") else (
        "st.global" if inventory.get("st.global") else "none")

    structure = {
        "n_loops": len(structural),
        "max_depth": max((lp.depth for lp in structural), default=0),
        "roles": sorted(lp.role for lp in loops if lp.role != "kernel"),
        "warp_specialized": warp_specialized,
        "persistent": persistent,
        "mma_loop_depth": max((lp.depth for lp in mma_loops), default=0),
        "mbarrier_inits": inventory.get("mbarrier.init", 0),
        "wgmma_wait_depths": wait_depths,
        "bulk_wait_depths": bulk_waits,
        "epilogue": epilogue,
    }

    # ---- L0 skeleton ----
    d = fn.directives
    block = tuple(launch["block"]) if launch and launch.get("block") else (
        d.get("reqntid") or d.get("maxntid") or None)
    threads = math.prod(block) if isinstance(block, tuple) and block else None
    cluster = tuple(launch["cluster"]) if launch and launch.get("cluster") else d.get("reqnctapercluster")
    shared = mod.shared + fn.shared
    skeleton = {
        "block": list(block) if isinstance(block, tuple) else None,
        "threads": threads,
        "warpgroups": math.ceil(threads / 128) if threads else None,
        "cluster": list(_pad3(cluster)) if isinstance(cluster, tuple) else [1, 1, 1],
        "maxnreg": d["maxnreg"][0] if isinstance(d.get("maxnreg"), tuple) else None,
        "setmaxnreg": sorted([list(x) for x in setmaxnreg]),
        "static_smem": sum(s.nbytes for s in shared if s.nbytes),
        "dynamic_smem": (launch or {}).get("dynamic_smem"),
        "uses_dynamic_smem": any(s.nbytes is None for s in shared),
        "grid": list(launch["grid"]) if launch and launch.get("grid") else None,
    }

    machine: dict = {}
    if ptxas_log:
        machine.update(parse_ptxas_log(ptxas_log, fn.name))
    if sass:
        machine.update(parse_sass(sass))

    return Fingerprint(
        entry=fn.name, target=mod.target, ptx_version=mod.version,
        skeleton=skeleton, inventory=dict(inventory), mma_signatures=dict(mma_sigs),
        tma_signatures=dict(tma_sigs), global_access_bits={k: dict(v) for k, v in gbits.items()},
        structure=structure, loops=loops, fp_flavor=dict(fp), machine=machine,
    )


def _kernel_tokens(instrs: list[Instr]):
    for ins in instrs:
        op = classify(ins)
        if op is not None and op.token is not None:
            yield op.token


def _structural_parent(cfg: CFG, lp: Loop) -> int | None:
    p = lp.parent
    while p is not None and cfg.loops[p].spin:
        p = cfg.loops[p].parent
    return p


def _pad3(t: tuple) -> tuple:
    return tuple(t) + (1,) * (3 - len(t))
