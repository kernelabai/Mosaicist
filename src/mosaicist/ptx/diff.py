"""Compare a reference fingerprint against a candidate, layer by layer.

Every comparison is a registered *check* with a weight; failed checks become
Discrepancy rows. A layer's distance is the failed weight over the checked
weight, and the overall distance D is a weighted mean over the layers that
had anything to check. D is only a tiebreaker for the convergence loop - the
rows are what Diagnose turns into edits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .align import Alignment, align
from .features import Fingerprint, LoopFP
from .ops import FP_SENSITIVE_BASES, token_kind

LAYER_WEIGHTS = {"L0": 3.0, "L1": 2.0, "L2": 2.0, "L3": 1.0, "L4": 1.0, "L5": 2.0}


@dataclass
class Discrepancy:
    layer: str
    key: str
    ref: object
    cand: object
    weight: float
    message: str
    ref_lines: list[int] = field(default_factory=list)
    cand_lines: list[int] = field(default_factory=list)
    ref_locs: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)


@dataclass
class LoopPair:
    ref_loop: int
    cand_loop: int
    ref_role: str
    cand_role: str
    alignment: Alignment
    cand_index: list[int]  # alignment position -> index into the candidate loop's tokens
    projected: bool  # candidate loop was fused; only the ref role's ops were compared


@dataclass
class DiffReport:
    discrepancies: list[Discrepancy]
    layer_distance: dict[str, float]
    distance: float
    loop_pairs: list[LoopPair]

    def keys(self) -> set[str]:
        return {d.key for d in self.discrepancies}

    def get(self, key: str) -> Discrepancy | None:
        return next((d for d in self.discrepancies if d.key == key), None)

    def to_dict(self) -> dict:
        return {
            "distance": self.distance,
            "layer_distance": self.layer_distance,
            "discrepancies": [asdict(d) for d in self.discrepancies],
            "loop_pairs": [
                {"ref_loop": p.ref_loop, "cand_loop": p.cand_loop, "ref_role": p.ref_role,
                 "cand_role": p.cand_role, "similarity": p.alignment.similarity,
                 "projected": p.projected}
                for p in self.loop_pairs
            ],
        }


class _Differ:
    def __init__(self) -> None:
        self.rows: list[Discrepancy] = []
        self.checked: dict[str, float] = {}
        self.failed: dict[str, float] = {}

    def check(self, layer: str, key: str, ref, cand, weight: float, message: str,
              differs: bool | None = None, **extra) -> None:
        self.checked[layer] = self.checked.get(layer, 0.0) + weight
        if differs is None:
            differs = ref != cand
        if differs:
            self.failed[layer] = self.failed.get(layer, 0.0) + weight
            self.rows.append(Discrepancy(layer, key, ref, cand, weight, message, **extra))


# Which candidate roles can stand in for a reference loop role.
_COMPATIBLE = {
    "mma": ("mma", "mma+load", "kernel"),
    "mma+load": ("mma+load", "mma", "kernel"),
    "kernel": ("kernel", "mma", "mma+load"),
    "load": ("load", "mma+load"),
    "store": ("store",),
    "outer": ("outer",),
}


_PRODUCER_KINDS = {"mbar.arrive_tx", "mbar.expect_tx", "tma.load", "cp.async", "cp.async.commit",
                   "cp.async.wait", "bulk.copy", "tma.prefetch"}
_SHARED_KINDS = {"mbar.wait"}  # a fused loop's single wait serves both roles


def _project(cl: LoopFP, ref_role: str) -> tuple[list[str], list[int], bool]:
    """Tokens of a fused (mma+load) candidate loop that belong to `ref_role`."""
    if cl.role != "mma+load" or ref_role not in ("mma", "load"):
        return cl.tokens, list(range(len(cl.tokens))), False
    keep = []
    for i, t in enumerate(cl.tokens):
        kind = token_kind(t)
        is_producer = kind in _PRODUCER_KINDS
        if kind in _SHARED_KINDS or (is_producer if ref_role == "load" else not is_producer):
            keep.append(i)
    return [cl.tokens[i] for i in keep], keep, True


def _pair_loops(ref: Fingerprint, cand: Fingerprint) -> list[LoopPair]:
    pairs = []
    for rl in ref.loops:
        options = [cl for cl in cand.loops if cl.role in _COMPATIBLE.get(rl.role, ())]
        if not options or not rl.tokens:
            continue
        scored = []
        for cl in options:
            toks, idx, projected = _project(cl, rl.role)
            scored.append((cl, align(rl.tokens, toks), idx, projected))
        cl, aln, idx, projected = max(scored, key=lambda x: (x[1].similarity, -abs(x[0].depth - rl.depth)))
        pairs.append(LoopPair(rl.id, cl.id, rl.role, cl.role, aln, idx, projected))
    return pairs


def diff(ref: Fingerprint, cand: Fingerprint) -> DiffReport:
    d = _Differ()
    rs, cs = ref.skeleton, cand.skeleton

    # ---- L0 skeleton ----
    if rs["warpgroups"] and cs["warpgroups"]:
        d.check("L0", "L0.warpgroups", rs["warpgroups"], cs["warpgroups"], 3.0,
                f"warpgroups per CTA: reference {rs['warpgroups']}, candidate {cs['warpgroups']}")
    d.check("L0", "L0.cluster", rs["cluster"], cs["cluster"], 2.0,
            f"cluster shape: reference {rs['cluster']}, candidate {cs['cluster']}")
    d.check("L0", "L0.setmaxnreg", rs["setmaxnreg"], cs["setmaxnreg"], 2.0,
            f"register reallocation (setmaxnreg): reference {rs['setmaxnreg'] or 'none'}, "
            f"candidate {cs['setmaxnreg'] or 'none'}")
    if rs["maxnreg"] or cs["maxnreg"]:
        d.check("L0", "L0.maxnreg", rs["maxnreg"], cs["maxnreg"], 1.0,
                f".maxnreg: reference {rs['maxnreg']}, candidate {cs['maxnreg']}")
    r_smem = (rs["static_smem"] or 0) + (rs["dynamic_smem"] or 0)
    c_smem = (cs["static_smem"] or 0) + (cs["dynamic_smem"] or 0)
    if r_smem and c_smem:
        d.check("L0", "L0.smem", r_smem, c_smem, 1.0,
                f"shared memory bytes: reference {r_smem}, candidate {c_smem}",
                differs=abs(r_smem - c_smem) > 0.25 * r_smem)
    if rs["grid"] and cs["grid"]:
        d.check("L0", "L0.grid", rs["grid"], cs["grid"], 2.0,
                f"grid: reference {rs['grid']}, candidate {cs['grid']}")

    # ---- L1 inventory ----
    r_mma, c_mma = set(ref.mma_signatures), set(cand.mma_signatures)
    if r_mma or c_mma:
        d.check("L1", "L1.mma_shape", sorted(r_mma), sorted(c_mma), 2.0,
                f"MMA instructions: reference {sorted(r_mma)}, candidate {sorted(c_mma)}")
    r_tma = ref.inventory.get("tma.load", 0) > 0
    c_tma = cand.inventory.get("tma.load", 0) > 0
    d.check("L1", "L1.tma_load", r_tma, c_tma, 3.0,
            "reference stages operands with TMA; candidate does not" if r_tma else
            "candidate uses TMA loads; reference does not")
    if r_tma and c_tma:
        r_mc = any(":" in t and t.split(":")[1].endswith(".mc") for t in ref.tma_signatures if t.startswith("tma.load"))
        c_mc = any(":" in t and t.split(":")[1].endswith(".mc") for t in cand.tma_signatures if t.startswith("tma.load"))
        d.check("L1", "L1.tma_multicast", r_mc, c_mc, 2.0,
                "reference multicasts TMA loads across the cluster; candidate does not" if r_mc else
                "candidate multicasts TMA loads; reference does not")
        r_dims = sorted({t.split(":")[1].split(".")[0] for t in ref.tma_signatures if t.startswith("tma.load")})
        c_dims = sorted({t.split(":")[1].split(".")[0] for t in cand.tma_signatures if t.startswith("tma.load")})
        d.check("L1", "L1.tma_dims", r_dims, c_dims, 1.0,
                f"TMA load dimensionality: reference {r_dims}, candidate {c_dims}")
    d.check("L1", "L1.epilogue_store", ref.structure["epilogue"], cand.structure["epilogue"], 2.0,
            f"epilogue stores: reference {ref.structure['epilogue']}, candidate {cand.structure['epilogue']}")
    r_st = max((int(b) for b in ref.global_access_bits["st"]), default=0)
    c_st = max((int(b) for b in cand.global_access_bits["st"]), default=0)
    if r_st and c_st:
        d.check("L1", "L1.st_global_width", r_st, c_st, 1.0,
                f"widest st.global: reference {r_st} bits, candidate {c_st} bits")
    for fam in ("ldmatrix", "stmatrix"):
        r_has, c_has = ref.inventory.get(fam, 0) > 0, cand.inventory.get(fam, 0) > 0
        if r_has or c_has:
            d.check("L1", f"L1.{fam}", r_has, c_has, 1.0,
                    f"{fam}: reference {'uses' if r_has else 'does not use'} it, "
                    f"candidate {'uses' if c_has else 'does not use'} it")

    pairs = _pair_loops(ref, cand)
    loops_r = {lp.id: lp for lp in ref.loops}
    loops_c = {lp.id: lp for lp in cand.loops}
    for p in pairs:
        rl, cl = loops_r[p.ref_loop], loops_c[p.cand_loop]
        if rl.mma and cl.mma:
            d.check("L1", f"L1.mma_per_stage[{rl.id}]", rl.mma_per_stage(), cl.mma_per_stage(), 2.0,
                    f"MMA instructions per pipeline stage in the {rl.role} loop: "
                    f"reference {rl.mma_per_stage():g}, candidate {cl.mma_per_stage():g}",
                    ref_lines=[rl.header_line], cand_lines=[cl.header_line])

    # ---- L2 structure ----
    rst, cst = ref.structure, cand.structure
    d.check("L2", "L2.warp_specialized", rst["warp_specialized"], cst["warp_specialized"], 3.0,
            "reference splits producer and consumer warp roles; candidate does not"
            if rst["warp_specialized"] else "candidate is warp-specialized; reference is not")
    d.check("L2", "L2.persistent", rst["persistent"], cst["persistent"], 2.0,
            "reference runs a persistent tile loop around its mainloop; candidate does not"
            if rst["persistent"] else "candidate is persistent; reference is not")
    if rst["wgmma_wait_depths"] or cst["wgmma_wait_depths"]:
        r_w = max(rst["wgmma_wait_depths"], default=None)
        c_w = max(cst["wgmma_wait_depths"], default=None)
        d.check("L2", "L2.mma_wait_depth", r_w, c_w, 3.0,
                f"wgmma groups left in flight in the mainloop: reference {r_w}, candidate {c_w}")
    d.check("L2", "L2.pipeline_barriers", rst["mbarrier_inits"], cst["mbarrier_inits"], 1.0,
            f"mbarrier.init count (pipeline depth proxy): reference {rst['mbarrier_inits']}, "
            f"candidate {cst['mbarrier_inits']}")
    if rst["bulk_wait_depths"] or cst["bulk_wait_depths"]:
        d.check("L2", "L2.store_wait_depth", rst["bulk_wait_depths"], cst["bulk_wait_depths"], 1.0,
                f"TMA-store wait depths: reference {rst['bulk_wait_depths']}, candidate {cst['bulk_wait_depths']}")

    # ---- L3 order ----
    sims = []
    for p in pairs:
        rl, cl = loops_r[p.ref_loop], loops_c[p.cand_loop]
        sims.append(p.alignment.similarity)
        diffs = p.alignment.differences()
        if not diffs:
            continue
        summary = _summarize(diffs, rl, cl, p.cand_index)
        scope = f" ({rl.role}-side ops of the fused loop)" if p.projected else ""
        d.rows.append(Discrepancy(
            "L3", f"L3.order[{rl.id}]", rl.tokens, [cl.tokens[i] for i in p.cand_index],
            round(2.0 * (1 - p.alignment.similarity), 3),
            f"{rl.role} loop (ref line {rl.header_line}) vs candidate {cl.role} loop "
            f"(line {cl.header_line}){scope}, similarity {p.alignment.similarity:.2f}: {summary}",
            ref_lines=[rl.token_lines[s.ref] for s in diffs if s.ref is not None],
            cand_lines=[cl.token_lines[p.cand_index[s.cand]] for s in diffs if s.cand is not None],
            ref_locs=[rl.token_locs[s.ref] for s in diffs if s.ref is not None and rl.token_locs[s.ref]],
            context={"ref_role": rl.role, "cand_role": cl.role, "projected": p.projected},
        ))

    # ---- L4 FP flavor ----
    r_fp, c_fp = set(ref.fp_flavor), set(cand.fp_flavor)
    bases = sorted({k.split(".")[0] for k in r_fp | c_fp})
    for base in bases:
        rk = sorted(k for k in r_fp if k.split(".")[0] == base)
        ck = sorted(k for k in c_fp if k.split(".")[0] == base)
        if base in ("fma", "mul", "add", "sub", "mad"):
            continue  # judged together as contraction below
        w = 2.0 if base in FP_SENSITIVE_BASES else 0.5
        d.check("L4", f"L4.{base}", rk, ck, w, f"{base} flavors: reference {rk or 'none'}, candidate {ck or 'none'}",
                differs=bool(rk) != bool(ck) or (rk and ck and rk != ck))
    r_c, c_c = _contraction(ref.fp_flavor), _contraction(cand.fp_flavor)
    if r_c is not None and c_c is not None:
        d.check("L4", "L4.contraction", round(r_c, 2), round(c_c, 2), 1.0,
                f"fraction of f32 mul/add fused into fma: reference {r_c:.2f}, candidate {c_c:.2f}",
                differs=abs(r_c - c_c) > 0.2)

    # ---- L5 machine ----
    rm, cm = ref.machine, cand.machine
    c_spill = cm.get("spill_stores", 0) + cm.get("sass_stl", 0)
    r_spill = rm.get("spill_stores", 0) + rm.get("sass_stl", 0)
    if cm or rm:
        d.check("L5", "L5.spills", r_spill if rm else None, c_spill, 3.0,
                f"register spills: reference {r_spill if rm else 'unknown'} "
                f"(bytes/STL), candidate {c_spill}",
                differs=(c_spill > 0 and r_spill == 0) or (r_spill > 0 and c_spill == 0 and bool(cm)))
    if rm.get("registers") and cm.get("registers"):
        d.check("L5", "L5.registers", rm["registers"], cm["registers"], 1.0,
                f"registers/thread: reference {rm['registers']}, candidate {cm['registers']}",
                differs=abs(rm["registers"] - cm["registers"]) > 0.1 * rm["registers"])

    layer_distance = {
        layer: min(1.0, d.failed.get(layer, 0.0) / w) for layer, w in d.checked.items() if w > 0
    }
    if sims:
        layer_distance["L3"] = 1.0 - sum(sims) / len(sims)
    total_w = sum(LAYER_WEIGHTS[k] for k in layer_distance)
    distance = sum(LAYER_WEIGHTS[k] * v for k, v in layer_distance.items()) / total_w if total_w else 0.0
    return DiffReport(d.rows, dict(sorted(layer_distance.items())), distance, pairs)


def _contraction(fp: dict[str, int]) -> float | None:
    f32 = lambda base: sum(v for k, v in fp.items() if k.startswith(base + ".") and k.endswith(".f32"))
    fma, mul, add = f32("fma"), f32("mul"), f32("add") + f32("sub")
    total = fma + mul + add
    return None if total == 0 else fma / (fma + (mul + add) / 2)


def _summarize(diffs, rl: LoopFP, cl: LoopFP, cand_index: list[int]) -> str:
    parts = []
    for s in diffs:
        if s.op == "subst":
            parts.append(f"{rl.tokens[s.ref]} -> {cl.tokens[cand_index[s.cand]]}")
        elif s.op == "del":
            parts.append(f"missing {rl.tokens[s.ref]}")
        else:
            parts.append(f"extra {cl.tokens[cand_index[s.cand]]}")
    # collapse runs of identical descriptions
    out, prev, n = [], None, 0
    for item in parts + [None]:
        if item == prev:
            n += 1
            continue
        if prev is not None:
            out.append(f"{prev} x{n}" if n > 1 else prev)
        prev, n = item, 1
    return "; ".join(out)
