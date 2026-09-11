"""Turn fingerprint discrepancy rows into ranked, tagged fixes.

Each fix names the Pallas Mosaic GPU lever that should close it and is tagged
by how it gets applied:

  knob         a parameter change the tuner can make deterministically
  rewrite      a structural source change for the LLM rewriter
  low-level    needs the underlying jax.experimental.mosaic_gpu API
  gap          no known Pallas spelling; goes to the residual report
  investigate  no rule yet; a human (or the rewriter) should look

Fixes are ordered by convergence phase (P1 skeleton -> P5 micro), then by the
weight of the rows they close, because fine-level rows are noise until the
coarse ones match.

The discrepancy rows are compiler-agnostic, but the *fix text* names Pallas
Mosaic GPU levers, since that is the intended candidate. `mosaicist diff` is
useful with any candidate (a CuTeDSL port, a C++ kernel), and then the rows
still apply while the suggested levers do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ptx.diff import Discrepancy, DiffReport

PHASES = {
    "P1": "skeleton",
    "P2": "data movement",
    "P3": "tensor core",
    "P4": "concurrency",
    "P5": "micro",
}


@dataclass
class Fix:
    phase: str
    tag: str
    title: str
    detail: str
    keys: list[str]
    weight: float
    layers: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"[{self.phase} {self.tag}] {self.title}"


class _Rows:
    def __init__(self, report: DiffReport):
        self.by_key = {d.key: d for d in report.discrepancies}
        self.used: set[str] = set()

    def get(self, key: str) -> Discrepancy | None:
        return None if key in self.used else self.by_key.get(key)

    def prefixed(self, prefix: str) -> list[Discrepancy]:
        return [d for k, d in self.by_key.items() if k.startswith(prefix) and k not in self.used]

    def take(self, rows: list[Discrepancy | None]) -> tuple[list[str], float, list[str]]:
        rows = [r for r in rows if r is not None]
        for r in rows:
            self.used.add(r.key)
        layers = sorted({r.layer for r in rows})
        return [r.key for r in rows], sum(r.weight for r in rows), layers


def _warp_structure(rows: _Rows) -> Fix | None:
    wg = rows.get("L0.warpgroups")
    ws = rows.get("L2.warp_specialized")
    if wg is None and ws is None:
        return None
    ctx = wg.context if wg is not None else {}
    ref_ws = bool(ws.ref) if ws is not None else ctx.get("ref_ws")
    cand_ws = bool(ws.cand) if ws is not None else ctx.get("cand_ws")
    rc, cc = ctx.get("ref_compute"), ctx.get("cand_compute")
    spills = rows.get("L5.spills")
    per_stage = rows.prefixed("L1.mma_per_stage")

    if cand_ws and not ref_ws and (rc is None or rc == cc):
        # Same compute warpgroups; the candidate adds a producer. Not obviously a regression.
        regs = rows.get("L0.setmaxnreg")
        keys, w, layers = rows.take([wg, ws, regs])
        n = f"{cc} " if cc else ""
        return Fix(
            "P1", "investigate",
            "Candidate adds a producer warpgroup; reference issues TMA from its compute warps",
            f"Both run {n}compute warpgroup(s). The candidate is warp-specialized (a memory warpgroup, "
            "with register reallocation) while the reference prefetches stages and issues the next TMA "
            "from inside its MMA loop. To mirror the reference, drop the memory warpgroup: use "
            f"plgpu.emit_pipeline with num_threads={rc or cc} instead of emit_pipeline_warp_specialized. "
            "Keep the warp-specialized structure if it measures faster; runtime decides.",
            keys, w, layers)
    if rc is not None and cc is not None and cc > rc:
        keys, w, layers = rows.take([wg])
        return Fix("P1", "knob", f"Candidate runs {cc} compute warpgroups per CTA; reference runs {rc}",
                   "Reduce num_compute_wgs (or num_threads) to the reference's compute warpgroup count.",
                   keys, w, layers)
    total = wg.ref if wg is not None else None
    producer = 1 if ref_ws else 0
    compute = (total - producer) if total else None
    if wg is not None:
        title = f"{wg.cand} warpgroup{'s' if wg.cand != 1 else ''} per CTA; reference runs {wg.ref}"
        if producer:
            title += f" (1 producer + {compute} compute)"
    else:
        title = "Reference splits producer and consumer warp roles"
    if spills is not None:
        title += "; candidate spills"
    detail = []
    if spills is not None:
        detail.append("The candidate spills registers where the reference does not: per-thread "
                      "accumulator pressure the reference avoids by splitting M across warpgroups.")
    if compute:
        detail.append(
            f"Run {compute} compute warpgroup(s)" + (" plus one memory warpgroup" if producer else "") +
            f": plgpu.kernel(..., num_threads={total}, thread_name=\"wg\") with "
            f"plgpu.emit_pipeline_warp_specialized(num_compute_wgs={compute}, wg_axis=\"wg\", ...)."
        )
    else:
        detail.append("Use plgpu.emit_pipeline_warp_specialized so a memory warpgroup issues TMA "
                      "while compute warpgroups run the MMA loop.")
    if per_stage:
        detail.append("This also explains the per-stage MMA count difference: one warpgroup is "
                      "issuing the work of several.")
    keys, w, layers = rows.take([wg, ws, spills, *per_stage])
    return Fix("P1", "rewrite", title, " ".join(detail), keys, w, layers)


def _spills(rows: _Rows) -> Fix | None:
    r = rows.get("L5.spills")
    if r is None or not r.cand:
        return None
    keys, w, layers = rows.take([r])
    return Fix("P1", "rewrite", f"Candidate spills registers ({r.cand}); reference {r.ref}",
               "Reduce live per-thread state: split the tile across more compute warpgroups, shrink "
               "the tile, or raise the compute warpgroups' register budget with plgpu.set_max_registers.",
               keys, w, layers)


def _cluster_multicast(rows: _Rows) -> Fix | None:
    cl = rows.get("L0.cluster")
    mc = rows.get("L1.tma_multicast")
    if cl is None and mc is None:
        return None
    parts = []
    if cl is not None:
        parts.append(f"Launch with cluster={tuple(cl.ref)} via plgpu.kernel(cluster=..., cluster_names=...).")
    if mc is not None and mc.ref:
        parts.append("Multicast the operand shared across the cluster: pass collective_axes=<cluster axis> "
                     "to plgpu.copy_gmem_to_smem for it (fall back to a manual plgpu.Barrier ring if the "
                     "pipeline helper cannot express the collective copy).")
    tag = "rewrite" if mc is not None else "knob"
    title = "Cluster shape / TMA multicast differ" if mc is not None else "Cluster shape differs"
    keys, w, layers = rows.take([cl, mc])
    return Fix("P2", tag, title, " ".join(parts) or "Match the reference cluster shape.", keys, w, layers)


def _persistence(rows: _Rows) -> Fix | None:
    p = rows.get("L2.persistent")
    if p is None:
        return None
    g = rows.get("L0.grid")
    grid = f" (grid: reference {g.ref}, candidate {g.cand})" if g is not None else ""
    keys, w, layers = rows.take([p, g])
    if p.ref:
        return Fix("P4", "rewrite", "Reference is persistent; candidate is not" + grid,
                   "Launch one CTA (or cluster) per SM and loop over output tiles with plgpu.nd_loop "
                   "(or plgpu.dynamic_scheduling_loop on sm_100a), overlapping each tile's epilogue "
                   "with the next tile's mainloop.", keys, w, layers)
    return Fix("P4", "investigate", "Candidate is persistent; reference is not" + grid,
               "To mirror the reference, launch one CTA per output tile (a grid over tiles, no nd_loop). "
               "A persistent loop mainly pays off when it hides the epilogue or the tail wave; keep it "
               "if it measures faster.", keys, w, layers)


_ADVISORY_HINTS = {
    "C7519": ("P3", "ptxas had to insert warpgroup.arrive (wgmma fences): registers used by wgmma are "
                    "touched by other code between MMA issues. Keep the accumulator in the WGMMA layout "
                    "through the loop and avoid reading it before plgpu.wgmma_wait."),
    "C7510": ("P3", "ptxas serialized wgmma.mma_async: the MMA pipeline cannot overlap. Usually caused by "
                    "accumulator registers being accessed between wgmma issues or by register pressure."),
}


def _advisories(rows: _Rows) -> list[Fix]:
    fixes = []
    for r in rows.prefixed("L5.ptxas_"):
        code = r.context.get("code", r.key.rsplit("_", 1)[-1])
        phase, hint = _ADVISORY_HINTS.get(code, ("P5", "See the ptxas message."))
        side = "candidate" if r.cand else "reference"
        keys, w, layers = rows.take([r])
        fixes.append(Fix(phase, "investigate", f"ptxas advisory {code} only in the {side}",
                         f"{hint} ({r.message})", keys, w, layers))
    return fixes


def _simple(key: str, phase: str, tag: str, title, detail) -> callable:
    def rule(rows: _Rows) -> Fix | None:
        r = rows.get(key)
        if r is None:
            return None
        keys, w, layers = rows.take([r])
        t = title(r) if callable(title) else title
        dt = detail(r) if callable(detail) else detail
        return Fix(phase, tag, t, dt, keys, w, layers)
    return rule


def _stages_detail(r: Discrepancy) -> str:
    return (f"Reference initializes {r.ref} mbarriers, candidate {r.cand}. With a full/empty barrier "
            f"pair per stage that suggests ~{max(1, r.ref // 2)} reference stages; set "
            f"max_concurrent_steps on the pipeline to match, then let the tuner search around it.")


def _wait_detail(r: Discrepancy) -> str:
    n = r.ref if r.ref is not None else 1
    return (f"Keep {n} wgmma group(s) in flight: plgpu.wgmma_wait({n}) in the pipeline body with "
            f"delay_release={n}, so the buffer released each step is the one {n} step(s) back.")


def _setmaxnreg_detail(r: Discrepancy) -> str:
    def split(pairs):
        pairs = pairs or []
        return (next((n for d, n in pairs if d == "dec"), None), next((n for d, n in pairs if d == "inc"), None))

    (r_dec, r_inc), (c_dec, c_inc) = split(r.ref), split(r.cand)
    if r.ref:
        s = f"Reference rebalances registers (producer {r_dec}, compute {r_inc})"
        s += f"; candidate {c_dec}/{c_inc}. " if r.cand else "; candidate does not. "
        if r_dec is not None:
            s += f"Set memory_registers={r_dec} on plgpu.emit_pipeline_warp_specialized, or "
        return s + "call plgpu.set_max_registers per warp role."
    return (f"Candidate rebalances registers (producer {c_dec}, compute {c_inc}) where the reference does "
            "not. This comes with its producer warpgroup; it goes away if that structure does.")


_SIMPLE_RULES = [
    _simple("L1.tma_load", "P2", "rewrite", "TMA staging differs",
            lambda r: ("Stage operands through TMA: plgpu.copy_gmem_to_smem, or BlockSpecs under "
                       "plgpu.emit_pipeline, instead of direct global loads.") if r.ref else
            "Candidate uses TMA where the reference does not; check whether the reference's LSU path is deliberate."),
    _simple("L2.pipeline_barriers", "P2", "knob", "Pipeline depth differs", _stages_detail),
    _simple("L1.tma_dims", "P2", "investigate", "TMA box rank differs",
            lambda r: (f"Reference TMA ranks {r.ref}, candidate {r.cand}. Tiling transforms raise the TMA rank "
                       "in Mosaic GPU; this is usually benign unless box sizes also differ.")),
    _simple("L1.mma_shape", "P3", "knob", "MMA instruction differs",
            lambda r: (f"Reference issues {r.ref}, candidate {r.cand}. Match the tile N and operand dtypes so "
                       "Mosaic GPU selects the same wgmma/tcgen05 shape; '/rs' means A comes from registers.")),
    _simple("L2.mma_wait_depth", "P3", "knob", lambda r: f"MMA drained every step (wait {r.cand}; reference {r.ref})",
            _wait_detail),
    _simple("L0.setmaxnreg", "P4", "knob", "Register rebalancing differs", _setmaxnreg_detail),
    _simple("L1.epilogue_store", "P4", "rewrite", "Epilogue store path differs",
            lambda r: (f"Reference epilogue: {r.ref}; candidate: {r.cand}. For TMA stores: plgpu.commit_smem(); "
                       "plgpu.copy_smem_to_gmem(...); plgpu.wait_smem_to_gmem(n), overlapping the next tile.")),
    _simple("L2.store_wait_depth", "P4", "knob", "TMA-store wait depth differs",
            lambda r: f"Reference waits with {r.ref}, candidate {r.cand}: adjust plgpu.wait_smem_to_gmem(n)."),
    _simple("L0.grid", "P1", "knob", "Grid differs",
            lambda r: f"Reference grid {r.ref}, candidate {r.cand}: check tile shape and grid mapping."),
    _simple("L0.smem", "P1", "knob", "Shared-memory footprint differs",
            lambda r: f"Reference {r.ref} B, candidate {r.cand} B: stage count or tile shape differs."),
    _simple("L0.maxnreg", "P1", "knob", ".maxnreg differs",
            lambda r: f"Reference {r.ref}, candidate {r.cand}."),
    _simple("L1.stmatrix", "P5", "rewrite", "stmatrix use differs",
            "Reference writes accumulator fragments to smem with stmatrix; check the candidate's epilogue "
            "layout (plgpu.Layout.WGMMA -> smem store) and its swizzle."),
    _simple("L1.ldmatrix", "P5", "rewrite", "ldmatrix use differs",
            "Operand fragments are loaded differently; check register-sourced MMA operands and layouts."),
    _simple("L1.st_global_width", "P5", "rewrite", "Global store vector width differs",
            lambda r: f"Widest st.global: reference {r.ref} bits, candidate {r.cand} bits; check epilogue layout."),
    _simple("L4.contraction", "P5", "rewrite", "FMA contraction differs",
            lambda r: (f"Fused fraction reference {r.ref}, candidate {r.cand}; match the epilogue's op order "
                       "(e.g. alpha*acc + beta*c as one fma).")),
    _simple("L5.registers", "P5", "investigate", "Register count differs",
            lambda r: f"Reference {r.ref}/thread, candidate {r.cand}/thread."),
]


def diagnose(report: DiffReport) -> list[Fix]:
    rows = _Rows(report)
    fixes: list[Fix] = []
    for rule in (_warp_structure, _spills, _cluster_multicast, _persistence, *_SIMPLE_RULES):
        if (fix := rule(rows)) is not None:
            fixes.append(fix)
    fixes.extend(_advisories(rows))
    for r in rows.prefixed("L4."):
        keys, w, layers = rows.take([r])
        fixes.append(Fix("P5", "rewrite", f"FP flavor differs: {r.key[3:]}", r.message, keys, w, layers))
    for r in rows.prefixed("L3.order"):
        role = r.context.get("ref_role", "")
        where = {"mma": "Mainloop", "mma+load": "Mainloop", "kernel": "Kernel", "load": "Producer loop",
                 "store": "Epilogue loop"}.get(role, f"{role.capitalize()} loop")
        keys, w, layers = rows.take([r])
        fixes.append(Fix("P5", "investigate", f"{where} op order differs", r.message, keys, w, layers))
    for key in sorted(set(rows.by_key) - rows.used):
        r = rows.by_key[key]
        keys, w, layers = rows.take([r])
        fixes.append(Fix("P5", "investigate", key, r.message, keys, w, layers))
    fixes.sort(key=lambda f: (f.phase, -f.weight))
    return fixes
