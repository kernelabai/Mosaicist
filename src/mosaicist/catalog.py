"""The construct catalog: CuTeDSL concept -> Pallas Mosaic GPU construct.

The machine-readable form of DESIGN §5's table, shared by the translator (which picks
constructs to emit) and Diagnose (which turns a PTX diff row back into a source edit).
Each entry carries the fingerprint signature the construct produces, which is what lets
a diff row name a fix instead of just a difference.

`tag` says how far the toolchain can go:
  ``core``      expressible in the documented Pallas API
  ``low-level`` needs jax.experimental.mosaic_gpu directly
  ``gap``       no analogue; goes in the report
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Construct:
    concept: str
    cutedsl: str
    pallas: str
    #: fingerprint keys this construct moves, e.g. "L0.block", "L1.tma"
    signature: tuple[str, ...] = ()
    tag: str = "core"
    #: knob names a tuner may vary, if this construct is parameterized
    knobs: tuple[str, ...] = ()


CONSTRUCTS: tuple[Construct, ...] = (
    Construct("launch", "@cute.kernel + .launch(grid, block, cluster, smem)",
              "plgpu.kernel(body, out_shape=, grid=, grid_names=, cluster=, num_threads=)",
              ("L0.grid", "L0.block", "L0.cluster"), knobs=("num_threads", "cluster")),
    Construct("tile gmem", "cute.local_tile / zipped_divide",
              "ref.at[pl.ds(i * t, t)] or plgpu.BlockSpec index maps",
              ("L0.grid",), knobs=("tile_m", "tile_n")),
    Construct("smem layout", "composed layout with Swizzle",
              "transforms=(plgpu.TilingTransform(...), plgpu.SwizzleTransform(b))",
              ("L1.ldmatrix", "L1.stmatrix"), knobs=("swizzle",)),
    Construct("TMA load", "cpasync.CopyBulkTensorTileG2SOp + mbarrier",
              "plgpu.copy_gmem_to_smem(src, dst, barrier); plgpu.barrier_wait(barrier)",
              ("L1.tma", "L1.tma_dims")),
    Construct("TMA multicast", "multicast TMA atom over a cluster",
              "cluster= on plgpu.kernel + collective_axes= on the copy",
              ("L0.cluster", "L1.tma"), knobs=("cluster",)),
    Construct("pipeline", "cutlass.pipeline.PipelineTmaAsync(stages=...)",
              "plgpu.emit_pipeline(max_concurrent_steps=, delay_release=) or a Barrier ring",
              ("L2.pipeline_barriers", "L2.stages"), knobs=("stages", "delay_release")),
    Construct("hopper mma", "warpgroup.MmaF16BF16Op + cute.gemm",
              "plgpu.wgmma(acc, a, b); plgpu.wgmma_wait(n); plgpu.ACC",
              ("L1.mma", "L2.wgmma_wait_depths"), knobs=("wgmma_wait",)),
    Construct("blackwell mma", "tcgen05 atoms + TMEM allocators",
              "plgpu.tcgen05_mma(..., collective_axis=); plgpu.TMEM; plgpu.async_load_tmem",
              ("L1.mma", "L0.cluster"), knobs=("collective",)),
    Construct("block-scaled mma", "tcgen05.mma.kind.block_scale",
              "plgpu.tcgen05_mma(a_scale=, b_scale=) + plgpu.async_copy_scales_to_tmem",
              ("L1.mma",)),
    Construct("warp roles", "branch on warp / warpgroup index",
              "num_threads= + lax.axis_index + pl.when; plgpu.warp_map",
              ("L0.warpgroups", "L2.warp_specialized"), knobs=("num_threads",)),
    Construct("registers", "setmaxnreg helpers in cute.arch",
              "plgpu.set_max_registers; memory_registers=",
              ("L0.setmaxnreg", "L5.ptxas_registers"), knobs=("max_registers",)),
    Construct("TMA store", "CopyBulkTensorTileS2GOp + proxy fence",
              "plgpu.commit_smem(); copy_smem_to_gmem(...); wait_smem_to_gmem(n)",
              ("L2.epilogue", "L2.store_wait_depth"), knobs=("store_wait",)),
    Construct("scheduling", "persistent tile scheduler; cluster launch control",
              "plgpu.nd_loop; plgpu.dynamic_scheduling_loop; try_cluster_cancel",
              ("L2.persistent",), knobs=("persistent",)),
    Construct("rasterization", "swizzled tile-index mapping", "plgpu.planar_snake",
              ("L2.persistent",), knobs=("grid_tile_width",)),
    Construct("fragments", "cute.make_fragment; register layout algebra",
              "arrays in plgpu.Layout.WGMMA / TCGEN05; plgpu.layout_cast",
              ("L4.fp_flavor",)),
    Construct("sub-warpgroup roles", "per-warp branching inside a warpgroup",
              "plgpu.warp_map inside one warpgroup", ("L0.warpgroups",), tag="low-level"),
    Construct("inline ptx", "cute.arch.inline_ptx",
              "jax.experimental.mosaic_gpu inline asm", (), tag="low-level"),
    Construct("layout algebra", "arbitrary CuTe layout algebra on fragments",
              "no analogue: reductions over sub-row groups have no layout solution",
              (), tag="gap"),
)

BY_SIGNATURE: dict[str, tuple[Construct, ...]] = {}
for _c in CONSTRUCTS:
    for _s in _c.signature:
        BY_SIGNATURE.setdefault(_s, ())
        BY_SIGNATURE[_s] += (_c,)


def for_row(key: str) -> tuple[Construct, ...]:
    """Constructs whose signature covers a fingerprint diff row key.

    Matches on the row key and on its layer-qualified prefix, so ``L1.tma_dims`` finds
    the TMA entry and ``L0.block`` finds launch.
    """
    if key in BY_SIGNATURE:
        return BY_SIGNATURE[key]
    hits = tuple(c for c in CONSTRUCTS if any(key.startswith(s) or s.startswith(key)
                                              for s in c.signature))
    return hits


def knobs() -> dict[str, tuple[Construct, ...]]:
    out: dict[str, tuple[Construct, ...]] = {}
    for c in CONSTRUCTS:
        for k in c.knobs:
            out.setdefault(k, ())
            out[k] += (c,)
    return out


def gaps() -> tuple[Construct, ...]:
    return tuple(c for c in CONSTRUCTS if c.tag == "gap")
