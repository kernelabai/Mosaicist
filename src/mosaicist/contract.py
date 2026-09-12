"""The contract: what a translation has to preserve.

Extracted deterministically from a reference capture (DESIGN §5, step 1). It fixes the
things v0 must match -- output tile shape, tile-to-block mapping, K traversal order,
accumulator type -- and deliberately says nothing about how to go fast, because v0's job
is to be obviously correct and leave the coarse fingerprint layers already aligned.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .bundle import Bundle
from .ptx.features import Fingerprint


@dataclass
class Operand:
    name: str
    shape: tuple[int, ...]
    dtype: str
    #: which logical axes this operand carries, e.g. ("l", "m", "k")
    axes: tuple[str, ...] = ()


@dataclass
class Contract:
    """Everything v0 must reproduce, and nothing about performance."""

    name: str
    arch: str
    operands: list[Operand] = field(default_factory=list)
    out_shape: tuple[int, ...] = ()
    out_dtype: str = "bfloat16"
    #: logical problem sizes, e.g. {"l": 8, "m": 512, "n": 2048, "k": 2048}
    dims: dict[str, int] = field(default_factory=dict)
    tile: dict[str, int] = field(default_factory=dict)  # {"m": 128, "n": 128, "k": 64}
    grid: list[int] | None = None
    block: list[int] | None = None
    cluster: list[int] | None = None
    mma: str | None = None  # "wgmma" | "tcgen05" | None
    mma_kind: str | None = None  # e.g. "mxf4nvf4", "f16"
    acc_dtype: str = "float32"
    stages: int | None = None
    #: things the reference does that v0 deliberately does not copy
    deferred: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(asdict(self), indent=2, default=list))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Contract":
        d = json.loads(Path(path).read_text())
        d["operands"] = [Operand(**o) for o in d.get("operands", [])]
        return cls(**d)


#: An inferred tile larger than this is not a tile -- it means the grid does not tile
#: that dimension, which is what a persistent launch looks like.
MAX_PLAUSIBLE_TILE = 256


def _tile_from_launch(fp: Fingerprint, dims: dict[str, int]) -> dict[str, int]:
    """Infer the output tile from the grid and the problem dims, where the grid tiles.

    For a tiled launch the grid extent is the tile count, so the tile is the dimension
    divided by it. A persistent launch breaks that: its grid is the SM count regardless
    of problem size, and dividing by it yields the whole dimension back. Rather than
    record a fiction, axes the grid does not plausibly tile are left out -- the B200
    reference measured here launches grid=[1,1,148] for a 512x2048 output, and the naive
    reading called that a 512x2048 "tile".
    """
    grid = [g for g in (fp.skeleton.get("grid") or []) if g]
    tile: dict[str, int] = {}
    if not grid or not dims:
        return tile
    # match grid axes to dims, largest-first, which is how a tiled GEMM maps them
    named = [k for k in ("m", "n", "l") if k in dims]
    for axis, extent in zip(named, grid):
        if extent and dims[axis] % extent == 0:
            candidate = dims[axis] // extent
            if candidate <= MAX_PLAUSIBLE_TILE:
                tile[axis] = candidate
    return tile


def _launch_deferrals(fp: Fingerprint, dims: dict[str, int],
                      tile: dict[str, int]) -> list[str]:
    """What the launch record says the reference does that v0 will not.

    The PTX alone hides both of these: the fingerprint's structural detectors read
    control flow, while a capture carries the real grid and block, and those are what
    give a persistent, warp-specialised kernel away.
    """
    out: list[str] = []
    grid = [g for g in (fp.skeleton.get("grid") or []) if g]
    threads = fp.skeleton.get("threads")

    if grid and any(k in dims for k in ("m", "n", "l")) and not tile:
        out.append(f"persistent scheduling (grid {grid} does not tile {dims})")
    if threads and threads % 128:
        out.append(f"warp specialization ({threads} threads is not whole warpgroups)")
    return out


def from_capture(bundle: Bundle, dims: dict[str, int], operands: list[Operand],
                 out_shape: tuple[int, ...], out_dtype: str = "bfloat16",
                 name: str | None = None) -> Contract:
    """Build a contract from a reference bundle plus the shapes the caller knows.

    Shapes and dtypes come from the caller because they are properties of the problem,
    not of the compiled kernel; everything else is read out of the capture.
    """
    fp = bundle.fingerprint()
    mma, kind = None, None
    for key in fp.mma_signatures:
        mma = "tcgen05" if key.startswith("tcgen05") else "wgmma"
        kind = key.split("kind::")[-1] if "kind::" in key else key.split(":")[-1]
        break

    tile = _tile_from_launch(fp, dims)
    deferred = []
    if fp.structure.get("warp_specialized"):
        deferred.append("warp specialization")
    if fp.structure.get("persistent"):
        deferred.append("persistent scheduling")
    if (fp.skeleton.get("cluster") or [1])[0] not in (0, 1, None):
        deferred.append("cluster / collective MMA")
    for extra in _launch_deferrals(fp, dims, tile):
        if not any(extra.split()[0] in d for d in deferred):
            deferred.append(extra)
    mma_loops = fp.mma_loops()

    return Contract(
        name=name or bundle.name,
        arch=bundle.arch or fp.target or "sm_90a",
        operands=list(operands),
        out_shape=tuple(out_shape),
        out_dtype=out_dtype,
        dims=dict(dims),
        tile=tile,
        grid=fp.skeleton.get("grid"),
        block=fp.skeleton.get("block"),
        cluster=fp.skeleton.get("cluster"),
        mma=mma,
        mma_kind=kind,
        stages=mma_loops[0].stages if mma_loops else None,
        deferred=deferred,
    )


def block_k_for(contract: Contract, smem_budget: int = 227 * 1024 // 2) -> int:
    """A K step that keeps one stage inside `smem_budget`, preferring the widest.

    Half the SM's shared memory by default: a block above that runs alone, and occupancy
    has dominated every tuning result on both architectures we target.
    """
    tm = contract.tile.get("m", 128)
    tn = contract.tile.get("n", 128)
    k = contract.dims.get("k", 0)
    bits = {"float4_e2m1fn": 4, "float8_e4m3fn": 8, "bfloat16": 16, "float16": 16}
    w = bits.get(contract.operands[0].dtype if contract.operands else "bfloat16", 16)
    for bk in (512, 256, 128, 64):
        if k % bk:
            continue
        per_stage = (tm + tn) * bk * w // 8
        if per_stage * 2 <= smem_budget:  # room for at least two stages
            return bk
    return 64 if not k or k % 128 else 128


def stages_for(contract: Contract, block_k: int, smem_budget: int = 227 * 1024 // 2) -> int:
    tm, tn = contract.tile.get("m", 128), contract.tile.get("n", 128)
    bits = {"float4_e2m1fn": 4, "float8_e4m3fn": 8, "bfloat16": 16, "float16": 16}
    w = bits.get(contract.operands[0].dtype if contract.operands else "bfloat16", 16)
    per_stage = (tm + tn) * block_k * w // 8
    epilogue = tm * tn * 2
    return max(1, min(8, (smem_budget - epilogue) // max(per_stage, 1)))


def num_k_blocks(contract: Contract, block_k: int) -> int:
    k = contract.dims.get("k", 0)
    return math.ceil(k / block_k) if k else 0
