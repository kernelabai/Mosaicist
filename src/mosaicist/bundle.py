"""The Bundle: everything captured about one compiled kernel, from either compiler.

A bundle directory holds `bundle.json` plus the artifacts it names
(PTX, ptxas log, SASS, outputs). CuTeDSL capture and Pallas capture both
write this schema, so nothing downstream cares which compiler produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ptx.features import Fingerprint, fingerprint

BUNDLE_FILE = "bundle.json"


@dataclass
class Launch:
    grid: list[int] | None = None
    block: list[int] | None = None
    cluster: list[int] | None = None
    dynamic_smem: int | None = None


@dataclass
class Bundle:
    kind: str  # "reference" (CuTeDSL) | "candidate" (Pallas Mosaic GPU)
    name: str
    arch: str | None = None  # e.g. "sm_90a"
    entry: str | None = None  # PTX .entry name, if the module has several
    ptx: str | None = None  # paths relative to the bundle directory
    ptxas_log: str | None = None
    sass: str | None = None
    source: str | None = None  # kernel source file
    launch: Launch = field(default_factory=Launch)
    timings: list[float] = field(default_factory=list)  # device-time samples
    versions: dict[str, str] = field(default_factory=dict)  # cutlass / jax / ptxas / driver
    root: Path | None = field(default=None, repr=False, compare=False)

    def path(self, rel: str | None) -> Path | None:
        if rel is None:
            return None
        return (self.root / rel) if self.root else Path(rel)

    def read(self, rel: str | None) -> str | None:
        p = self.path(rel)
        return p.read_text() if p is not None and p.exists() else None

    def fingerprint(self) -> Fingerprint:
        ptx = self.read(self.ptx)
        if ptx is None:
            raise FileNotFoundError(f"bundle {self.name!r} has no readable PTX ({self.ptx})")
        return fingerprint(ptx, entry=self.entry, launch=asdict(self.launch),
                           ptxas_log=self.read(self.ptxas_log), sass=self.read(self.sass))

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("root")
        (d / BUNDLE_FILE).write_text(json.dumps(data, indent=2))
        return d / BUNDLE_FILE

    @classmethod
    def load(cls, directory: str | Path) -> "Bundle":
        d = Path(directory)
        data = json.loads((d / BUNDLE_FILE).read_text())
        data["launch"] = Launch(**(data.get("launch") or {}))
        return cls(**data, root=d)

    @classmethod
    def from_ptx(cls, ptx_path: str | Path, kind: str, ptxas_log: str | Path | None = None,
                 entry: str | None = None) -> "Bundle":
        """A minimal bundle around a bare PTX file (no launch record, no timings)."""
        p = Path(ptx_path)
        return cls(kind=kind, name=p.stem, ptx=p.name, entry=entry,
                   ptxas_log=str(Path(ptxas_log).resolve()) if ptxas_log else None, root=p.parent)
