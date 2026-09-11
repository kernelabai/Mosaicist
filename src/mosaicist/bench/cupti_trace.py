"""Device time and launch records via CUPTI activity tracing.

Works in-process for any compiler (CuTeDSL, XLA / Mosaic GPU): every kernel
launch produces a record with its grid, block, cluster, shared memory,
registers, and device start/end timestamps. Activity tracing does not need
performance-counter permissions (RmProfilingAdminOnly), unlike ncu.

Requires `cupti-python` matching the CUDA major version of the process
(12.x for jax[cuda12], 13.x for CUDA 13 torch/CuTeDSL stacks).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable


@dataclass
class KernelRecord:
    name: str
    start_ns: int
    end_ns: int
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    cluster: tuple[int, int, int]
    dynamic_smem: int
    static_smem: int
    registers: int
    local_mem_per_thread: int

    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.start_ns) / 1e3

    def launch(self) -> dict:
        """The Bundle `launch` record for this kernel."""
        return {"grid": list(self.grid), "block": list(self.block),
                "cluster": list(self.cluster), "dynamic_smem": self.dynamic_smem}

    def to_dict(self) -> dict:
        return asdict(self)


class KernelTrace:
    """Context manager collecting a KernelRecord for every kernel launched inside it."""

    def __init__(self, buffer_bytes: int = 8 << 20):
        self.records: list[KernelRecord] = []
        self._buffer_bytes = buffer_bytes

    def __enter__(self) -> "KernelTrace":
        from cupti import cupti

        self._cupti = cupti
        kinds = (cupti.ActivityKind.CONCURRENT_KERNEL, cupti.ActivityKind.KERNEL)

        def requested():
            return self._buffer_bytes, 0

        def completed(activities):
            for a in activities:
                if a.kind in kinds:
                    name = a.name.decode() if isinstance(a.name, bytes) else str(a.name)
                    self.records.append(KernelRecord(
                        name=name, start_ns=int(a.start), end_ns=int(a.end),
                        grid=(a.grid_x, a.grid_y, a.grid_z),
                        block=(a.block_x, a.block_y, a.block_z),
                        cluster=(max(1, a.cluster_x), max(1, a.cluster_y), max(1, a.cluster_z)),
                        dynamic_smem=int(a.dynamic_shared_memory), static_smem=int(a.static_shared_memory),
                        registers=int(a.registers_per_thread),
                        local_mem_per_thread=int(a.local_memory_per_thread),
                    ))

        cupti.activity_register_callbacks(requested, completed)
        cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
        return self

    def __exit__(self, *exc) -> None:
        self._cupti.activity_flush_all(1)
        self._cupti.activity_disable(self._cupti.ActivityKind.CONCURRENT_KERNEL)

    def matching(self, name_substr: str | None = None, exclude: tuple[str, ...] = ()) -> list[KernelRecord]:
        return [r for r in self.records
                if (name_substr is None or name_substr in r.name) and not any(x in r.name for x in exclude)]


def time_kernel(run: Callable[[], None], sync: Callable[[], None], flush: Callable[[], None] | None,
                reps: int = 50, warmup: int = 10, name_substr: str | None = None,
                exclude: tuple[str, ...] = ()) -> tuple[list[float], KernelRecord]:
    """Device times (us) of the kernel launched by `run`, with an optional L2 flush before each rep.

    `flush` launches its own kernels; they are excluded from the result by
    comparing against the kernels seen in a flush-only probe.
    """
    for _ in range(warmup):
        run()
    sync()
    flush_names: set[str] = set()
    if flush is not None:
        with KernelTrace() as probe:
            flush()
            sync()
        flush_names = {r.name for r in probe.records}
    with KernelTrace() as tr:
        for _ in range(reps):
            if flush is not None:
                flush()
            run()
        sync()
    recs = [r for r in tr.matching(name_substr, exclude) if r.name not in flush_names]
    if not recs:
        raise RuntimeError(f"no kernel records captured (saw {sorted({r.name for r in tr.records})[:5]})")
    # a run() may launch several kernels; time the one that dominates
    by_name: dict[str, list[KernelRecord]] = {}
    for r in recs:
        by_name.setdefault(r.name, []).append(r)
    main = max(by_name.values(), key=lambda rs: sum(r.duration_us for r in rs))
    return [r.duration_us for r in main], main[0]
