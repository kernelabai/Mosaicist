"""Capture workers: run one kernel and write a Bundle.

Both compilers dump their intermediates through environment variables that have to be
set before the kernel compiles, and the two live in different virtualenvs (CuTeDSL wants
one CUDA major version, JAX another), so a capture is always its own process. That is
what `mosaicist capture` is: point it at a module exposing `build()` and it produces a
bundle directory the rest of the pipeline can read without knowing which compiler ran.

A capture module looks like this::

    def build():
        import jax.numpy as jnp
        from my_kernel import matmul
        a, b = make_inputs()
        return matmul, (a, b)

`build()` returns `(callable, args)`. The callable is invoked as `callable(*args)`; its
result is what gets saved for the numerics gate.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .bundle import Bundle, Launch

#: Dump switches per compiler. Both are read when the kernel is compiled, not when it is
#: called, so they must be in the environment before `build()` imports anything.
DUMP_ENV = {
    "pallas": {
        "MOSAIC_GPU_DUMP_PTX": "1",
        "MOSAIC_GPU_DUMP_PTXAS": "1",
        "MOSAIC_GPU_DUMP_SASS": "1",
        "MOSAIC_GPU_DUMP_TO": None,  # filled in with the dump directory
    },
    "cutedsl": {
        "CUTE_DSL_KEEP_PTX": "1",
        "CUTE_DSL_KEEP_CUBIN": "1",
        "CUTE_DSL_DUMP_DIR": None,
    },
}

KIND_OF = {"pallas": "candidate", "cutedsl": "reference"}


def dump_env(compiler: str, dump_dir: Path) -> dict[str, str]:
    """The environment a capture subprocess needs, ready to merge into os.environ."""
    if compiler not in DUMP_ENV:
        raise ValueError(f"unknown compiler {compiler!r}; expected one of {sorted(DUMP_ENV)}")
    return {k: (str(dump_dir) if v is None else v) for k, v in DUMP_ENV[compiler].items()}


def _pick(paths: list[Path], prefer: str | None = None) -> Path | None:
    """The most plausible artifact when a compiler dumps several.

    Prefer a name containing `prefer`, then the largest file: a run may dump helper
    kernels (a memset, a transpose) alongside the one being measured, and the kernel of
    interest is essentially always the biggest.
    """
    if not paths:
        return None
    if prefer:
        named = [p for p in paths if prefer in p.name]
        if named:
            paths = named
    return max(paths, key=lambda p: p.stat().st_size)


def collect_artifacts(compiler: str, dump_dir: Path, out: Path,
                      prefer: str | None = None) -> dict[str, str | None]:
    """Move the dumped PTX/SASS/ptxas log into the bundle directory.

    Returns bundle-relative names. SASS is disassembled from the cubin when the compiler
    dumps one rather than SASS directly, which is what CuTeDSL does.
    """
    out.mkdir(parents=True, exist_ok=True)
    found: dict[str, str | None] = {"ptx": None, "sass": None, "ptxas_log": None}

    ptx = _pick(sorted(dump_dir.rglob("*.ptx")), prefer)
    if ptx is not None:
        shutil.copy(ptx, out / "kernel.ptx")
        found["ptx"] = "kernel.ptx"

    sass = _pick(sorted(dump_dir.rglob("*.sass")), prefer)
    if sass is not None:
        shutil.copy(sass, out / "kernel.sass")
        found["sass"] = "kernel.sass"
    else:
        cubin = _pick(sorted(dump_dir.rglob("*.cubin")), prefer)
        if cubin is not None and shutil.which("cuobjdump"):
            try:
                text = subprocess.run(["cuobjdump", "-sass", str(cubin)], check=True,
                                      capture_output=True, text=True, timeout=300).stdout
                (out / "kernel.sass").write_text(text)
                found["sass"] = "kernel.sass"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass  # SASS is optional; L5 rows simply stay empty

    log = _pick([p for p in dump_dir.rglob("*") if p.suffix in (".log", ".txt")
                 and "ptxas" in p.name.lower()], prefer)
    if log is not None:
        shutil.copy(log, out / "ptxas.log")
        found["ptxas_log"] = "ptxas.log"
    return found


def load_build(entry: str):
    """Resolve ``module:function`` (default function ``build``) to a callable."""
    mod_name, _, fn_name = entry.partition(":")
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name or "build")
    return fn


def time_and_record(fn, args, reps: int, warmup: int) -> tuple[list[float], dict, Launch, object]:
    """Run the kernel under CUPTI, returning per-rep device times and the launch record.

    The kernel that dominates device time is the one measured: a step may launch small
    helpers around it, and those are not what is being converged.
    """
    from .bench.cupti_trace import KernelTrace

    out = fn(*args)
    _block(out)
    for _ in range(warmup):
        _block(fn(*args))

    with KernelTrace() as tr:
        for _ in range(reps):
            _block(fn(*args))

    by_name: dict[str, list] = {}
    for r in tr.records:
        by_name.setdefault(r.name, []).append(r)
    if not by_name:
        return [], {}, Launch(), out
    main = max(by_name.values(), key=lambda rs: sum(r.duration_us for r in rs))
    per_rep = [sum(r.duration_us for r in main[i::reps]) for i in range(min(reps, len(main)))]
    rec = main[0]
    launch = Launch(grid=list(rec.grid), block=list(rec.block), cluster=list(rec.cluster),
                    dynamic_smem=rec.dynamic_smem)
    resources = {"registers": rec.registers, "static_smem": rec.static_smem,
                 "local_mem_per_thread": rec.local_mem_per_thread}
    return per_rep, resources, launch, out


def _to_numpy(a):
    """A host float array from a JAX array, a torch tensor, or anything array-like.

    Torch tensors have to come off the device explicitly, and narrow float types
    (e2m1, e4m3) have no numpy equivalent, so those are saved as raw bytes -- enough
    for a bitwise comparison, which is what the gate does with them.
    """
    import numpy as np

    if hasattr(a, "detach"):  # torch: off the device, and into a dtype numpy has
        a = a.detach().cpu()
        if hasattr(a, "dtype") and str(a.dtype) in ("torch.bfloat16", "torch.float16",
                                                    "torch.float8_e4m3fn",
                                                    "torch.float8_e5m2"):
            a = a.float()
        return a.numpy().astype(np.float32)
    try:
        return np.asarray(a, dtype=np.float32)
    except (TypeError, ValueError):
        # narrow float types numpy cannot hold: keep the bits, which is all the gate
        # needs for a bitwise comparison
        return np.asarray(a).view(np.uint8).astype(np.float32)


def _block(x):
    """Wait for a result from either framework without importing both."""
    try:
        import jax

        return jax.block_until_ready(x)
    except Exception:  # noqa: BLE001 - torch, numpy, or anything already synchronous
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        return x


def versions(compiler: str) -> dict[str, str]:
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for name in (("jax", "jax") if compiler == "pallas" else ("cutlass", "nvidia-cutlass-dsl")):
        try:
            out[name[1]] = importlib.import_module(name[0]).__version__
        except Exception:  # noqa: BLE001
            pass
    if shutil.which("ptxas"):
        try:
            v = subprocess.run(["ptxas", "--version"], capture_output=True, text=True,
                               timeout=30).stdout
            out["ptxas"] = v.strip().splitlines()[-1].strip()
        except Exception:  # noqa: BLE001
            pass
    return out


def capture(entry: str, compiler: str, outdir: str | Path, *, name: str | None = None,
            reps: int = 30, warmup: int = 5, arch: str | None = None,
            prefer: str | None = None, save_outputs: bool = True) -> Bundle:
    """Run one kernel with dumping on and write a bundle to `outdir`.

    Must run in a process whose environment already has the dump switches set -- see
    `dump_env` and `mosaicist capture`, which re-executes itself to guarantee it.
    """
    out = Path(outdir)
    dump_dir = Path(os.environ.get("MOSAIC_GPU_DUMP_TO")
                    or os.environ.get("CUTE_DSL_DUMP_DIR") or (out / "dump"))
    dump_dir.mkdir(parents=True, exist_ok=True)

    fn, args = load_build(entry)()
    times, resources, launch, result = time_and_record(fn, args, reps, warmup)
    found = collect_artifacts(compiler, dump_dir, out, prefer)

    if save_outputs and result is not None:
        import numpy as np

        arrays = result if isinstance(result, (tuple, list)) else [result]
        for i, a in enumerate(arrays):
            np.save(out / f"out{i}.npy", _to_numpy(a))

        # the float64 oracle, if the module offers one: the numerics gate needs a third
        # opinion, and computing it here keeps it beside the outputs it judges
        module = importlib.import_module(entry.partition(":")[0])
        oracle_fn = getattr(module, "reference", None)
        if callable(oracle_fn):
            try:
                ref_out = oracle_fn(*args)
                for i, a in enumerate(ref_out if isinstance(ref_out, (tuple, list)) else [ref_out]):
                    np.save(out / f"oracle{i}.npy", _to_numpy(a).astype(np.float64))
            except Exception:  # noqa: BLE001 - an oracle is optional
                pass

    bundle = Bundle(kind=KIND_OF[compiler], name=name or entry, arch=arch,
                    ptx=found["ptx"], ptxas_log=found["ptxas_log"], sass=found["sass"],
                    source=entry, launch=launch, timings=times, resources=resources,
                    versions=versions(compiler), root=out)
    bundle.save(out)
    return bundle


def capture_subprocess(entry: str, compiler: str, outdir: str | Path, *,
                       python: str | None = None, reps: int = 30,
                       extra_env: dict[str, str] | None = None,
                       timeout: int = 1800) -> Bundle:
    """Capture in a fresh process, so the dump switches are set before anything imports.

    `python` selects the interpreter, which is how a CuTeDSL reference and a Pallas
    candidate get captured from one driver despite needing different virtualenvs.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(dump_env(compiler, out / "dump"))
    env.update(extra_env or {})
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    cmd = [python or sys.executable, "-m", "mosaicist.cli", "capture", entry,
           "--compiler", compiler, "--out", str(out), "--reps", str(reps), "--in-process"]
    subprocess.run(cmd, check=True, env=env, timeout=timeout)
    return Bundle.load(out)
