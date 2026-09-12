"""Generate a v0 kernel from a contract and check it against a float64 oracle.

This is M2 end to end without a capture in front of it: contract -> translator -> a
runnable module -> the numerics gate.
"""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.contract import Contract, Operand  # noqa: E402
from mosaicist.translate import emit_v0  # noqa: E402
from mosaicist.verify.numerics import GateConfig, compare_outputs  # noqa: E402

arch = sys.argv[1] if len(sys.argv) > 1 else "sm_100a"
mma = "tcgen05" if arch == "sm_100a" else "wgmma"
M, N, K = 512, 512, 1024

contract = Contract(
    name=f"gemm_{arch}", arch=arch,
    operands=[Operand("a", (M, K), "bfloat16", ("m", "k")),
              Operand("b", (K, N), "bfloat16", ("k", "n"))],
    out_shape=(M, N), out_dtype="bfloat16",
    dims={"m": M, "n": N, "k": K}, tile={"m": 128, "n": 128},
    mma=mma, mma_kind="f16",
)
out_dir = pathlib.Path(__file__).parent / "generated"
path = emit_v0(contract, out_dir / f"v0_{arch}.py")
print(f"emitted {path.relative_to(pathlib.Path.cwd()) if path.is_absolute() else path}")

sys.path.insert(0, str(out_dir))
mod = __import__(f"v0_{arch}")
fn, args = mod.build()
got = np.asarray(jax.block_until_ready(fn(*args)), np.float32)
oracle = mod.reference(*args)
# the reference must land in the same output format as the candidate, or the gate
# compares a bf16 kernel against an fp32 one and rejects it for being bf16
ref = np.asarray(jnp.dot(args[0], args[1], preferred_element_type=jnp.float32)
                 .astype(jnp.bfloat16), np.float32)

report = compare_outputs(ref, got, oracle, "bf16", GateConfig())
print(report.summary())
print("max |v0 - xla| /|xla| =",
      float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)))
sys.exit(0 if report.passed else 1)
