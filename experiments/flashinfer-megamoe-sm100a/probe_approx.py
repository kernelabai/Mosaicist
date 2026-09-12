"""Does approx_math remove the div.full.f32 cost, and does it change the output?

The PTX for the fused silu+quantize kernel has 448 div.full.f32 against 256
ex2.approx.f32: the exponential is already the hardware instruction, and the
full-precision divides are what cost. `approx_math=True` sets the `afn` fast-math flag,
which should turn them into single-instruction approximate divides. The outputs are
e2m1 and e4m3 -- 3 and 4 mantissa bits -- so there is a lot of precision to spare, but
whether it is *bit*-identical to the reference is a question for the hardware.
"""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mosaicist.bench.cupti_trace import KernelTrace  # noqa: E402

import quantize_kernels as qk  # noqa: E402
from nvfp4 import quantize_nvfp4, silu_mul  # noqa: E402

L, ROWS, K, N = 8, 512, 2048, 1024
x = jax.random.normal(jax.random.key(0), (L, ROWS, K), jnp.float32).astype(jnp.bfloat16)
gu = jax.random.normal(jax.random.key(1), (L, ROWS, 2 * N), jnp.float32).astype(jnp.bfloat16) * 3
gs = jnp.full((L,), 64.0, jnp.float32)
mask = jnp.full((L,), ROWS, jnp.int32)

want = {
    "quantize": quantize_nvfp4(x.astype(jnp.float32), gs),
    "silu+quantize": quantize_nvfp4(silu_mul(gu), gs),
}

for approx in (False, True):
    qk.APPROX_MATH = approx
    # APPROX_MATH is read when the kernel is traced, but jax.jit caches on the function
    # object and its avals -- without this the second setting silently reuses the first
    # compilation and both rows report the same number.
    jax.clear_caches()
    for name, fn, arg in [("quantize", qk.quantize_nvfp4_pallas, x),
                          ("silu+quantize", qk.silu_mul_quantize_nvfp4_pallas, gu)]:
        f = jax.jit(fn)
        q, sf = jax.block_until_ready(f(arg, gs, mask))
        wq, wsf = want[name]
        same_q = np.array_equal(np.asarray(q, np.float32), np.asarray(wq, np.float32))
        same_sf = np.array_equal(np.asarray(sf, np.float32), np.asarray(wsf, np.float32))
        diff = float((np.asarray(q, np.float32) != np.asarray(wq, np.float32)).mean())
        with KernelTrace() as tr:
            for _ in range(20):
                jax.block_until_ready(f(arg, gs, mask))
        us = sum(r.duration_us for r in tr.records if "mosaic" in r.name) / 20
        tag = "bit-exact" if (same_q and same_sf) else f"{diff:.3%} values differ"
        print(f"approx_math={str(approx):<6}{name:<15}{us:7.1f} us  {tag}")
