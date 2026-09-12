"""Lower the sm_100a kernels for Blackwell on whatever GPU is present.

Mosaic GPU builds its kernel module in Python and picks the target from the default
device's compute capability (`mosaic.gpu.core._infer_arch`). Pointing that at (10, 0)
runs the entire Mosaic lowering -- including the tcgen05 and scale-copy shape checks
that plain tracing does not reach -- and then asks ptxas for sm_100a code.

That is not a correctness test: it proves the kernel builds and assembles for
Blackwell, not that it computes the right answer. It is the most that can be checked
without the hardware, and it catches the layout and shape mistakes that are otherwise
invisible until someone with a B200 runs it.
"""

import sys

import jax
import jax.numpy as jnp
from jax.experimental.mosaic.gpu import core as mgpu_core

mgpu_core._infer_arch = lambda: (10, 0)  # pretend the default device is sm_100a

from masked_gemm import GemmConfig, masked_grouped_gemm  # noqa: E402
from nvfp4 import FP4_DTYPE, SF_DTYPE  # noqa: E402

sd = jax.ShapeDtypeStruct
ok = True


def attempt(label, fn, *args):
    global ok
    try:
        lowered = jax.jit(fn).lower(*args)
        text = lowered.as_text()
        has_mma = "tcgen05" in text or "mosaic" in text
        print(f"PASS  {label:<40} lowered ({len(text) // 1024} KiB IR"
              f"{', tcgen05 present' if has_mma else ''})")
    except Exception as e:  # noqa: BLE001
        msg = " ".join(str(e).split())
        print(f"FAIL  {label:<40} {msg[:220]}")
        ok = False


def gemm_args(l=2, m=256, k=256, n=128):
    return (sd((l, m, k), FP4_DTYPE), sd((l, m // 128, k // 64, 32, 16), SF_DTYPE),
            sd((l, n, k), FP4_DTYPE), sd((l, n // 128, k // 64, 32, 16), SF_DTYPE),
            sd((l,), jnp.float32), sd((l,), jnp.int32))


attempt("masked_grouped_gemm", lambda *a: masked_grouped_gemm(*a, GemmConfig()), *gemm_args())
attempt("masked_grouped_gemm block_k=256",
        lambda *a: masked_grouped_gemm(*a, GemmConfig(block_k=256, stages=2)),
        *gemm_args(k=512))

# Negative control: the checks above only mean something if they can fail. Breaking the
# scale tiling constant makes the smem scale buffer the wrong shape for the TMEM ref,
# which only `async_copy_scales_smem_to_tmem` can catch -- so if this lowers, the run
# above was not reaching the scale path at all and the PASSes are worthless.
import masked_gemm as _mg  # noqa: E402

# Widening the scale vector leaves every TMA in bounds but makes the TMEM scale ref
# too narrow for the smem slab feeding it -- an error only the scale path reports.
_real_vec = _mg.SF_VEC_SIZE
_mg.SF_VEC_SIZE = _real_vec * 2
try:
    jax.jit(lambda *a: _mg.masked_grouped_gemm(*a, GemmConfig())).lower(*gemm_args())
except Exception as e:  # noqa: BLE001
    print(f"PASS  {'negative control rejects bad scales':<40} "
          f"{' '.join(str(e).split())[:90]}")
else:
    print(f"FAIL  {'negative control rejects bad scales':<40} "
          "lowered anyway -- the scale checks are not running")
    ok = False
finally:
    _mg.SF_VEC_SIZE = _real_vec

try:
    from quantize_kernels import quantize_nvfp4_pallas, silu_mul_quantize_nvfp4_pallas
except ImportError:
    print("SKIP  quantize kernels (not written yet)")
else:
    l, m, k, n = 2, 256, 512, 256
    attempt("quantize_nvfp4_pallas", quantize_nvfp4_pallas,
            sd((l, m, k), jnp.bfloat16), sd((l,), jnp.float32), sd((l,), jnp.int32))
    attempt("silu_mul_quantize_nvfp4_pallas", silu_mul_quantize_nvfp4_pallas,
            sd((l, m, 2 * n), jnp.bfloat16), sd((l,), jnp.float32), sd((l,), jnp.int32))

sys.exit(0 if ok else 1)
