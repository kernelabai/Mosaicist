"""Single-problem driver for mixed_gemm.HopperMixedInputGemmKernel.

    python run_single.py --mnk 2048,5120,8192 --c 512          # scale groups of 512 along K
    python run_single.py --mnk 2048,5120,8192 --c 0            # convert-only

Checks D against problems.reference (bit-exact with the default small-integer
inputs) and reports device time two ways: CUDA events around each launch (what
the C++ example reports) and CUPTI kernel duration.
"""

import argparse
import sys
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, str(Path(__file__).parent))
from mixed_gemm import HopperMixedInputGemmKernel  # noqa: E402
from problems import flops, make_problems, reference  # noqa: E402

QUANT = {"e5m2": (torch.float8_e5m2, cutlass.Float8E5M2), "e4m3": (torch.float8_e4m3fn, cutlass.Float8E4M3FN)}


def kernel_tensors(gp, quant_cute):
    """Problem-space tensors -> kernel-space cute tensors (mode order (rows, K, L))."""
    a = from_dlpack(gp.Bq.view(torch.uint8).unsqueeze(0).permute(1, 2, 0), assumed_align=16)
    a.element_type = quant_cute
    a = a.mark_layout_dynamic(leading_dim=1)
    scale_t = gp.scale.T.contiguous() if gp.scale is not None else torch.ones(1, gp.n, dtype=torch.bfloat16,
                                                                               device=gp.Bq.device)
    s = from_dlpack(scale_t.unsqueeze(0).permute(2, 1, 0), assumed_align=16).mark_layout_dynamic(leading_dim=0)
    b = from_dlpack(gp.A.unsqueeze(0).permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    c = from_dlpack(gp.D.unsqueeze(0).permute(2, 1, 0), assumed_align=16).mark_layout_dynamic(leading_dim=0)
    return a, s, b, c, scale_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnk", default="2048,5120,8192")
    ap.add_argument("--c", type=int, default=512, help="scale group size along K; 0 = convert-only")
    ap.add_argument("--quant", choices=list(QUANT), default="e5m2")
    ap.add_argument("--tile", default="128,16")
    ap.add_argument("--raster", choices=["m", "n"], default="m", help="persistent tile order (C++: along M)")
    ap.add_argument("--swizzle", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    m, n, k = (int(x) for x in args.mnk.split(","))
    tq, cq = QUANT[args.quant]
    gp = make_problems([(m, n, k)], quant_dtype=tq, scale_c=args.c or None, alpha=1.0, beta=0.0)[0]
    a, s, b, c, _keep = kernel_tensors(gp, cq)

    gemm = HopperMixedInputGemmKernel(tile_shape_mn=tuple(int(x) for x in args.tile.split(",")),
                                      scale_granularity_k=args.c, swizzle_size=args.swizzle,
                                      raster_along_m=args.raster == "m")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    compiled = cute.compile(gemm, a, s, b, c, max_clusters, stream)

    compiled(a, s, b, c, stream)
    torch.cuda.synchronize()
    ref = reference(gp)
    exact = torch.equal(gp.D, ref)
    max_err = (gp.D.float() - ref.float()).abs().max().item()
    print(f"correctness: {'BIT-EXACT' if exact else 'MISMATCH'}  max|D-ref|={max_err:g}  "
          f"(stages={gemm.ab_stage}, tile={gemm.tile_shape_mnk})")

    for _ in range(args.warmup):
        compiled(a, s, b, c, stream)
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.iterations)]
    for e0, e1 in events:
        e0.record()
        compiled(a, s, b, c, stream)
        e1.record()
    torch.cuda.synchronize()
    ms = sum(e0.elapsed_time(e1) for e0, e1 in events) / len(events)
    print(f"CuTeDSL avg runtime (CUDA events): {ms:.4f} ms   {flops([gp]) / ms / 1e9:.1f} TFLOP/s")
    try:
        from mosaicist.bench.cupti_trace import time_kernel

        times, rec = time_kernel(lambda: compiled(a, s, b, c, stream), torch.cuda.synchronize, None,
                                 reps=args.iterations, warmup=2)
        med = sorted(times)[len(times) // 2]
        print(f"CuTeDSL median device time (CUPTI):  {med / 1e3:.4f} ms   grid={rec.grid} block={rec.block} "
              f"regs={rec.registers} smem={rec.dynamic_smem}")
    except Exception as e:  # CUPTI is optional
        print(f"(CUPTI timing unavailable: {e})")
    sys.exit(0 if exact else 1)


if __name__ == "__main__":
    main()
