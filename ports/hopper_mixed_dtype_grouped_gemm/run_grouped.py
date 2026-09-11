"""Driver for the CuTeDSL port of CUTLASS example 69 (Hopper mixed-dtype grouped GEMM).

Mirrors the C++ example's options:

    python run_grouped.py --groups 16 --m 2048 --n 5120 --k 8192 --c 512      # fixed shapes
    python run_grouped.py --groups 6                                          # random M (16..1024), N=2048, K=512
    python run_grouped.py --groups 16 --m 2048 --n 5120 --k 8192 --c 0        # convert-only (mode 0)

Omitted --m is randomized per group like the example (alignment * [1, 64]); omitted
--alpha / --beta are random per group (1..5 and 0..4). Every group's D is checked against
the reference (bit-exact with the default small-integer inputs); timing is the mean of
CUDA-event-timed launches, which is what the C++ example reports.
"""

import argparse
import random
import sys
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, str(Path(__file__).parent))
from grouped_mixed_gemm import BYTES_PER_TENSORMAP, NUM_TENSORMAPS, HopperMixedInputGroupedGemmKernel  # noqa: E402
from grouped_utils import NUM_TENSORS  # noqa: E402
from problems import flops, make_problems, reference  # noqa: E402
from run_single import QUANT, kernel_tensors  # noqa: E402


def build(groups, quant_cute, tile_mn):
    """Kernel-space metadata for all groups (kM = N, kN = M)."""
    dev = groups[0].A.device
    shapes, strides, ptrs, keep = [], [], [], []
    for gp in groups:
        km, kn, k = gp.n, gp.m, gp.k
        if km % 16 or k % 16:
            raise ValueError(f"N and K must be multiples of 16 (128-bit TMA alignment of fp8 rows); got {gp.n}, {gp.k}")
        scale_t = gp.scale.T.contiguous() if gp.scale is not None else torch.ones(1, km, dtype=torch.bfloat16,
                                                                                   device=dev)
        keep.append(scale_t)
        shapes.append([km, kn, k, 1])
        strides.append([
            [k, 1],  # A (kM, K)   = Bq (N, K) row-major
            [k, 1],  # B (kN, K)   = A (M, K) row-major
            [1, km],  # S (kM, K/c) = scale^T, kM contiguous
            [1, km],  # D (kM, kN)  = D (M, N) row-major, kM contiguous
            [1, km],  # C (kM, kN)  = C (M, N) row-major
        ])
        ptrs.append([gp.Bq.data_ptr(), gp.A.data_ptr(), scale_t.data_ptr(), gp.D.data_ptr(), gp.C.data_ptr()])
    t_shapes = torch.tensor(shapes, dtype=torch.int32, device=dev)
    t_strides = torch.tensor(strides, dtype=torch.int32, device=dev)
    t_ptrs = torch.tensor(ptrs, dtype=torch.int64, device=dev)
    assert t_strides.shape[1] == NUM_TENSORS
    t_alpha = torch.tensor([gp.alpha for gp in groups], dtype=torch.float32, device=dev)
    t_beta = torch.tensor([gp.beta for gp in groups], dtype=torch.float32, device=dev)
    total_tiles = sum(-(-s[0] // tile_mn[0]) * -(-s[1] // tile_mn[1]) for s in shapes)
    return t_shapes, t_strides, t_ptrs, t_alpha, t_beta, total_tiles, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--m", type=int, default=None, help="all groups' M (default: random per group)")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--c", type=int, default=512, help="scale group size along K; 0 = convert-only")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--quant", choices=list(QUANT), default="e5m2")
    ap.add_argument("--tile", default="128,16")
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--benchmark", help="problem file in the C++ example's format: lines 'idx MxNxK'")
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    if args.benchmark:
        shapes = []
        for line in Path(args.benchmark).read_text().split("\n"):
            if line.strip():
                _, ext = line.split()
                m, n, k = (int(x) for x in ext.split("x"))
                shapes.append((m, n, k))
        args.groups = len(shapes)
    else:
        shapes = [(args.m if args.m is not None else 16 * rnd.randint(1, 64), args.n, args.k)
                  for _ in range(args.groups)]
    tq, cq = QUANT[args.quant]
    groups = make_problems(shapes, quant_dtype=tq, scale_c=args.c or None, alpha=args.alpha, beta=args.beta,
                           seed=args.seed)
    tile = tuple(int(x) for x in args.tile.split(","))
    t_shapes, t_strides, t_ptrs, t_alpha, t_beta, total_tiles, keep = build(groups, cq, tile)
    a0, s0, b0, d0, keep0 = kernel_tensors(groups[0], cq)

    hw = cutlass.utils.HardwareInfo()
    max_clusters = hw.get_max_active_clusters(1)
    workspace = torch.empty((max_clusters, NUM_TENSORMAPS, BYTES_PER_TENSORMAP // 8), dtype=torch.int64,
                            device="cuda")
    to_cute = lambda t: from_dlpack(t, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    gemm = HopperMixedInputGroupedGemmKernel(tile_shape_mn=tile, scale_granularity_k=args.c)
    kargs = (a0, s0, b0, d0, args.groups, to_cute(t_shapes), to_cute(t_strides), to_cute(t_ptrs), to_cute(t_alpha),
             to_cute(t_beta), total_tiles, to_cute(workspace), max_clusters, stream)
    compiled = cute.compile(gemm, *kargs)
    run_args = tuple(x for i, x in enumerate(kargs) if i not in (4, 10, 12))  # drop Constexpr args

    compiled(*run_args)
    torch.cuda.synchronize()
    bad = []
    for i, gp in enumerate(groups):
        ref = reference(gp)
        if not torch.equal(gp.D, ref):
            bad.append((i, (gp.m, gp.n, gp.k), (gp.D.float() - ref.float()).abs().max().item()))
    print(f"groups={args.groups} tiles={total_tiles} stages={gemm.ab_stage} "
          f"shapes={args.benchmark or ('fixed ' + str(shapes[0]) if args.m is not None else 'random M')} c={args.c}")
    print("correctness:", "BIT-EXACT (all groups)" if not bad else f"MISMATCH in {len(bad)} groups: {bad[:4]}")

    for _ in range(args.warmup):
        compiled(*run_args)
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(args.iterations)]
    for e0, e1 in events:
        e0.record()
        compiled(*run_args)
        e1.record()
    torch.cuda.synchronize()
    ms = sum(e0.elapsed_time(e1) for e0, e1 in events) / len(events)
    print(f"CuTeDSL avg runtime: {ms:.4f} ms   {flops(groups) / ms / 1e9:.1f} TFLOP/s")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
