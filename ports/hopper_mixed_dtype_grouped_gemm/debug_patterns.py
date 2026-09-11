"""Localize mixed_gemm bugs with structured inputs (convert-only, one small problem)."""

import sys
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mixed_gemm import HopperMixedInputGemmKernel  # noqa: E402
from problems import make_problems, reference  # noqa: E402
from run_single import kernel_tensors  # noqa: E402

m, n, k = 32, 128, 128
gp = make_problems([(m, n, k)], scale_c=None)[0]
gemm = HopperMixedInputGemmKernel(tile_shape_mn=(128, 16), scale_granularity_k=0)
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
a, s, b, c, keep = kernel_tensors(gp, cutlass.Float8E5M2)
compiled = cute.compile(gemm, a, s, b, c, cutlass.utils.HardwareInfo().get_max_active_clusters(1), stream)


def run(A, Bq, label):
    gp.A.copy_(A)
    gp.Bq.copy_(Bq.to(torch.float8_e5m2))
    gp.D.zero_()
    compiled(a, s, b, c, stream)
    torch.cuda.synchronize()
    ref = reference(gp)
    bad = (gp.D != ref)
    print(f"{label:<34} exact={not bad.any().item()}  bad={bad.sum().item()}/{bad.numel()}")
    return gp.D.float().cpu(), ref.float().cpu()


dev = "cuda"
ones_mk = torch.ones(m, k, dtype=torch.bfloat16, device=dev)
ones_nk = torch.ones(n, k, device=dev)
rowid = torch.arange(n, device=dev, dtype=torch.float32)[:, None].expand(n, k) % 7
colk = (torch.arange(k, device=dev, dtype=torch.float32)[None, :].expand(n, k) % 5)
run(ones_mk, ones_nk, "A=1, B=1 (expect K everywhere)")
run(torch.randint(-2, 3, (m, k), device=dev).bfloat16(), ones_nk, "A=rand, B=1 (A-path neutral)")
D, R = run(ones_mk, rowid, "A=1, B[n,k]=n%7 (B rows)")
print("  D[0,:16]  ", D[0, :16].tolist())
print("  ref[0,:16]", R[0, :16].tolist())
D, R = run(ones_mk, colk, "A=1, B[n,k]=k%5 (K order)")
print("  D[0,:8]  ", D[0, :8].tolist(), " ref", R[0, :8].tolist())
for kk in (0, 1, 2, 8, 16, 63, 64):
    Bk = torch.zeros(n, k, device=dev)
    Bk[:, kk] = rowid[:, 0] + 1
    D, R = run(ones_mk, Bk, f"B nonzero only at k={kk}")
    if not torch.equal(D, R):
        print("   D[0,:16]  ", D[0, :16].tolist())
        print("   ref[0,:16]", R[0, :16].tolist())

print("--- sign / value patterns ---")
for label, Bpat in (
    ("B = -(n%7+1)", -(rowid + 1)),
    ("B = (n%5)-2 (mixed sign)", (torch.arange(n, device=dev, dtype=torch.float32)[:, None].expand(n, k) % 5) - 2),
    ("B = (k%5)-2 (mixed sign along k)", colk - 2),
    ("B = 0.5*(n%3) (fractions)", 0.5 * (torch.arange(n, device=dev, dtype=torch.float32)[:, None].expand(n, k) % 3)),
    ("B random", torch.randint(-2, 3, (n, k), device=dev).float()),
):
    D, R = run(ones_mk, Bpat, label)
    if not torch.equal(D, R):
        print("   D[0,:16]  ", D[0, :16].tolist())
        print("   ref[0,:16]", R[0, :16].tolist())
