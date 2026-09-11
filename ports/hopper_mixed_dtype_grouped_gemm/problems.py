"""Problem generation and reference for the example-69 port.

Problem-space conventions follow CUTLASS examples/69_hopper_mixed_dtype_grouped_gemm:

    D_g = alpha_g * (A_g @ dequant(B_g)^T) + beta_g * C_g        for each group g
    dequant(B)[n, k] = MmaType(B_q[n, k]) * scale[n, k // c]

  A_g      (M, K)       MmaType (bf16), row-major (K contiguous)
  B_g      (N, K)       QuantType (fp8 e5m2 / int4 / int8), column-major B == (N, K) K contiguous
  scale_g  (N, K / c)   bf16
  C_g, D_g (M, N)       fp16, row-major

The kernel runs the swapped/transposed problem D^T = dequant(B) @ A^T so the
quantized operand is wgmma's register-sourced A. In kernel coordinates:
kM = N, kN = M, and the output D^T (kM, kN) is kM-contiguous ("m-major").
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class Group:
    m: int
    n: int
    k: int
    alpha: float
    beta: float
    A: torch.Tensor  # (M, K) bf16
    Bq: torch.Tensor  # (N, K) quant dtype
    scale: torch.Tensor | None  # (N, K // c) bf16, or None for convert-only
    C: torch.Tensor  # (M, N) fp16
    D: torch.Tensor  # (M, N) fp16 output


def make_problems(
    shapes: list[tuple[int, int, int]],
    quant_dtype=torch.float8_e5m2,
    scale_c: int | None = 512,
    alpha: float | None = 1.0,
    beta: float | None = 0.0,
    seed: int = 2020,
    device: str = "cuda",
) -> list[Group]:
    """Random problems like the C++ example (uniform ints in small ranges, exact in bf16/fp8)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    rnd = random.Random(seed)
    groups = []
    for m, n, k in shapes:
        A = torch.randint(-2, 3, (m, k), generator=g).to(torch.bfloat16)
        Bq = torch.randint(-2, 3, (n, k), generator=g).to(torch.float32).to(quant_dtype)
        scale = None
        if scale_c is not None:
            assert k % scale_c == 0, "group-wise scaling needs K divisible by c"
            scale = (torch.randint(1, 5, (n, k // scale_c), generator=g).to(torch.float32) / 4).to(torch.bfloat16)
        C = torch.randint(-2, 3, (m, n), generator=g).to(torch.float16)
        a = float(rnd.randint(1, 5)) if alpha is None else alpha
        b = float(rnd.randint(0, 4)) if beta is None else beta
        groups.append(Group(m, n, k, a, b, A.to(device), Bq.to(device),
                            None if scale is None else scale.to(device), C.to(device),
                            torch.empty(m, n, dtype=torch.float16, device=device)))
    return groups


def random_shapes(groups: int, n: int = 2048, k: int = 512, alignment: int = 16, seed: int = 2020):
    """The example's default: M random in alignment * [1, 64], fixed N and K."""
    rnd = random.Random(seed)
    return [(alignment * rnd.randint(1, 64), n, k) for _ in range(groups)]


def dequant(gp: Group, mma_dtype=torch.bfloat16) -> torch.Tensor:
    """dequant(B) as the kernel computes it: convert to MmaType, multiply by scale in MmaType."""
    b = gp.Bq.to(torch.float32).to(mma_dtype)
    if gp.scale is None:
        return b
    c = gp.k // gp.scale.shape[1]
    return (b * gp.scale.repeat_interleave(c, dim=1)).to(mma_dtype)


def reference(gp: Group) -> torch.Tensor:
    """fp32-accumulated reference of D (M, N) in fp16."""
    acc = gp.A.to(torch.float32) @ dequant(gp).to(torch.float32).T
    return (gp.alpha * acc + gp.beta * gp.C.to(torch.float32)).to(torch.float16)


def flops(groups: list[Group]) -> int:
    return sum(2 * gp.m * gp.n * gp.k for gp in groups)
