# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Derived from CUTLASS examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py
# (BSD-3-Clause, see that file for the full license text). Modified for mixed-input
# (narrow A converted in registers, register-sourced WGMMA) as a CuTeDSL port of
# CUTLASS C++ examples/69_hopper_mixed_dtype_grouped_gemm.

"""Hopper mixed-input GEMM in CuTeDSL (single problem; grouping is layered on top).

Kernel-space problem (after the example's swap/transpose):

    C[m, n] = sum_k  convert(A[m, k]) * scale[m, k // c]  *  B[n, k]

  A      (kM, K)       narrow QuantType (Float8E5M2 / Float8E4M3FN / Int8), K-major
  scale  (kM, K // c)  MmaType (bf16), kM-major; omitted in convert-only mode
  B      (kN, K)       MmaType (bf16), K-major
  C      (kM, kN)      fp16, kM-major (it is D^T of a row-major problem-space D)

A is TMA-loaded to shared memory in its narrow type, copied to registers along
the WGMMA A-fragment layout, converted (and scaled) to MmaType there, and fed
to a register-sourced WGMMA. B stays in shared memory. The CTA is warp-
specialized: one DMA warpgroup (TMA producer) and two MMA warpgroups
(cooperative: each owns 64 of the 128 tile rows), persistent over tiles.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warpgroup
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def lds_u16(addr, imm: int = 0, *, loc=None, ip=None):
    """ld.shared.u16 at a 32-bit smem address plus an immediate byte offset.

    Side-effecting asm so it is never hoisted above the pipeline's barrier wait.
    """
    v = llvm.inline_asm(
        T.i16(), [cutlass.Int32(addr).ir_value(loc=loc, ip=ip)],
        f"ld.shared.u16 $0, [$1+{int(imm)}];", "=h,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Uint16(v)

import os

_DEBUG_LAYOUTS = os.environ.get("MIXED_GEMM_DEBUG_LAYOUTS") == "1"
# Ablation switches for performance investigation (results are wrong when set):
#   MIXED_GEMM_EXP=skip_transform   no smem->reg loads / conversion inside the mainloop
#   MIXED_GEMM_EXP=skip_convert     keep the smem loads, skip conversion + scale
#   MIXED_GEMM_EXP=skip_mma         no WGMMA (measures the data path alone)
_EXP = os.environ.get("MIXED_GEMM_EXP", "")
# Conversion: "ptx" = explicit fp8x2 PTX sequence on 16-bit pairs (see _dequant_fp8x2); "ssa" = TensorSSA .to()
_CONVERT = os.environ.get("MIXED_GEMM_CONVERT", "ptx")
# Unroll factor of the consumer k-tile loop (amortizes the per-k-tile boundary code).
_K_UNROLL = int(os.environ.get("MIXED_GEMM_K_UNROLL", "4"))
# A smem->register addressing: "precomputed" (per-thread swizzled offsets computed once; loads use
# a per-stage base + immediates) or "cute" (CuTe partition; swizzle recomputed per load).
_A_ADDR = os.environ.get("MIXED_GEMM_A_ADDR", "precomputed")
_STAGES = int(os.environ.get("MIXED_GEMM_STAGES", "0"))  # cap on pipeline stages (0 = fill smem)


class HopperMixedInputGemmKernel:
    def __init__(
        self,
        mma_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        tile_shape_mn: tuple[int, int] = (128, 16),
        scale_granularity_k: int = 0,  # 0 = convert-only; else scale group size c along K
        swizzle_size: int = 1,
        raster_along_m: bool = False,
    ):
        self.mma_dtype = mma_dtype
        self.acc_dtype = acc_dtype
        self.scale_granularity_k = scale_granularity_k
        self.has_scale = scale_granularity_k > 0
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.cluster_shape_mn = (1, 1)
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        # Cooperative: two MMA warpgroups split the 128 tile rows (64 each), like the C++ kernel.
        self.atom_layout_mnk = (2, 1, 1) if tile_shape_mn[0] == 128 else (1, 1, 1)
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.threads_per_warp_group = 128
        self.threads_per_cta = (self.num_dma_warp_groups + self.num_mma_warp_groups) * 128
        self.num_mma_threads = self.num_mma_warp_groups * 128
        self.load_warp_id = 0
        self.epi_store_warp_id = self.num_dma_warp_groups * 4
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.num_mma_threads)

    # ------------------------------------------------------------------ host-side setup
    def _setup_attributes(self):
        tm, tn, _ = self.tile_shape_mnk
        if tm not in (64, 128):
            raise ValueError("CTA tile M must be 64 or 128")
        if tn % 8 != 0 or not 8 <= tn <= 256:
            raise ValueError("CTA tile N must be a multiple of 8 in [8, 256]")
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.mma_dtype,
            self.mma_dtype,
            warpgroup.OperandMajorMode.K,  # register A is always K-major
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, tn),
            a_source=warpgroup.OperandSource.RMEM,
        )
        mma_inst_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])  # 16 for 16-bit MMA types
        self.tile_shape_mnk = (tm, tn, mma_inst_k * 4)  # K tile of 64 elements, as in the C++ example
        tk = self.tile_shape_mnk[2]
        if self.has_scale and self.scale_granularity_k % tk != 0:
            raise ValueError(f"scale group size {self.scale_granularity_k} must be a multiple of tile K {tk}")

        self.epi_tile = (min(128, tm), min(32, tn))
        self.ab_stage, self.epi_stage = self._compute_stages()
        self._make_smem_layouts()

    def _compute_stages(self):
        tm, tn, tk = self.tile_shape_mnk
        a_bytes = tm * tk * self.a_dtype.width // 8
        b_bytes = tn * tk * self.mma_dtype.width // 8
        s_bytes = tm * self.mma_dtype.width // 8 if self.has_scale else 0
        epi_stage = 4
        epi_bytes = cute.size(self.epi_tile) * self.c_dtype.width // 8 * epi_stage
        # alignment padding: each operand buffer is 1024B-aligned
        per_stage = a_bytes + b_bytes + s_bytes
        extra = getattr(self, "extra_smem_bytes", 0)  # buffers a subclass adds to SharedStorage
        ab_stage = (self.smem_capacity - 1024 - epi_bytes - extra - 3 * self.buffer_align_bytes) // per_stage
        if _STAGES:
            ab_stage = min(ab_stage, _STAGES)
        return min(ab_stage, 32), epi_stage

    def _make_smem_layouts(self):
        tm, tn, tk = self.tile_shape_mnk
        a_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.ROW_MAJOR, self.a_dtype, tk), self.a_dtype
        )
        self.a_smem_layout_staged = cute.tile_to_shape(a_atom, (tm, tk, self.ab_stage), order=(0, 1, 2))
        b_is_k_major = self.b_layout.sm90_mma_major_mode() == warpgroup.OperandMajorMode.K
        b_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(self.b_layout, self.mma_dtype, tk if b_is_k_major else tn),
            self.mma_dtype,
        )
        self.b_smem_layout_staged = cute.tile_to_shape(
            b_atom, (tn, tk, self.ab_stage), order=(0, 1, 2) if b_is_k_major else (1, 0, 2)
        )
        # One scale per tile row per stage (c >= tile K, so a k-tile never straddles scale groups).
        self.s_smem_layout_staged = cute.make_layout((tm, 1, self.ab_stage), stride=(1, tm, tm))
        c_major_size = self.epi_tile[1] if self.c_layout.is_n_major_c() else self.epi_tile[0]
        c_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(self.c_layout, self.c_dtype, c_major_size), self.c_dtype
        )
        self.epi_smem_layout_staged = cute.tile_to_shape(
            c_atom, (*self.epi_tile, self.epi_stage), order=(1, 0, 2) if self.c_layout.is_m_major_c() else (0, 1, 2)
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        scale: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.c_dtype = c.element_type
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        if cutlass.const_expr(b.element_type != self.mma_dtype):
            raise TypeError(f"B must be {self.mma_dtype}, got {b.element_type}")
        self._setup_attributes()
        tm, tn, tk = self.tile_shape_mnk

        g2s = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, a, cute.slice_(self.a_smem_layout_staged, (None, None, 0)), (tm, tk)
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, b, cute.slice_(self.b_smem_layout_staged, (None, None, 0)), (tn, tk)
        )
        tma_atom_s, tma_tensor_s = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, scale, cute.slice_(self.s_smem_layout_staged, (None, None, 0)), (tm, 1)
        )
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            c,
            cute.slice_(self.epi_smem_layout_staged, (None, None, 0)),
            self.epi_tile,
        )

        gc = cute.zipped_divide(c, tiler=(tm, tn))
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, (1, 1, 1), self.swizzle_size, self.raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)], self.buffer_align_bytes
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.mma_dtype, cute.cosize(self.b_smem_layout_staged)], self.buffer_align_bytes
            ]
            sS: cute.struct.Align[
                cute.struct.MemRange[self.mma_dtype, cute.cosize(self.s_smem_layout_staged)], 128
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.epi_smem_layout_staged)], self.buffer_align_bytes
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_atom_a, tma_tensor_a, tma_atom_s, tma_tensor_s, tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c, self.tiled_mma, tile_sched_params,
            self.a_smem_layout_staged, self.b_smem_layout_staged, self.s_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(1, 1, 1), min_blocks_per_mp=1, stream=stream)

    # ------------------------------------------------------------------ device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        s_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        tm, tn, tk = self.tile_shape_mnk
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)
            if cutlass.const_expr(self.has_scale):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_s)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(
            self.mma_dtype, b_smem_layout
        )
        if cutlass.const_expr(self.has_scale):
            tma_copy_bytes += tm * self.mma_dtype.width // 8

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warp_groups * 4),
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sS = storage.sS.get_tensor(s_smem_layout_staged)
        sC = storage.sC.get_tensor(epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner)

        # (bM, bK, RestM, RestK, RestL) etc.
        gA_mkl = cute.local_tile(mA_mkl, (tm, tk), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, (tn, tk), (None, None, None))
        gS_mkl = cute.local_tile(mS_mkl, (tm, 1), (None, None, None))
        gC_mnl = cute.local_tile(mC_mnl, (tm, tn), (None, None, None))

        one_cta = cute.make_layout(1)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, 0, one_cta, cute.group_modes(sA, 0, 2), cute.group_modes(gA_mkl, 0, 2)
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, 0, one_cta, cute.group_modes(sB, 0, 2), cute.group_modes(gB_nkl, 0, 2)
        )
        tSsS, tSgS = cute.nvgpu.cpasync.tma_partition(
            tma_atom_s, 0, one_cta, cute.group_modes(sS, 0, 2), cute.group_modes(gS_mkl, 0, 2)
        )

        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.threads_per_warp_group)
        mma_tidx = tidx - self.num_dma_warp_groups * self.threads_per_warp_group
        # Per-thread slice: register-sourced A is partitioned per thread.
        thr_mma = tiled_mma.get_slice(mma_tidx)
        tCsB = thr_mma.partition_B(sB)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCgC = thr_mma.partition_C(gC_mnl)
        accumulators = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        k_tiles_per_scale = self.scale_granularity_k // tk if self.has_scale else 1

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        # ---------------------------------------------------------------- producer
        if warp_idx == self.load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
            while work_tile.is_valid_tile:
                mc = work_tile.tile_idx
                tAgA_k_tiles = tAgA[(None, mc[0], None, mc[2])]
                tBgB_k_tiles = tBgB[(None, mc[1], None, mc[2])]
                tSgS_groups = tSgS[(None, mc[0], None, mc[2])]
                producer_state.reset_count()
                for k_tile in range(k_tile_cnt):
                    mainloop_pipeline.producer_acquire(producer_state)
                    bar = mainloop_pipeline.producer_get_barrier(producer_state)
                    kt = producer_state.count
                    stage = producer_state.index
                    cute.copy(tma_atom_a, tAgA_k_tiles[(None, kt)], tAsA[(None, stage)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_b, tBgB_k_tiles[(None, kt)], tBsB[(None, stage)], tma_bar_ptr=bar)
                    if cutlass.const_expr(self.has_scale):
                        cute.copy(
                            tma_atom_s, tSgS_groups[(None, kt // k_tiles_per_scale)], tSsS[(None, stage)],
                            tma_bar_ptr=bar,
                        )
                    mainloop_pipeline.producer_commit(producer_state)
                    producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(producer_state)

        # ---------------------------------------------------------------- consumers
        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            # smem -> rmem copy of narrow A along the WGMMA A-fragment layout
            # Copy exactly the two K-adjacent narrow values the A fragment pairs up (16 bits for
            # 8-bit types): each load is then one ready-made e5m2x2 / int8x2 conversion input, as in
            # the C++ collective (4x LDS.U16 per k-block). Wider auto-vectorized loads bring bytes
            # in an order that costs a byte-permute per value to untangle.
            s2r_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.a_dtype, num_bits_per_copy=2 * self.a_dtype.width
            )
            tiled_s2r_a = cute.make_tiled_copy_A(s2r_atom, tiled_mma)
            thr_s2r_a = tiled_s2r_a.get_slice(mma_tidx)
            tCsA_s2r = thr_s2r_a.partition_S(sA)  # (CPY, CPY_M, CPY_K, STAGE)
            tCsA_mma = thr_mma.partition_A(sA)  # (MMA, MMA_M, MMA_K, STAGE) - shape template
            # The register fragment must use the MMA's canonical layout: make_fragment_like on the
            # swizzled smem partition would copy its stride order, and the register-sourced WGMMA
            # packs values assuming the canonical order. The narrow staging fragment mirrors it.
            tCrA = tiled_mma.make_fragment_A(tCsA_mma[(None, None, None, 0)])
            tCrA_q = cute.make_fragment_like(tCrA, self.a_dtype)
            tCrA_q_s2r = thr_s2r_a.retile(tCrA_q)
            num_k_blocks = cute.size(tCrA, mode=[2])
            if cutlass.const_expr(_DEBUG_LAYOUTS):
                print("tCsA_mma      ", tCsA_mma.layout)
                print("tCrA          ", tCrA.layout, tCrA.element_type)
                print("tCsA_s2r      ", tCsA_s2r.layout)
                print("tCrA_q_s2r    ", tCrA_q_s2r.layout)
            # Scale broadcast along K, partitioned exactly like A so it multiplies elementwise.
            sS_bcast = cute.make_tensor(sS.iterator, cute.make_layout((tm, tk, self.ab_stage), stride=(1, 0, tm)))
            tCsS = thr_mma.partition_A(sS_bcast)  # (MMA, MMA_M, MMA_K, STAGE)

            # epilogue partitions
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout, elem_ty_d=self.c_dtype, elem_ty_acc=self.acc_dtype
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4), self.c_dtype
            )
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma))
            thr_copy_r2s = tiled_copy_r2s.get_slice(mma_tidx)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD = cute.make_rmem_tensor(cute.make_layout(rD_shape[:3]).shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(cute.make_layout(rD_shape[:3]).shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_threads),
            )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            # Per-thread scale registers: the scale is constant along K within a k-tile, so each
            # thread holds its rows' values once per tile (the C++ loads it at k-block 0).
            tCrS = cute.make_fragment_like(tCrA[(None, None, 0)])
            # Narrow-A staging: either the CuTe fp8 fragment (tCrA_q) or, with precomputed
            # addressing, 16-bit pairs in a (pairs, k-blocks) u16 register tensor.
            a_pre = self._precompute_a(sA, a_smem_layout_staged, thr_mma, num_k_blocks)
            a_q16 = cute.make_rmem_tensor((4, num_k_blocks), cutlass.Uint16)
            # (Transform steps are methods taking their tensors explicitly: CuTeDSL can't capture traced
            # values in closures used inside staged loops, and it only sees loop-carried state that the
            # loop body itself reassigns.)

            # Schedule (the C++ collective's): each k-block has its own A register slice; loads run
            # two k-blocks ahead and conversion one ahead of the MMA. After committing k-block j,
            # wait_group(num_k_blocks - 1) guarantees the MMA that last read slice (j + 1) has
            # finished, so it can be refilled while the other MMAs run. A smem stage is released
            # one k-tile later, when every MMA reading its B has completed.
            mma_in_flight = num_k_blocks - 1

            while work_tile.is_valid_tile:
                mc = work_tile.tile_idx
                gC_tile = gC_mnl[(None, None, *mc)]
                read_state.reset_count()
                release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)

                mainloop_pipeline.consumer_wait(read_state)
                self._prepare_k_tile(tiled_s2r_a, tCsA_s2r, tCrA_q_s2r, tCsA_mma, tCsS, tCrS, tCrA_q, tCrA,
                                     read_state.index, a_pre, a_q16)
                for k_tile in cutlass.range(k_tile_cnt, unroll=_K_UNROLL):
                    stage = read_state.index
                    for kb in cutlass.range_constexpr(num_k_blocks):
                        if cutlass.const_expr(_EXP != "skip_mma"):
                            warpgroup.fence()
                            cute.gemm(
                                tiled_mma, accumulators, tCrA[(None, None, kb)], tCrB[(None, None, kb, stage)],
                                accumulators,
                            )
                            warpgroup.commit_group()
                            warpgroup.wait_group(mma_in_flight)
                        if cutlass.const_expr(kb < num_k_blocks - 1):
                            if cutlass.const_expr(kb + 2 < num_k_blocks and _EXP != "skip_transform"):
                                self._stage_a(tiled_s2r_a, tCsA_s2r, tCrA_q_s2r, a_pre, a_q16, kb + 2, stage)
                            if cutlass.const_expr(_EXP not in ("skip_transform", "skip_convert")):
                                self._convert_a(tCrA_q, a_pre, a_q16, tCrS, tCrA, kb + 1)
                        else:
                            if k_tile > 0:
                                mainloop_pipeline.consumer_release(release_state)
                                release_state.advance()
                            read_state.advance()
                            if k_tile + 1 < k_tile_cnt:
                                mainloop_pipeline.consumer_wait(read_state)
                                self._prepare_k_tile(tiled_s2r_a, tCsA_s2r, tCrA_q_s2r, tCsA_mma, tCsS, tCrS,
                                                     tCrA_q, tCrA, read_state.index, a_pre, a_q16)

                warpgroup.wait_group(0)
                mainloop_pipeline.consumer_release(release_state)
                release_state.advance()

                # ---- epilogue: acc -> fp16 -> smem -> TMA store
                tCgC_epi = cute.zipped_divide(gC_tile, self.epi_tile)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c, 0, one_cta, cute.group_modes(sC, 0, 2), tCgC_epi
                )
                epi_tile_num = cute.size(tCgC_epi, mode=[1])
                epi_tile_shape = tCgC_epi.shape[1]
                epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))
                num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num
                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    for v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[v] = tRS_rAcc[epi_idx * size_tRS_rD + v]
                    tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))
                    epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(tRS_sD, mode=[3])
                    cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    if warp_idx == self.epi_store_warp_id:
                        cute.copy(
                            tma_atom_c, bSG_sD[(None, epi_buffer)], bSG_gD[(None, epi_tile_layout.get_hier_coord(epi_idx))]
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            tma_store_pipeline.producer_tail()

    # ------------------------------------------------------------------ transform steps
    def _load_a(self, tiled_s2r_a, tCsA_s2r, tCrA_q_s2r, kb, stage):
        """Narrow A k-block: smem -> registers."""
        cute.copy(tiled_s2r_a, tCsA_s2r[(None, None, kb, stage)], tCrA_q_s2r[(None, None, kb)])

    def _load_scale(self, tCsS, tCrS, stage):
        if cutlass.const_expr(self.has_scale):
            tCrS.store(tCsS[(None, None, 0, stage)].load())

    def _dequant(self, tCrA_q, tCrS, tCrA, kb):
        """Narrow -> MmaType (times the scale, in MmaType) into register slice kb."""
        if cutlass.const_expr(self._use_fp8x2_ptx()):
            self._convert_fp8x2(cute.recast_tensor(tCrA_q[(None, None, kb)], cutlass.Uint16), tCrS, tCrA, kb)
            return
        a_val = tCrA_q[(None, None, kb)].load().to(self.mma_dtype)
        if cutlass.const_expr(self.has_scale):
            a_val = a_val * tCrS.load()
        tCrA[(None, None, kb)].store(a_val)

    def _prepare_k_tile(self, tiled_s2r_a, tCsA_s2r, tCrA_q_s2r, tCsA_mma, tCsS, tCrS, tCrA_q, tCrA, stage,
                        a_pre, a_q16):
        """First steps of a k-tile: its scale, and k-block 0 converted (plus k-block 1 staged)."""
        self._load_scale(tCsS, tCrS, stage)
        self._stage_a(tiled_s2r_a, tCsA_s2r, tCrA_q_s2r, a_pre, a_q16, 0, stage)
        self._stage_a(tiled_s2r_a, tCsA_s2r, tCrA_q_s2r, a_pre, a_q16, 1, stage)
        self._convert_a(tCrA_q, a_pre, a_q16, tCrS, tCrA, 0)

    def _use_fp8x2_ptx(self):
        return (_CONVERT == "ptx" and self.a_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN)
                and self.mma_dtype == cutlass.BFloat16)

    def _convert_fp8x2(self, src, tCrS, tCrA, kb):
        """fp8 -> bf16 (x scale) two values at a time, as the C++ NumericArrayConverter does.

        `src` is the k-block's narrow fragment viewed as u16: the A fragment pairs two K-adjacent
        values per 32-bit register, and those two bytes are adjacent in smem, so each pair arrives
        via one 16-bit smem load and converts with one cvt.f16x2.<fp8>x2. fp16 ->
        bf16 goes through fp32 (exact), and the row scale multiplies in bf16x2. The scale fragment
        has A's layout, so viewed as u32 each element is the (s, s) broadcast for its pair.
        """
        fmt = "e5m2x2" if self.a_dtype == cutlass.Float8E5M2 else "e4m3x2"
        dst = cute.recast_tensor(tCrA[(None, None, kb)], cutlass.Uint32)
        scl = cute.recast_tensor(tCrS, cutlass.Uint32)
        body = (
            "{ .reg .b32 h2; .reg .b16 lo, hi; .reg .f32 flo, fhi;\n"
            f"  cvt.rn.f16x2.{fmt} h2, {{$r0}};\n"
            "  mov.b32 {lo, hi}, h2;\n"
            "  cvt.f32.f16 flo, lo;\n"
            "  cvt.f32.f16 fhi, hi;\n"
        )
        for i in range(cute.size(dst)):  # plain Python: unrolled at trace time
            if cutlass.const_expr(self.has_scale):
                dst[i] = cute.arch.inline_ptx(
                    body + "  cvt.rn.bf16x2.f32 h2, fhi, flo;\n  mul.rn.bf16x2 {$w0}, h2, {$r1};\n}",
                    write_only_types=[cutlass.Uint32], read_only_args=[src[i], scl[i]],
                )
            else:
                dst[i] = cute.arch.inline_ptx(
                    body + "  cvt.rn.bf16x2.f32 {$w0}, fhi, flo;\n}",
                    write_only_types=[cutlass.Uint32], read_only_args=[src[i]],
                )

    # ------------------------------------------------------------------ precomputed A addressing
    def _precompute_a(self, sA, a_smem_layout_staged, thr_mma, num_k_blocks):
        """Per-thread swizzled smem offsets for the narrow A pairs, computed once per kernel.

        Returns None (use the CuTe partition) unless the fp8x2 PTX conversion is active and the
        layout has the structure the shortcut relies on. Otherwise returns
        (smem base address, stage bytes, per-k-block swizzled offsets, per-pair immediates).

        Why it is exact: a thread's pre-swizzle offset o0 comes from partitioning an *unswizzled*
        view of the same layout; the swizzle Swizzle<B,M,S> is o ^ ((o & ymask) >> S). The stage
        stride and the row+8 pair delta are multiples of 2^(B+M+S), so they never touch the
        swizzle's source or target bits; the k+8 pair delta stays below 2^M together with the
        thread's own k offset (< 8 bytes), so it never carries into them. Hence
        swizzle(o0 + kb + stage + pair) = swizzle(o0 + kb) + stage + pair.
        """
        if cutlass.const_expr(not (_A_ADDR == "precomputed" and self._use_fp8x2_ptx())):
            return None
        outer, sw = a_smem_layout_staged.outer, a_smem_layout_staged.inner
        B, M, S = int(sw.num_bits), int(sw.num_base), int(sw.num_shift)
        elem_bytes = self.a_dtype.width // 8
        plain = cute.make_tensor(cute.make_ptr(self.a_dtype, 0, cute.AddressSpace.smem, assumed_align=1024), outer)
        tp = thr_mma.partition_A(plain)  # (MMA, MMA_M, MMA_K, STAGE)
        lay = tp.layout
        off = lambda crd: int(cute.crd2idx(crd, lay)) * elem_bytes
        stage_bytes = off(((0, 0, 0), 0, 0, 1))
        kb_bytes = [off(((0, 0, 0), 0, kb, 0)) for kb in range(num_k_blocks)]
        pair_d = [off(((0, v1, v2), 0, 0, 0)) for v2 in (0, 1) for v1 in (0, 1)]  # recast order: v1 fastest
        top = 1 << (B + M + S)
        # each pair delta = (multiple of 2^(B+M+S)) + (low part that, with the thread's own k offset
        # of < 8 bytes, stays below 2^M); the pair itself must be two adjacent elements
        ok = (stage_bytes % top == 0 and all((d % top) + 8 <= (1 << M) for d in pair_d)
              and off(((1, 0, 0), 0, 0, 0)) == elem_bytes)
        if cutlass.const_expr(_DEBUG_LAYOUTS):
            print(f"precomputed A addressing: ok={ok} swizzle=({B},{M},{S}) stage={stage_bytes} kb={kb_bytes} pairs={pair_d}")
        if cutlass.const_expr(not ok):
            return None
        ymask = ((1 << B) - 1) << (M + S)
        o0 = cutlass.Int32(tp[(None, None, None, 0)].iterator.toint())
        base = cutlass.Int32(sA.iterator.toint())
        kb_off = []
        for kb in range(num_k_blocks):
            o = o0 + kb_bytes[kb]
            kb_off.append(o ^ ((o & ymask) >> S))
        return (base, stage_bytes, kb_off, pair_d)

    def _stage_a(self, tiled_s2r_a, tCsA_s2r, tCrA_q_s2r, a_pre, a_q16, kb, stage):
        """Stage narrow A k-block kb from smem into registers."""
        if cutlass.const_expr(a_pre is None):
            self._load_a(tiled_s2r_a, tCsA_s2r, tCrA_q_s2r, kb, stage)
        else:
            base, stage_bytes, kb_off, pair_d = a_pre
            addr = base + stage * stage_bytes + kb_off[kb]
            for p in range(4):
                a_q16[p, kb] = lds_u16(addr, pair_d[p])

    def _convert_a(self, tCrA_q, a_pre, a_q16, tCrS, tCrA, kb):
        if cutlass.const_expr(a_pre is None):
            self._dequant(tCrA_q, tCrS, tCrA, kb)
        else:
            self._convert_fp8x2(a_q16[(None, kb)], tCrS, tCrA, kb)
