# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# CuTeDSL port of CUTLASS C++ examples/69_hopper_mixed_dtype_grouped_gemm. The grouped
# machinery (per-group TMA descriptor updates, group-aware persistent scheduler) follows
# examples/python/CuTeDSL/cute/hopper/kernel/grouped_gemm/grouped_gemm.py (BSD-3-Clause).

"""Hopper mixed-input *grouped* GEMM in CuTeDSL.

For each group g (kernel coordinates, i.e. after the example's swap/transpose):

    D_g = alpha_g * (convert(A_g) * scale_g) @ B_g^T  +  beta_g * C_g

  A_g      (kM, K)       narrow QuantType, K-major           (problem-space B, the weights)
  scale_g  (kM, K / c)   bf16, kM-major (omitted for c = 0)
  B_g      (kN, K)       bf16, K-major                        (problem-space A, the activations)
  C_g, D_g (kM, kN)      fp16, kM-major                       (problem-space row-major M x N)

Per-group shapes, strides and pointers live in device arrays; each CTA keeps one TMA
descriptor per tensor in a global workspace and rewrites it when its tile stream crosses
into a new group (A, B, scale by the load warp; D by the epilogue store warp). The mainloop
(register-sourced WGMMA with in-register conversion) is inherited from
mixed_gemm.HopperMixedInputGemmKernel.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warpgroup
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from grouped_utils import (T_A, T_B, T_C, T_D, T_S, StaticPersistentGroupTileScheduler, group_tensor,
                           tma_load_2d_elected)
from mixed_gemm import _K_UNROLL, HopperMixedInputGemmKernel

import os

NUM_TENSORMAPS = 4  # A, B, scale, D
SCHED_DEPTH = 8  # tile records buffered between the scheduler warp and the tile consumers
SCHED_FIELDS = 8  # valid, group, m_idx, n_idx, M, N, K, k_tiles
_DEBUG_SCHED = os.environ.get("GROUPED_DEBUG_SCHED") == "1"  # printf tile-record traffic of CTA 0
BYTES_PER_TENSORMAP = 128


class HopperMixedInputGroupedGemmKernel(HopperMixedInputGemmKernel):
    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,  # dtype / majorness carriers (shapes irrelevant)
        initial_scale: cute.Tensor,
        initial_b: cute.Tensor,
        initial_d: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_mnkl: cute.Tensor,  # (G, 4) Int32, kernel coordinates
        strides: cute.Tensor,  # (G, 5, 2) Int32: (row stride, col stride) of A, B, S, D, C
        ptrs: cute.Tensor,  # (G, 5) Int64
        alpha: cute.Tensor,  # (G,) Float32
        beta: cute.Tensor,  # (G,) Float32
        total_num_tiles: cutlass.Constexpr[int],
        tensormaps: cute.Tensor,  # (num_ctas, NUM_TENSORMAPS, 16) Int64 workspace
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = initial_a.element_type
        self.c_dtype = initial_d.element_type
        self.b_layout = utils.LayoutEnum.from_tensor(initial_b)
        self.c_layout = utils.LayoutEnum.from_tensor(initial_d)
        if cutlass.const_expr(initial_b.element_type != self.mma_dtype):
            raise TypeError(f"B must be {self.mma_dtype}, got {initial_b.element_type}")
        # sCsrc (C source tile) + staged problem shapes + alignment slack
        tm0, tn0 = self.tile_shape_mnk[0], self.tile_shape_mnk[1]
        self.sched_warp_id = 1  # an otherwise idle warp of the DMA warpgroup
        self.extra_smem_bytes = (tm0 * tn0 * self.c_dtype.width // 8 + group_count * 16
                                 + SCHED_DEPTH * (SCHED_FIELDS * 4 + 16) + 512)
        self._setup_attributes()
        tm, tn, tk = self.tile_shape_mnk
        g2s = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, initial_a, cute.slice_(self.a_smem_layout_staged, (None, None, 0)), (tm, tk)
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, initial_b, cute.slice_(self.b_smem_layout_staged, (None, None, 0)), (tn, tk)
        )
        tma_atom_s, tma_tensor_s = cute.nvgpu.cpasync.make_tiled_tma_atom(
            g2s, initial_scale, cute.slice_(self.s_smem_layout_staged, (None, None, 0)), (tm, 1)
        )
        tma_atom_d, tma_tensor_d = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            initial_d,
            cute.slice_(self.epi_smem_layout_staged, (None, None, 0)),
            self.epi_tile,
        )
        tile_sched_params = utils.PersistentTileSchedulerParams(
            (1, 1, cutlass.Int32(total_num_tiles)), (1, 1, 1)
        )
        grid = StaticPersistentGroupTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)], self.buffer_align_bytes
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.mma_dtype, cute.cosize(self.b_smem_layout_staged)], self.buffer_align_bytes
            ]
            sS: cute.struct.Align[cute.struct.MemRange[self.mma_dtype, cute.cosize(self.s_smem_layout_staged)], 128]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.epi_smem_layout_staged)], self.buffer_align_bytes
            ]
            sCsrc: cute.struct.Align[cute.struct.MemRange[self.c_dtype, tm * tn], 128]  # C source tile
            # (G, 4) problem shapes, staged once: the group search reads them on every tile
            problem_shapes: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, group_count * 4], 16]
            sched_buf: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, SCHED_DEPTH * SCHED_FIELDS], 16]
            sched_barriers: cute.struct.MemRange[cutlass.Int64, SCHED_DEPTH * 2]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_atom_a, tma_tensor_a, tma_atom_s, tma_tensor_s, tma_atom_b, tma_tensor_b, tma_atom_d, tma_tensor_d,
            self.tiled_mma, tile_sched_params,
            self.a_smem_layout_staged, self.b_smem_layout_staged, self.s_smem_layout_staged,
            self.epi_smem_layout_staged,
            group_count, problem_shape_mnkl, strides, ptrs, alpha, beta, tensormaps,
        ).launch(grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(1, 1, 1), min_blocks_per_mp=1, stream=stream)

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        s_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        group_count: cutlass.Constexpr[int],
        problem_sizes_mnkl: cute.Tensor,
        strides: cute.Tensor,
        ptrs: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        tensormaps: cute.Tensor,
    ):
        tm, tn, tk = self.tile_shape_mnk
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, cute.slice_(a_smem_layout_staged, (None, None, 0)))
        tma_copy_bytes += cute.size_in_bytes(self.mma_dtype, cute.slice_(b_smem_layout_staged, (None, None, 0)))
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
        # Tile records: one scheduler warp runs the group search and publishes each tile's
        # (group, m, n, shape, k-tiles) here; the load warp and all MMA threads just read them
        # (as the C++ kernel does with its scheduler warp), instead of 9 warps each searching.
        sched_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.sched_barriers.data_ptr(),
            num_stages=SCHED_DEPTH,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 + self.num_mma_threads),
            defer_sync=True,
        )
        sbuf = storage.sched_buf.get_tensor(cute.make_layout((SCHED_FIELDS, SCHED_DEPTH), stride=(1, SCHED_FIELDS)))
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sS = storage.sS.get_tensor(s_smem_layout_staged)
        sC = storage.sC.get_tensor(epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner)

        # TMA coordinate tensors of the placeholder tensors: tile indices beyond their extents are
        # fine (coordinates only); the per-group descriptors carry the real shapes and addresses.
        gA_mkl = cute.local_tile(mA_mkl, (tm, tk), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, (tn, tk), (None, None, None))
        gS_mkl = cute.local_tile(mS_mkl, (tm, 1), (None, None, None))
        gD_mnl = cute.local_tile(mD_mnl, (tm, tn), (None, None, None))
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
        thr_mma = tiled_mma.get_slice(mma_tidx)
        tCsB = thr_mma.partition_B(sB)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCgD = thr_mma.partition_C(gD_mnl)
        accumulators = cute.make_rmem_tensor(tCgD.shape[:3], self.acc_dtype)
        k_tiles_per_scale = self.scale_granularity_k // tk if self.has_scale else 1
        cluster_tile_shape_mnk = (tm, tn, tk)

        grid_dim = cute.arch.grid_dim()
        bid = cute.arch.block_idx()
        cta_idx = bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]
        tmap_mgr = utils.TensorMapManager(utils.TensorMapUpdateMode.GMEM, BYTES_PER_TENSORMAP)
        tmap_a = tmap_mgr.get_tensormap_ptr(tensormaps[(cta_idx, 0, None)].iterator)
        tmap_b = tmap_mgr.get_tensormap_ptr(tensormaps[(cta_idx, 1, None)].iterator)
        tmap_s = tmap_mgr.get_tensormap_ptr(tensormaps[(cta_idx, 2, None)].iterator)
        tmap_d = tmap_mgr.get_tensormap_ptr(tensormaps[(cta_idx, 3, None)].iterator)

        # Stage the problem-shape table in smem. The group search runs at the start of every tile in
        # both the producer and the consumers; reading it from global memory put a dependent load on
        # each tile's critical path (C++ hides this behind a scheduler warp and an smem pipeline).
        sP = storage.problem_shapes.get_tensor(cute.make_layout((group_count, 4), stride=(4, 1)))
        sP_flat = cute.make_tensor(sP.iterator, cute.make_layout(group_count * 4))
        gP_flat = cute.make_tensor(problem_sizes_mnkl.iterator, cute.make_layout(group_count * 4))
        for i in cutlass.range(tidx, group_count * 4, self.threads_per_cta, unroll=1):
            sP_flat[i] = gP_flat[i]
        cute.arch.sync_threads()

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        # ---------------------------------------------------------------- scheduler warp
        if warp_idx == self.sched_warp_id:
            tile_sched = StaticPersistentGroupTileScheduler.create(
                tile_sched_params, bid, grid_dim, cluster_tile_shape_mnk, utils.create_initial_search_state(),
                group_count, sP,
            )
            work_tile = tile_sched.initial_work_tile_info()
            sched_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, SCHED_DEPTH)
            lane = cute.arch.lane_idx()
            while work_tile.is_valid_tile:
                gi = work_tile.group_search_result
                sched_pipeline.producer_acquire(sched_state)
                if lane == 0:
                    i = sched_state.index
                    sbuf[(0, i)] = cutlass.Int32(1)
                    sbuf[(1, i)] = gi.group_idx
                    sbuf[(2, i)] = gi.cta_tile_idx_m
                    sbuf[(3, i)] = gi.cta_tile_idx_n
                    sbuf[(4, i)] = gi.problem_shape_m
                    sbuf[(5, i)] = gi.problem_shape_n
                    sbuf[(6, i)] = gi.problem_shape_k
                    sbuf[(7, i)] = gi.cta_tile_count_k
                if cutlass.const_expr(_DEBUG_SCHED):
                    if lane == 0 and cta_idx == 0:
                        cute.printf("sched  publish slot=%d g=%d m=%d n=%d k=%d", sched_state.index,
                                    gi.group_idx, gi.cta_tile_idx_m, gi.cta_tile_idx_n, gi.cta_tile_count_k)
                sched_pipeline.producer_commit(sched_state)
                sched_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            sched_pipeline.producer_acquire(sched_state)  # terminator record
            if lane == 0:
                sbuf[(0, sched_state.index)] = cutlass.Int32(0)
            sched_pipeline.producer_commit(sched_state)
            sched_state.advance()
            sched_pipeline.producer_tail(sched_state)

        # ---------------------------------------------------------------- producer
        if warp_idx == self.load_warp_id:
            tmap_mgr.init_tensormap_from_atom(tma_atom_a, tmap_a, self.load_warp_id)
            tmap_mgr.init_tensormap_from_atom(tma_atom_b, tmap_b, self.load_warp_id)
            if cutlass.const_expr(self.has_scale):
                tmap_mgr.init_tensormap_from_atom(tma_atom_s, tmap_s, self.load_warp_id)
            tmap_mgr.fence_tensormap_initialization()
            last_group = cutlass.Int32(-1)
            sched_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, SCHED_DEPTH)
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
            valid, g, m_idx, n_idx, pm, pn, pk, k_cnt = self._sched_pop(sched_pipeline, sbuf, sched_state)
            sched_state.advance()
            if cutlass.const_expr(_DEBUG_SCHED):
                if (tidx == 0 or tidx == 128) and cta_idx == 0:
                    cute.printf("tid=%d pop valid=%d g=%d m=%d n=%d k=%d", tidx, valid, g, m_idx, n_idx, k_cnt)
            while valid != 0:
                if k_cnt > 0:
                    if g != last_group:
                        if cutlass.const_expr(self.has_scale):
                            k_groups = (pk + self.scale_granularity_k - 1) // self.scale_granularity_k
                            tmap_mgr.update_tensormap(
                                (
                                    group_tensor(g, self.a_dtype, pm, pk, strides,
                                                 ptrs, T_A),
                                    group_tensor(g, self.mma_dtype, pn, pk, strides,
                                                 ptrs, T_B),
                                    group_tensor(g, self.mma_dtype, pm, k_groups, strides, ptrs, T_S),
                                ),
                                (tma_atom_a, tma_atom_b, tma_atom_s),
                                (tmap_a, tmap_b, tmap_s),
                                self.load_warp_id,
                                (None, None, None),
                            )
                            tmap_mgr.fence_tensormap_update(tmap_s)
                        else:
                            tmap_mgr.update_tensormap(
                                (
                                    group_tensor(g, self.a_dtype, pm, pk, strides,
                                                 ptrs, T_A),
                                    group_tensor(g, self.mma_dtype, pn, pk, strides,
                                                 ptrs, T_B),
                                ),
                                (tma_atom_a, tma_atom_b),
                                (tmap_a, tmap_b),
                                self.load_warp_id,
                                (None, None),
                            )
                        tmap_mgr.fence_tensormap_update(tmap_a)
                        tmap_mgr.fence_tensormap_update(tmap_b)
                    desc_a = tmap_mgr.get_tensormap_ptr(tmap_a, cute.AddressSpace.gmem)
                    desc_b = tmap_mgr.get_tensormap_ptr(tmap_b, cute.AddressSpace.gmem)
                    desc_s = tmap_mgr.get_tensormap_ptr(tmap_s, cute.AddressSpace.gmem)
                    for k_tile in cutlass.range(0, k_cnt, 1, unroll=1):
                        mainloop_pipeline.producer_acquire(producer_state)
                        bar = mainloop_pipeline.producer_get_barrier(producer_state)
                        stage = producer_state.index
                        # (K, rows) coordinates for the K-major A/B boxes; (rows, k-group) for the scale
                        tma_load_2d_elected(tAsA[(None, stage)].iterator, desc_a, k_tile * tk, m_idx * tm, bar)
                        tma_load_2d_elected(tBsB[(None, stage)].iterator, desc_b, k_tile * tk, n_idx * tn, bar)
                        if cutlass.const_expr(self.has_scale):
                            tma_load_2d_elected(tSsS[(None, stage)].iterator, desc_s, m_idx * tm,
                                                k_tile // k_tiles_per_scale, bar)
                        mainloop_pipeline.producer_commit(producer_state)
                        producer_state.advance()
                last_group = g
                valid, g, m_idx, n_idx, pm, pn, pk, k_cnt = self._sched_pop(sched_pipeline, sbuf, sched_state)
                sched_state.advance()
                if cutlass.const_expr(_DEBUG_SCHED):
                    if (tidx == 0 or tidx == 128) and cta_idx == 0:
                        cute.printf("tid=%d pop valid=%d g=%d m=%d n=%d k=%d", tidx, valid, g, m_idx, n_idx, k_cnt)
            mainloop_pipeline.producer_tail(producer_state)

        # ---------------------------------------------------------------- consumers
        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            if warp_idx == self.epi_store_warp_id:
                tmap_mgr.init_tensormap_from_atom(tma_atom_d, tmap_d, self.epi_store_warp_id)
                tmap_mgr.fence_tensormap_initialization()

            s2r_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.a_dtype,
                                           num_bits_per_copy=2 * self.a_dtype.width)
            tiled_s2r_a = cute.make_tiled_copy_A(s2r_atom, tiled_mma)
            thr_s2r_a = tiled_s2r_a.get_slice(mma_tidx)
            tCsA_s2r = thr_s2r_a.partition_S(sA)
            tCsA_mma = thr_mma.partition_A(sA)
            tCrA = tiled_mma.make_fragment_A(tCsA_mma[(None, None, None, 0)])
            tCrA_q = cute.make_fragment_like(tCrA, self.a_dtype)
            tCrA_q_s2r = thr_s2r_a.retile(tCrA_q)
            num_k_blocks = cute.size(tCrA, mode=[2])
            sS_bcast = cute.make_tensor(sS.iterator, cute.make_layout((tm, tk, self.ab_stage), stride=(1, 0, tm)))
            tCsS = thr_mma.partition_A(sS_bcast)
            tCrS = cute.make_fragment_like(tCrA[(None, None, 0)])
            a_pre = self._precompute_a(sA, a_smem_layout_staged, thr_mma, num_k_blocks)
            a_q16 = cute.make_rmem_tensor((4, num_k_blocks), cutlass.Uint16)
            # C source: cp.async (not register loads - a register-destination load in flight would be
            # waited on by the mainloop's first wgmma.fence) of the kM-contiguous tile into smem, one
            # 16-byte chunk per consumer thread, read back in the epilogue along the accumulator layout.
            vec = 128 // self.c_dtype.width
            assert (tm // vec) * tn == self.num_mma_threads, "C prefetch maps one 16B chunk per consumer thread"
            sCsrc = storage.sCsrc.get_tensor(cute.make_layout((tm, tn), stride=(1, tm)))
            tiled_g2s_c = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), self.c_dtype, num_bits_per_copy=128),
                cute.make_layout((tm // vec, tn), stride=(1, tm // vec)),
                cute.make_layout((vec, 1)),
            )
            thr_g2s_c = tiled_g2s_c.get_slice(mma_tidx)
            tGsC = thr_g2s_c.partition_D(sCsrc)
            tCsCsrc = thr_mma.partition_C(sCsrc)
            c_frag = cute.make_rmem_tensor(accumulators.layout, self.c_dtype)

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

            sched_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, SCHED_DEPTH)
            read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            mma_in_flight = num_k_blocks - 1
            last_group = cutlass.Int32(-1)
            # Per-group epilogue parameters, carried across tiles and refreshed on a group change so
            # that no tile starts with a dependent global load (beta gates the C prefetch).
            alpha_g = cutlass.Float32(0.0)
            beta_g = cutlass.Float32(0.0)
            c_base = cutlass.Int64(0)
            c_s0 = cutlass.Int32(0)
            c_s1 = cutlass.Int32(0)
            tiles_done = cutlass.Int32(0)

            valid, g, m_idx, n_idx, pm, pn, pk, k_cnt = self._sched_pop(sched_pipeline, sbuf, sched_state)
            sched_state.advance()
            if cutlass.const_expr(_DEBUG_SCHED):
                if (tidx == 0 or tidx == 128) and cta_idx == 0:
                    cute.printf("tid=%d pop valid=%d g=%d m=%d n=%d k=%d", tidx, valid, g, m_idx, n_idx, k_cnt)
            while valid != 0:
                if g != last_group:
                    alpha_g = alpha[g]
                    beta_g = beta[g]
                    c_base = ptrs[(g, T_C)]
                    c_s0 = strides[(g, T_C, 0)]
                    c_s1 = strides[(g, T_C, 1)]
                    if warp_idx == self.epi_store_warp_id:
                        tmap_mgr.update_tensormap(
                            (group_tensor(g, self.c_dtype, pm, pn, strides, ptrs,
                                          T_D),),
                            (tma_atom_d,), (tmap_d,), self.epi_store_warp_id, (None,),
                        )
                        tmap_mgr.fence_tensormap_update(tmap_d)

                # Prefetch this tile's C source into smem (beta != 0); it lands during the mainloop.
                if beta_g != 0.0:
                    mC = cute.make_tensor(
                        cute.make_ptr(self.c_dtype, c_base, cute.AddressSpace.gmem, assumed_align=16),
                        # the host guarantees kM % 8 == 0 (fp8 TMA alignment), so columns start 16B-aligned
                        cute.make_layout((pm, pn),
                                         stride=(c_s0, cute.assume(c_s1, divby=128 // self.c_dtype.width))),
                    )
                    tGgC = thr_g2s_c.partition_S(cute.local_tile(mC, (tm, tn), (m_idx, n_idx)))
                    tGcC = thr_g2s_c.partition_S(cute.local_tile(
                        cute.make_identity_tensor((pm, pn)), (tm, tn), (m_idx, n_idx)
                    ))
                    # one chunk per thread; kM % 8 == 0, so a chunk is in bounds iff its first element is
                    if cute.elem_less(tGcC[0], (pm, pn)):
                        cute.copy(tiled_g2s_c, tGgC, tGsC)
                    cute.arch.cp_async_commit_group()

                read_state.reset_count()
                release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                if k_cnt > 0:
                    mainloop_pipeline.consumer_wait(read_state)
                    self._prepare_k_tile(tiled_s2r_a, tCsA_s2r, tCrA_q_s2r, tCsA_mma, tCsS, tCrS, tCrA_q, tCrA,
                                         read_state.index, a_pre, a_q16)
                    for k_tile in cutlass.range(k_cnt, unroll=_K_UNROLL):
                        stage = read_state.index
                        for kb in cutlass.range_constexpr(num_k_blocks):
                            warpgroup.fence()
                            cute.gemm(tiled_mma, accumulators, tCrA[(None, None, kb)], tCrB[(None, None, kb, stage)],
                                      accumulators)
                            warpgroup.commit_group()
                            warpgroup.wait_group(mma_in_flight)
                            if cutlass.const_expr(kb < num_k_blocks - 1):
                                if cutlass.const_expr(kb + 2 < num_k_blocks):
                                    self._stage_a(tiled_s2r_a, tCsA_s2r, tCrA_q_s2r, a_pre, a_q16, kb + 2, stage)
                                self._convert_a(tCrA_q, a_pre, a_q16, tCrS, tCrA, kb + 1)
                            else:
                                if k_tile > 0:
                                    mainloop_pipeline.consumer_release(release_state)
                                    release_state.advance()
                                read_state.advance()
                                if k_tile + 1 < k_cnt:
                                    mainloop_pipeline.consumer_wait(read_state)
                                    self._prepare_k_tile(tiled_s2r_a, tCsA_s2r, tCrA_q_s2r, tCsA_mma, tCsS, tCrS,
                                                         tCrA_q, tCrA, read_state.index, a_pre, a_q16)
                    warpgroup.wait_group(0)
                    mainloop_pipeline.consumer_release(release_state)
                    release_state.advance()

                # ---- epilogue: D = alpha * acc + beta * C  (fp32), -> fp16 -> smem -> TMA store
                if beta_g != 0.0:
                    cute.arch.cp_async_wait_group(0)
                    self.epilog_sync_barrier.arrive_and_wait()  # every thread's chunk is visible
                    c_frag.store(tCsCsrc.load())
                    accumulators.store(accumulators.load() * alpha_g + c_frag.load().to(self.acc_dtype) * beta_g)
                else:
                    accumulators.store(accumulators.load() * alpha_g)
                gD_tile = gD_mnl[(None, None, m_idx, n_idx, 0)]
                tDgD_epi = cute.zipped_divide(gD_tile, self.epi_tile)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_d, 0, one_cta, cute.group_modes(sC, 0, 2), tDgD_epi
                )
                epi_tile_num = cute.size(tDgD_epi, mode=[1])
                epi_tile_shape = tDgD_epi.shape[1]
                epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))
                num_prev_epi_tiles = tiles_done * epi_tile_num
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
                            tma_atom_d, bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, epi_tile_layout.get_hier_coord(epi_idx))],
                            tma_desc_ptr=tmap_mgr.get_tensormap_ptr(tmap_d, cute.AddressSpace.generic),
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                last_group = g
                tiles_done += 1
                valid, g, m_idx, n_idx, pm, pn, pk, k_cnt = self._sched_pop(sched_pipeline, sbuf, sched_state)
                sched_state.advance()
                if cutlass.const_expr(_DEBUG_SCHED):
                    if (tidx == 0 or tidx == 128) and cta_idx == 0:
                        cute.printf("tid=%d pop valid=%d g=%d m=%d n=%d k=%d", tidx, valid, g, m_idx, n_idx, k_cnt)
            tma_store_pipeline.producer_tail()

    def _sched_pop(self, sched_pipeline, sbuf, sched_state):
        """Wait for, read and release the current tile record published by the scheduler warp."""
        sched_pipeline.consumer_wait(sched_state)
        i = sched_state.index
        rec = tuple(sbuf[(f, i)] for f in range(SCHED_FIELDS))
        sched_pipeline.consumer_release(sched_state)
        # NOTE: the caller advances sched_state in the kernel body: the DSL detects loop-carried
        # state from the loop body's AST, so an advance() inside this helper would be lost.
        return rec
