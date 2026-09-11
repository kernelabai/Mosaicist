# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# The scheduler wrapper below is copied from CUTLASS
# examples/python/CuTeDSL/cute/hopper/kernel/grouped_gemm/grouped_gemm.py (BSD-3-Clause),
# where it is defined locally because it is not yet part of cutlass.utils.

"""Grouped-GEMM scheduling and per-group tensor construction for the example-69 port."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cutlass_dsl import extract_mlir_values, new_from_mlir_values


class _GroupedWorkTileInfo:
    """Work tile info for grouped GEMM: carries is_valid_tile + group_search_result."""

    def __init__(self, is_valid_tile, group_search_result):
        self._is_valid_tile = is_valid_tile
        self.group_search_result = group_search_result

    @property
    def is_valid_tile(self):
        return self._is_valid_tile

    def __extract_mlir_values__(self):
        values = extract_mlir_values(self._is_valid_tile)
        values.extend(extract_mlir_values(self.group_search_result))
        return values

    def __new_from_mlir_values__(self, values):
        n_valid = len(extract_mlir_values(self._is_valid_tile))
        is_valid = new_from_mlir_values(self._is_valid_tile, values[:n_valid])
        gsr = new_from_mlir_values(self.group_search_result, values[n_valid:])
        return _GroupedWorkTileInfo(is_valid, gsr)


class StaticPersistentGroupTileScheduler:
    """Grouped-GEMM-aware persistent tile scheduler (StaticPersistentTileScheduler +
    GroupedGemmTileSchedulerHelper): walks the linearized tiles of all groups."""

    def __init__(self, tile_sched, group_helper, problem_sizes_mnkl):
        self._tile_sched = tile_sched
        self._group_helper = group_helper
        self._problem_sizes_mnkl = problem_sizes_mnkl

    def __extract_mlir_values__(self):
        values = extract_mlir_values(self._tile_sched)
        values.extend(extract_mlir_values(self._group_helper))
        return values

    def __new_from_mlir_values__(self, values):
        n_tile = len(extract_mlir_values(self._tile_sched))
        tile_sched = new_from_mlir_values(self._tile_sched, values[:n_tile])
        group_helper = new_from_mlir_values(self._group_helper, values[n_tile:])
        return StaticPersistentGroupTileScheduler(tile_sched, group_helper, self._problem_sizes_mnkl)

    @staticmethod
    def create(tile_sched_params, bid, grid_dim, cluster_tile_shape_mnk, search_state, group_count,
               problem_sizes_mnkl):
        tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, bid, grid_dim)
        group_helper = utils.GroupedGemmTileSchedulerHelper(
            group_count, tile_sched_params, cluster_tile_shape_mnk, search_state
        )
        return StaticPersistentGroupTileScheduler(tile_sched, group_helper, problem_sizes_mnkl)

    @staticmethod
    def get_grid_shape(tile_sched_params, max_active_clusters):
        return utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

    def initial_work_tile_info(self):
        return self.get_current_work()

    def get_current_work(self):
        base = self._tile_sched.get_current_work()
        # Invalid tiles would make delinearize_z loop forever: clamp their z to 0 (always valid);
        # the result is only used under `while work_tile.is_valid_tile`.
        valid_int = base.is_valid_tile.to(cutlass.Int32)
        safe_tile_idx = (base.tile_idx[0], base.tile_idx[1], base.tile_idx[2] * valid_int)
        gsr = self._group_helper.delinearize_z(safe_tile_idx, self._problem_sizes_mnkl)
        return _GroupedWorkTileInfo(base.is_valid_tile, gsr)

    def advance_to_next_work(self, *, advance_count=1):
        self._tile_sched.advance_to_next_work(advance_count=advance_count)

    @property
    def num_tiles_executed(self):
        return self._tile_sched.num_tiles_executed


# Order of per-group tensors in the metadata arrays.
T_A, T_B, T_S, T_D, T_C = range(5)
NUM_TENSORS = 5


def group_tensor(group_idx, dtype, rows, cols, strides, ptrs, which):
    """Global tensor (rows, cols, 1) of group `group_idx` from the (G, 5, 2) Int32 stride array
    and the (G, 5) Int64 pointer array. strides[g, which] = (stride of rows, stride of cols)."""
    ptr = cute.make_ptr(dtype, ptrs[(group_idx, which)], cute.AddressSpace.gmem, assumed_align=16)
    s_reg = cute.make_rmem_tensor(cute.make_layout(2), strides.element_type)
    cute.autovec_copy(strides[(group_idx, which, None)], s_reg)
    return cute.make_tensor(
        ptr, cute.make_layout((rows, cols, cutlass.Int32(1)), stride=(s_reg[0], s_reg[1], cutlass.Int32(0)))
    )


# ----------------------------------------------------------------------------------------------
# TMA loads through a per-CTA global-memory descriptor, issued with an elect_sync *predicate*.
#
# Adapted from CUTLASS examples/python/CuTeDSL/cute/hopper/kernel/grouped_gemm/grouped_gemm.py
# (BSD-3-Clause). Issuing `cute.copy(..., tma_desc_ptr=...)` inside the elect-one `if` lets ptxas
# sink an R2UR into the predicated block, producing an illegal "@P0 R2UR" on sm_90a
# (CUDA_ERROR_ILLEGAL_INSTRUCTION, 715). Passing the predicate to the NVVM TMA op keeps every
# operand computation unconditional.
from cutlass._mlir.dialects import llvm as _llvm_d  # noqa: E402
from cutlass._mlir.dialects import nvvm as _nvvm_d  # noqa: E402
from cutlass.cute.core import AddressSpace as _CuteAddressSpace  # noqa: E402
from cutlass.cutlass_dsl import dsl_user_op as _dsl_user_op  # noqa: E402


def _tma_load_nvvm(dst_smem, desc_generic, coords, mbar, predicate, loc=None, ip=None):
    if hasattr(_nvvm_d, "TMALoadMode"):
        _nvvm_d.CpAsyncBulkTensorGlobalToSharedClusterOp(
            dst_smem, desc_generic, coords, mbar, [], predicate=predicate,
            mode=_nvvm_d.TMALoadMode.TILE, loc=loc, ip=ip,
        )
    else:
        _nvvm_d.CpAsyncBulkTensorGlobalToSharedClusterOp(
            dst_smem, desc_generic, coords, mbar, [], predicate=predicate,
            loadMode=_nvvm_d.CpAsyncBulkTensorLoadMode.TILE, loc=loc, ip=ip,
        )


@_dsl_user_op
def tma_load_2d_elected(smem_ptr, desc_gmem_ptr, c0, c1, mbar, *, loc=None, ip=None):
    """One elected thread of the warp issues a 3-D (c0, c1, 0) TMA tile load (L = 1)."""
    desc = _llvm_d.addrspacecast(
        _llvm_d.PointerType.get(_CuteAddressSpace.generic), desc_gmem_ptr.llvm_ptr, loc=loc, ip=ip
    )
    coords = [cutlass.Int32(c0).ir_value(loc=loc, ip=ip), cutlass.Int32(c1).ir_value(loc=loc, ip=ip),
              cutlass.Int32(0).ir_value(loc=loc, ip=ip)]
    elected = _nvvm_d.elect_sync(loc=loc, ip=ip)
    _tma_load_nvvm(smem_ptr.llvm_ptr, desc, coords, mbar.llvm_ptr, elected, loc=loc, ip=ip)
