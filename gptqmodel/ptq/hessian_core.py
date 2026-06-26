# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shared Hessian accumulation helpers used by PTQ stats and legacy GPTQ."""

from __future__ import annotations

import contextlib
import threading
from typing import Dict, Optional, Tuple

import torch

from ..utils.torch import torch_sync

_WORKSPACE_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}
_WORKSPACE_LOCKS: Dict[Tuple[str, Optional[int]], threading.Lock] = {}
_BF16_SUPPORT_CACHE: Dict[Tuple[str, Optional[int]], bool] = {}


def _device_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    dev = torch.device(device)
    return dev.type, dev.index


def _needs_workspace_resize(
    workspace: Optional[torch.Tensor],
    dtype: torch.dtype,
    required_rows: int,
    cols: int,
) -> bool:
    if workspace is None:
        return True
    if workspace.ndim != 2:
        return True
    if workspace.dtype != dtype:
        return True
    if workspace.shape[1] != cols:
        return True
    if workspace.shape[0] < required_rows:
        return True
    return False


@contextlib.contextmanager
def lease_workspace(
    device: torch.device,
    dtype: torch.dtype,
    cols: int,
    required_rows: int,
):
    key = _device_cache_key(device)
    lock = _WORKSPACE_LOCKS.setdefault(key, threading.Lock())
    with lock:
        workspace = _WORKSPACE_CACHE.pop(key, None)
        reused = workspace is not None and not _needs_workspace_resize(
            workspace,
            dtype,
            required_rows,
            cols,
        )
        if not reused:
            rows = max(required_rows, 1)
            workspace = torch.empty((rows, cols), dtype=dtype, device=device)
    try:
        yield workspace, reused
    finally:
        with lock:
            _WORKSPACE_CACHE[key] = workspace


def device_supports_bfloat16(device: torch.device) -> bool:
    cache_key = _device_cache_key(device)
    cached = _BF16_SUPPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    dev = torch.device(device)
    if dev.type == "meta":
        _BF16_SUPPORT_CACHE[cache_key] = False
        return False

    try:
        a = torch.zeros((1, 1), dtype=torch.bfloat16, device=dev)
        b = torch.zeros((1, 1), dtype=torch.bfloat16, device=dev)
        _ = torch.matmul(a, b)
        support = True
    except Exception:
        support = False

    _BF16_SUPPORT_CACHE[cache_key] = support
    return support


def preferred_staging_dtype(
    hessian_cfg,
    input_dtype: torch.dtype,
    device: torch.device,
) -> torch.dtype:
    staging_dtype = hessian_cfg.staging_dtype
    if staging_dtype == torch.float32:
        return torch.float32
    if input_dtype not in (torch.float16, torch.bfloat16):
        return torch.float32
    if staging_dtype == torch.bfloat16:
        if not device_supports_bfloat16(device):
            return torch.float32
        return torch.bfloat16
    if staging_dtype == torch.float16:
        return torch.float16
    return torch.float32


def resolve_hessian_chunk_size(
    hessian_cfg,
    columns: int,
    rows: int,
    stage_dtype: torch.dtype,
) -> Optional[int]:
    if rows == 0:
        return None

    cfg_chunk = hessian_cfg.chunk_size
    if cfg_chunk is not None:
        return max(1, min(cfg_chunk, rows))

    bytes_budget = hessian_cfg.chunk_bytes
    if bytes_budget is not None:
        bytes_per_row = columns * torch.tensor([], dtype=stage_dtype).element_size()
        if bytes_per_row > 0:
            chunk_rows = bytes_budget // bytes_per_row
            if chunk_rows > 0:
                return max(1, min(int(chunk_rows), rows))
        return 1

    return None


@contextlib.contextmanager
def borrow_materialized_chunk_fp32(
    hessian_cfg,
    columns: int,
    chunk: torch.Tensor,
    rows: int,
):
    if rows == 0:
        yield chunk.new_zeros((0, columns), dtype=torch.float32)
        return

    device = chunk.device
    stage_dtype = preferred_staging_dtype(hessian_cfg, chunk.dtype, device)
    with lease_workspace(device, stage_dtype, columns, rows) as (staging_workspace, staging_reused):
        staging_view = staging_workspace[:rows, :]
        staging_view.copy_(chunk.to(dtype=stage_dtype))
        if stage_dtype == torch.float32:
            try:
                yield staging_view
            finally:
                if device.type == "cuda":
                    torch.cuda.current_stream(device).synchronize()
        else:
            with lease_workspace(device, torch.float32, columns, rows) as (
                fp32_workspace,
                _fp32_reused,
            ):
                fp32_view = fp32_workspace[:rows, :]
                fp32_view.copy_(staging_view.to(torch.float32))
                try:
                    yield fp32_view
                finally:
                    if device.type == "cuda":
                        torch.cuda.current_stream(device).synchronize()


def compute_hessian_xtx(
    hessian_cfg,
    columns: int,
    matrix: torch.Tensor,
) -> torch.Tensor:
    rows = matrix.shape[0]
    if rows == 0:
        return torch.zeros((columns, columns), dtype=torch.float32, device=matrix.device)

    stage_dtype = preferred_staging_dtype(hessian_cfg, matrix.dtype, matrix.device)
    chunk_size = resolve_hessian_chunk_size(hessian_cfg, columns, rows, stage_dtype)

    if chunk_size is None:
        mat32 = matrix.to(dtype=torch.float32)
        xtx = torch.matmul(mat32.T, mat32)
        del mat32
        torch_sync(device=xtx.device)
        return xtx

    xtx_accum = torch.zeros((columns, columns), dtype=torch.float32, device=matrix.device)
    for start in range(0, rows, chunk_size):
        rows_this = min(chunk_size, rows - start)
        source = matrix[start : start + rows_this]
        with borrow_materialized_chunk_fp32(hessian_cfg, columns, source, rows_this) as materialized:
            xtx_accum.add_(torch.matmul(materialized.T, materialized))
    torch_sync(device=xtx_accum.device)
    return xtx_accum
