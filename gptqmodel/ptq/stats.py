# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Streaming activation statistics for PTQ (H, QR factor, bounded row buffer)."""

from __future__ import annotations

import contextlib
import threading
from typing import Dict, Optional, Tuple

import torch

from ..quantization.config import HessianConfig
from ..quantization.qr_gptq_linalg import merge_qr_factors, update_qr_factor
from ..utils.torch import torch_sync

# Reuse GPTQ workspace leasing for chunked materialization.
from ..quantization.gptq import (  # noqa: E402
    _device_supports_bfloat16,
    _lease_workspace,
)


class StatisticsCollector:
    """
    Accumulate sufficient statistics without storing the full activation matrix.

    Also exposed as ``HessianAccumulator`` for the PTQ protocol naming.

    Tiered memory policy:
    - GPU: one microbatch / chunk during forward hook
    - CPU: merged H or QR factor (O(c^2))
    - Bounded row buffer (O(k*c)) for transform optimization
    """

    DEFAULT_ROW_BUFFER_MAX = 2048

    def __init__(
        self,
        *,
        columns: int,
        hessian: HessianConfig,
        row_buffer_max_rows: Optional[int] = None,
        stats_device: torch.device | None = None,
    ) -> None:
        self.columns = int(columns)
        self.hessian = hessian
        self.row_buffer_max_rows = (
            int(row_buffer_max_rows)
            if row_buffer_max_rows is not None
            else self.DEFAULT_ROW_BUFFER_MAX
        )
        self.stats_device = torch.device(stats_device or "cpu")

        self.nsamples = 0
        self._lock = threading.Lock()
        self._hessian_dirty = False

        self._device_hessian_partials: Dict[torch.device, torch.Tensor] = {}
        self._device_sample_counts: Dict[torch.device, int] = {}
        self._device_qr_partials: Dict[torch.device, torch.Tensor] = {}

        self.H: Optional[torch.Tensor] = None
        self.qr_R: Optional[torch.Tensor] = None
        self.row_buffer: Optional[torch.Tensor] = None
        self._row_buffer_rows = 0

    def _uses_qr_factorization(self) -> bool:
        return getattr(self.hessian, "factorization", "cholesky") == "qr"

    def preferred_staging_dtype(self, input_dtype: torch.dtype, device: torch.device) -> torch.dtype:
        staging_dtype = self.hessian.staging_dtype
        if staging_dtype == torch.float32:
            return torch.float32
        if input_dtype not in (torch.float16, torch.bfloat16):
            return torch.float32
        if staging_dtype == torch.bfloat16:
            if not _device_supports_bfloat16(device):
                return torch.float32
            return torch.bfloat16
        if staging_dtype == torch.float16:
            return torch.float16
        return torch.float32

    def resolve_hessian_chunk_size(self, rows: int, stage_dtype: torch.dtype) -> Optional[int]:
        if rows == 0:
            return None
        cfg_chunk = self.hessian.chunk_size
        if cfg_chunk is not None:
            return max(1, min(cfg_chunk, rows))
        bytes_budget = self.hessian.chunk_bytes
        if bytes_budget is not None:
            bytes_per_row = self.columns * torch.tensor([], dtype=stage_dtype).element_size()
            if bytes_per_row > 0:
                chunk_rows = bytes_budget // bytes_per_row
                if chunk_rows > 0:
                    return max(1, min(int(chunk_rows), rows))
            return 1
        return None

    @contextlib.contextmanager
    def _borrow_materialized_chunk_fp32(self, chunk: torch.Tensor, rows: int):
        if rows == 0:
            yield chunk.new_zeros((0, self.columns), dtype=torch.float32)
            return

        device = chunk.device
        stage_dtype = self.preferred_staging_dtype(chunk.dtype, device)
        with _lease_workspace(device, stage_dtype, self.columns, rows) as (
            staging_workspace,
            staging_reused,
        ):
            staging_view = staging_workspace[:rows, :]
            staging_view.copy_(chunk.to(dtype=stage_dtype))
            if stage_dtype == torch.float32:
                try:
                    yield staging_view
                finally:
                    if device.type == "cuda":
                        torch.cuda.current_stream(device).synchronize()
            else:
                with _lease_workspace(device, torch.float32, self.columns, rows) as (
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

    def compute_hessian_xtx(self, matrix: torch.Tensor) -> torch.Tensor:
        rows = matrix.shape[0]
        if rows == 0:
            return torch.zeros((self.columns, self.columns), dtype=torch.float32, device=matrix.device)

        stage_dtype = self.preferred_staging_dtype(matrix.dtype, matrix.device)
        chunk_size = self.resolve_hessian_chunk_size(rows, stage_dtype)

        if chunk_size is None:
            mat32 = matrix.to(dtype=torch.float32)
            xtx = torch.matmul(mat32.T, mat32)
            del mat32
            torch_sync(device=xtx.device)
            return xtx

        xtx_accum = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=matrix.device)
        for start in range(0, rows, chunk_size):
            rows_this = min(chunk_size, rows - start)
            source = matrix[start : start + rows_this]
            with self._borrow_materialized_chunk_fp32(source, rows_this) as materialized:
                xtx_accum.add_(torch.matmul(materialized.T, materialized))
        torch_sync(device=xtx_accum.device)
        return xtx_accum

    def _update_qr_from_matrix(self, matrix: torch.Tensor, device: torch.device) -> None:
        rows = matrix.shape[0]
        if rows == 0:
            return
        stage_dtype = self.preferred_staging_dtype(matrix.dtype, matrix.device)
        chunk_size = self.resolve_hessian_chunk_size(rows, stage_dtype)
        R = self._device_qr_partials.get(device)
        if chunk_size is None:
            R = update_qr_factor(R, matrix.to(dtype=torch.float32))
        else:
            for start in range(0, rows, chunk_size):
                rows_this = min(chunk_size, rows - start)
                source = matrix[start : start + rows_this]
                with self._borrow_materialized_chunk_fp32(source, rows_this) as materialized:
                    R = update_qr_factor(R, materialized)
        self._device_qr_partials[device] = R

    def _append_row_buffer(self, rows: torch.Tensor) -> None:
        if self.row_buffer_max_rows <= 0 or rows.numel() == 0:
            return
        rows_cpu = rows.detach().to(device=self.stats_device, dtype=torch.float16)
        if self._row_buffer_rows >= self.row_buffer_max_rows:
            return
        need = self.row_buffer_max_rows - self._row_buffer_rows
        take = rows_cpu[:need]
        if self.row_buffer is None:
            self.row_buffer = take.clone()
        else:
            self.row_buffer = torch.cat([self.row_buffer, take], dim=0)
        self._row_buffer_rows = self.row_buffer.shape[0]

    def add_batch(self, inp: torch.Tensor) -> None:
        """Ingest one activation batch shaped [*, in_features]."""
        if inp.dim() < 2:
            raise ValueError(f"StatisticsCollector expects rank-2+ input, got {tuple(inp.shape)}")
        reshaped = inp.reshape(-1, inp.shape[-1]).contiguous()
        if reshaped.shape[-1] != self.columns:
            if reshaped.shape[-1] < self.columns:
                pad = reshaped.new_zeros((reshaped.shape[0], self.columns - reshaped.shape[-1]))
                reshaped = torch.cat([reshaped, pad], dim=1)
            else:
                reshaped = reshaped[:, : self.columns]

        batch_rows = reshaped.shape[0]
        if batch_rows == 0:
            return

        retain_qr = self._uses_qr_factorization()
        try:
            xtx = self.compute_hessian_xtx(reshaped).to(dtype=torch.float32)
            activation_matrix = reshaped.to(dtype=torch.float32).detach() if retain_qr else None
        except RuntimeError as exc:
            if torch.device(reshaped.device).type == "cuda" and "out of memory" in str(exc).lower():
                reshaped_cpu = reshaped.to(device=torch.device("cpu"))
                del reshaped
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                xtx = self.compute_hessian_xtx(reshaped_cpu).to(dtype=torch.float32).detach()
                activation_matrix = (
                    reshaped_cpu.to(dtype=torch.float32).detach() if retain_qr else None
                )
                dev = torch.device("cpu")
                del reshaped_cpu
            else:
                raise
        else:
            xtx = xtx.detach()
            dev = torch.device(reshaped.device)
            del reshaped

        with self._lock:
            existing = self._device_hessian_partials.get(dev)
            if existing is None:
                self._device_hessian_partials[dev] = xtx
            else:
                existing.add_(xtx)
            if retain_qr and activation_matrix is not None:
                self._update_qr_from_matrix(activation_matrix, dev)
                del activation_matrix
            self._device_sample_counts[dev] = self._device_sample_counts.get(dev, 0) + batch_rows
            self.nsamples += batch_rows
            self._hessian_dirty = True

        rows_for_buffer = inp.reshape(-1, inp.shape[-1]).contiguous()
        if rows_for_buffer.shape[-1] != self.columns:
            if rows_for_buffer.shape[-1] < self.columns:
                pad = rows_for_buffer.new_zeros(
                    (rows_for_buffer.shape[0], self.columns - rows_for_buffer.shape[-1])
                )
                rows_for_buffer = torch.cat([rows_for_buffer, pad], dim=1)
            else:
                rows_for_buffer = rows_for_buffer[:, : self.columns]
        self._append_row_buffer(rows_for_buffer)

    def finalize(self, target_device: Optional[torch.device] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Merge partial statistics onto ``stats_device`` and return ``(H, qr_R)``."""
        device = torch.device(target_device or self.stats_device)
        with self._lock:
            if not self._hessian_dirty and self.H is not None:
                return self.H, self.qr_R

            total_samples = sum(self._device_sample_counts.values())
            result_accum = torch.zeros(
                (self.columns, self.columns),
                dtype=torch.float32,
                device=device,
            )

            if total_samples > 0:
                for partial in self._device_hessian_partials.values():
                    result_accum.add_(partial.to(device=device, dtype=torch.float32))
                result_accum.mul_(2.0 / float(total_samples))

            self.H = result_accum
            self.nsamples = total_samples
            self._hessian_dirty = False
            self._device_hessian_partials.clear()
            self._device_sample_counts.clear()

            if self._uses_qr_factorization() and self._device_qr_partials:
                merged_R: Optional[torch.Tensor] = None
                for partial_R in self._device_qr_partials.values():
                    partial_R = partial_R.to(device=device, dtype=torch.float32)
                    merged_R = partial_R if merged_R is None else merge_qr_factors(merged_R, partial_R)
                self.qr_R = merged_R
            else:
                self.qr_R = None
            self._device_qr_partials.clear()

            return self.H, self.qr_R

    def to_context(
        self,
        *,
        module_name: str,
        rows: int,
        expected_calibration_tokens: Optional[int] = None,
    ) -> "ModuleCalibContext":
        from .context import ModuleCalibContext

        self.finalize()
        return ModuleCalibContext(
            module_name=module_name,
            columns=self.columns,
            rows=rows,
            nsamples=self.nsamples,
            expected_calibration_tokens=expected_calibration_tokens,
            H=self.H,
            qr_R=self.qr_R,
            row_buffer=self.row_buffer,
        )

    def free(self) -> None:
        """Release accumulated tensors."""
        self.H = None
        self.qr_R = None
        self.row_buffer = None
        self._row_buffer_rows = 0
        self.nsamples = 0
        self._device_hessian_partials.clear()
        self._device_sample_counts.clear()
        self._device_qr_partials.clear()
        self._hessian_dirty = False


# Protocol-facing alias for the Chen et al. / GPTQ Hessian accumulation stage.
HessianAccumulator = StatisticsCollector
