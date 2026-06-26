# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Based on original gptq algorithm and code from https://github.com/IST-DASLab/gptq

import contextlib
import math
import os
import sys
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.nn.modules.conv import _ConvNd

from ..looper.named_module import NamedModule
from ..quantization import QuantizeConfig
from ..quantization.config import FallbackStrategy, SmoothMSE
from ..utils.device import get_device
from ..utils.logger import setup_logger
from ..utils.torch import torch_sync
from .fallback_smooth import mse_optimal_quant, smooth_block
from .gar import (
    compose_final_perm,
    compute_global_perm,
    compute_local_perms,
    extend_perm_with_tail,
    invert_perm,
)
from .npu_linalg import npu_inverse_cholesky_factor
from .qr_gptq_linalg import (
    apply_column_perm_to_qr,
    cholesky_hessian_inverse,
    merge_qr_factors,
    qr_hessian_inverse,
    update_qr_factor,
)
from ..ptq.hessian_core import device_supports_bfloat16, lease_workspace
from ..ptq.module_shape import get_number_of_rows_and_cols
from ..ptq.solvers.hessian import compute_hessian_inverse
from ..ptq.stats import StatisticsCollector
from .quantizer import HF_OPTIMUM, Quantizer

# Backward-compatible aliases for modules that imported these from gptq.
_lease_workspace = lease_workspace
_device_supports_bfloat16 = device_supports_bfloat16


log = setup_logger()


class GPTQ:
    @staticmethod
    def resolve_module_source(module: nn.Module) -> nn.Module:
        """Resolve the dense module view GPTQ should quantize for one wrapper."""

        if isinstance(module, NamedModule):
            quant_source = module.state.get("quant_source_module")
            if isinstance(quant_source, nn.Module):
                return quant_source
            return module.module
        return module

    def __init__(self, module: nn.Module, qcfg: Optional[QuantizeConfig] = None):
        self.lock = threading.Lock()

        # self.num_tied_handles = 0
        # if qcfg.tied_gptq_handle is not None:
        #     qcfg.tied_gptq_handle.num_tied_handles += 1

        # Flags indicating issues
        # self.issue_zero_samples = False
        # self.issue_nan_hessian = False
        # self.issue_non_invertible = False

        # self.W = module.weight
        resolved_module = self.resolve_module_source(module)
        self.rows, self.columns = get_number_of_rows_and_cols(resolved_module)
        if isinstance(module, NamedModule):
            self.module = resolved_module
            self.name = module.name
            self._named_module = module
        else:
            self.name = HF_OPTIMUM
            self.module = resolved_module
            self._named_module = None

        self._original_rows = self.rows
        self._original_columns = self.columns
        if self._named_module is not None:
            pad_info = self._named_module.state.get("tp_pad_info")
        else:
            pad_info = getattr(self.module, "_tp_pad_info", None)
        if isinstance(pad_info, dict):
            pad_cols = int(pad_info.get("pad_cols", 0) or 0)
            pad_cols = max(pad_cols, 0)
        else:
            pad_info = None
            pad_cols = 0

        self._tp_pad_info = pad_info
        self._tp_pad_cols = pad_cols
        if self._tp_pad_cols:
            self.columns += self._tp_pad_cols

        module_device = get_device(self.module)
        setattr(self.module, "target_device", module_device)

        if module_device.type == "meta":
            self._final_hessian_device_hint = torch.device("cpu")
        else:
            self._final_hessian_device_hint = torch.device(module_device)

        self.validate_module(self.module)

        self.qcfg = qcfg if qcfg else QuantizeConfig()  # HF compat will not pass qcfg
        self._validate_act_group_aware_shape()

        self.module_copy = None

        self.H = None
        self.nsamples = 0

        self.quantizer = self.create_quantizer(name=self.name)

        # fwd counter
        self.fwd_counter = 0

        self.fallback = self.qcfg.fallback
        self.expected_nsamples: Optional[float] = None

        self.H: Optional[torch.Tensor] = None

        # Store per-device Hessian contributions so multi-GPU calibration can
        # keep local accumulators and merge only once when quantization begins.
        self._device_hessian_partials: Dict[torch.device, torch.Tensor] = {}
        self._device_sample_counts: Dict[torch.device, int] = {}
        self._device_qr_partials: Dict[torch.device, torch.Tensor] = {}
        self._qr_R: Optional[torch.Tensor] = None
        self._hessian_dirty: bool = False
        self._stats_collector = StatisticsCollector(
            columns=self.columns,
            hessian=self.qcfg.hessian,
            row_buffer_max_rows=0,
            stats_device=self._final_hessian_device_hint,
        )

        self._borrow_workspace_stats = {
            "requests": 0,
            "staging_requests": 0,
            "staging_hits": 0,
            "staging_misses": 0,
            "materialized_requests": 0,
            "materialized_hits": 0,
            "materialized_misses": 0,
        }
        self._borrow_workspace_totals = {
            "requests": 0,
            "materialized_hits": 0,
            "materialized_misses": 0,
            "staging_hits": 0,
            "staging_misses": 0,
        }
        self._borrow_workspace_last_summary: Optional[Dict[str, object]] = None
        self._borrow_workspace_stage_dtype: Optional[torch.dtype] = None
        self._borrow_workspace_last_chunk_rows: Optional[int] = None

    def _validate_act_group_aware_shape(self) -> None:
        if not getattr(self.qcfg, "act_group_aware", False):
            return

        group_size = int(getattr(self.qcfg, "group_size", -1) or -1)
        if group_size <= 0:
            raise ValueError(
                f"Quantization: Module `{self.name}` -> `act_group_aware=True` requires `group_size > 0`, "
                f"got `{group_size}`."
            )

    @staticmethod
    def validate_module(module):
        assert isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d,
                                   transformers.Conv1D)), f"We supports only linear and convolutional layers. actual = `{module}`"

    # def has_hessian_issues(self) -> bool:
    #     return any([self.issue_zero_samples, self.issue_nan_hessian, self.issue_non_invertible])

    def create_quantizer(self, name: str) -> Quantizer:
        return Quantizer(qcfg=self.qcfg, name=name)

    def shape(self):
        if hasattr(self, "module"):
            return self.module.weight.shape
        else:
            return (0, 0)

    def mock_hessian_inverse(self, H: torch.Tensor):
        """Mock hessian inverse for fast testing"""
        damp = self.qcfg.damp_percent
        # Return identity matrix instead of complex inversion
        identity = torch.eye(H.shape[0], dtype=torch.float32, device=H.device)
        return identity, damp

    def log_cpu_fallback(self, stage: str, source_device: torch.device) -> None:
        """Explain when a memory-heavy GPTQ step moves from CUDA to CPU."""

        log.warn(
            "Quantization: Module `%s` -> CUDA OOM during %s on %s; falling back to CPU. "
            "Due to this fallback, the calculation may take much longer than normal.",
            self.name,
            stage,
            source_device,
        )

    def clone_module(self, copy=True, device: torch.device = None):
        if not device:
            device = self.module.weight.data.device

        clone = self.module.weight.data.to(copy=copy, device=device)

        if isinstance(self.module, _ConvNd):
            clone = clone.flatten(1)

        if isinstance(self.module, transformers.pytorch_utils.Conv1D):
            clone = clone.t()

        if self._tp_pad_cols:
            pad = torch.zeros(
                (clone.shape[0], self._tp_pad_cols),
                dtype=clone.dtype,
                device=clone.device,
            )
            clone = torch.cat((clone, pad), dim=1)

        return clone.float()

    @staticmethod
    def truncate_last_dim(tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.dim() == 0:
            return tensor

        trim = min(length, tensor.shape[-1])
        if trim == tensor.shape[-1]:
            return tensor

        return tensor.narrow(tensor.dim() - 1, 0, trim).contiguous()

    def _uses_qr_factorization(self) -> bool:
        return getattr(self.qcfg.hessian, "factorization", "cholesky") == "qr"

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
                source = matrix[start:start + rows_this]
                with self.borrow_materialized_chunk_fp32(source, rows_this) as materialized:
                    R = update_qr_factor(R, materialized)

        self._device_qr_partials[device] = R

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor, batch_index: Optional[int] = None):
        retain_activations = self._uses_qr_factorization()
        batch_token_size, xtx, device, activation_matrix = self.process_batch(
            inp,
            retain_activations=retain_activations,
        )
        if batch_token_size == 0 or xtx is None:
            return

        dev = torch.device(device)

        with self.lock:
            self.fwd_counter += 1
            self._stats_collector.accumulate_partial(
                xtx=xtx,
                device=dev,
                batch_rows=batch_token_size,
                activation_matrix=activation_matrix if retain_activations else None,
            )
            self.nsamples = self._stats_collector.nsamples
            self._hessian_dirty = True
            if retain_activations and activation_matrix is not None:
                del activation_matrix

    def preferred_staging_dtype(self, input_dtype: torch.dtype, device: torch.device) -> torch.dtype:
        device = torch.device(device)

        staging_dtype = self.qcfg.hessian.staging_dtype
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

        cfg_chunk = self.qcfg.hessian.chunk_size
        if cfg_chunk is not None:
            return max(1, min(cfg_chunk, rows))

        bytes_budget = self.qcfg.hessian.chunk_bytes
        if bytes_budget is not None:
            bytes_per_row = self.columns * torch.tensor([], dtype=stage_dtype).element_size()
            if bytes_per_row > 0:
                chunk_rows = bytes_budget // bytes_per_row
                if chunk_rows > 0:
                    return max(1, min(int(chunk_rows), rows))
            return 1

        return None

    @contextlib.contextmanager
    def borrow_materialized_chunk_fp32(
        self,
        chunk: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        if rows == 0:
            yield chunk.new_zeros((0, self.columns), dtype=torch.float32)
            return

        device = chunk.device
        stage_dtype = self.preferred_staging_dtype(chunk.dtype, device)

        stats = self._borrow_workspace_stats
        stats["requests"] += 1

        with _lease_workspace(device, stage_dtype, self.columns, rows) as (
            staging_workspace,
            staging_reused,
        ):
            stats["staging_requests"] += 1
            if staging_reused:
                stats["staging_hits"] += 1
            else:
                stats["staging_misses"] += 1

            staging_view = staging_workspace[:rows, :]
            staging_view.copy_(chunk.to(dtype=stage_dtype))

            if stage_dtype == torch.float32:
                stats["materialized_requests"] += 1
                if staging_reused:
                    stats["materialized_hits"] += 1
                else:
                    stats["materialized_misses"] += 1

                try:
                    yield staging_view
                finally:
                    if device.type == "cuda":
                        torch.cuda.current_stream(device).synchronize()
            else:
                with _lease_workspace(
                    device,
                    torch.float32,
                    self.columns,
                    rows,
                ) as (
                    fp32_workspace,
                    fp32_reused,
                ):
                    stats["materialized_requests"] += 1
                    if fp32_reused:
                        stats["materialized_hits"] += 1
                    else:
                        stats["materialized_misses"] += 1

                    try:
                        fp32_view = fp32_workspace[:rows, :]
                        fp32_view.copy_(staging_view.to(torch.float32))
                        yield fp32_view
                    finally:
                        if device.type == "cuda":
                            torch.cuda.current_stream(device).synchronize()

    def compute_hessian_xtx(self, matrix: torch.Tensor) -> torch.Tensor:
        return self._stats_collector.compute_hessian_xtx(matrix)

    def process_batch(
        self,
        inp: torch.Tensor,
        retain_activations: bool = False,
    ) -> Tuple[int, Optional[torch.Tensor], torch.device, Optional[torch.Tensor]]:
        # print(f"inp = {inp}")
        # print(f"self.module = {self.module} device = {self.module.target_device}")
        inp_device = get_device(inp)

        #inp = inp.to(device=self.module.target_device, dtype=torch.float32)

        # input reshaping
        if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
            reshaped_inp = inp.reshape(-1, inp.shape[-1])
        else:
            if isinstance(self.module, nn.Conv1d):
                reshaped_inp = inp.reshape(
                    inp.size(0) * self.module.groups,
                    inp.size(1) // self.module.groups,
                    inp.shape[2],
                    1,
                )
                unfold = nn.Unfold(
                    self.module.kernel_size + (1,),
                    dilation=self.module.dilation + (1,),
                    padding=self.module.padding + (0,),
                    stride=self.module.stride + (1,),
                )
                # output size (batch_size, channels * \prod kernel_size, num_patches)
                reshaped_inp = unfold(reshaped_inp)
            else:
                reshaped_inp = inp.reshape(
                    inp.size(0) * self.module.groups,
                    inp.size(1) // self.module.groups,
                    inp.shape[2],
                    inp.shape[3],
                )
                unfold = nn.Unfold(
                    self.module.kernel_size,
                    dilation=self.module.dilation,
                    padding=self.module.padding,
                    stride=self.module.stride,
                )
                # output size (batch_size, channels * \prod kernel_size, num_patches)
                reshaped_inp = unfold(reshaped_inp)
            reshaped_inp = reshaped_inp.transpose(1, 2).flatten(0, 1)

        # Delay dtype conversion until we materialize Hessian chunks to avoid unnecessary temporaries
        reshaped_inp = reshaped_inp.contiguous()
        if self._tp_pad_cols:
            pad = reshaped_inp.new_zeros((reshaped_inp.shape[0], self._tp_pad_cols))
            reshaped_inp = torch.cat((reshaped_inp, pad), dim=1)
            del pad
        canonical_device = torch.device(inp_device)

        batch_token_size = reshaped_inp.shape[0]

        if batch_token_size == 0:
            del reshaped_inp
            return 0, None, canonical_device, None

        activation_matrix = None
        try:
            xtx = self.compute_hessian_xtx(reshaped_inp).to(dtype=torch.float32)
            if retain_activations:
                activation_matrix = reshaped_inp.to(dtype=torch.float32).detach()
        except RuntimeError as exc:
            if (
                torch.device(inp_device).type == "cuda"
                and "out of memory" in str(exc).lower()
            ):
                log.warn(
                    "GPTQ module '%s' fell back to CPU Hessian accumulation due to GPU OOM during batch processing.",
                    getattr(self, "name", "<unknown>"),
                )
                reshaped_inp_cpu = reshaped_inp.to(device=torch.device("cpu"))
                del reshaped_inp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                canonical_device = torch.device("cpu")
                xtx = self.compute_hessian_xtx(reshaped_inp_cpu).to(dtype=torch.float32)
                xtx = xtx.detach()
                if retain_activations:
                    activation_matrix = reshaped_inp_cpu.to(dtype=torch.float32).detach()
                del reshaped_inp_cpu
            else:
                del reshaped_inp
                raise
        else:
            xtx = xtx.detach()
            del reshaped_inp

        self._snapshot_borrow_workspace_stats(context="process_batch")
        return batch_token_size, xtx, canonical_device, activation_matrix

    def _select_hessian_target_device(self, requested: Optional[torch.device]) -> torch.device:
        if requested is not None:
            return torch.device(requested)

        hint = getattr(self, "_final_hessian_device_hint", None)
        if hint is not None:
            return torch.device(hint)

        if self._stats_collector._device_hessian_partials:
            partial_device = next(iter(self._stats_collector._device_hessian_partials.keys()))
            return torch.device(partial_device)

        return torch.device("cpu")

    def materialize_global_hessian(self, target_device: Optional[torch.device] = None) -> None:
        device = self._select_hessian_target_device(target_device)

        with self.lock:
            if not self._hessian_dirty and self.H is not None:
                if self.H.device != device:
                    self.H = self.H.to(device=device)
                return

            self._stats_collector.finalize(target_device=device)
            self.H = self._stats_collector.H
            self._qr_R = self._stats_collector.qr_R
            self.nsamples = self._stats_collector.nsamples
            self._hessian_dirty = False
            if self.H is not None:
                self._final_hessian_device_hint = self.H.device
            else:
                self._final_hessian_device_hint = device

    def finalize_hessian(self, target_device: Optional[torch.device] = None) -> torch.Tensor:
        self.materialize_global_hessian(target_device=target_device)
        if self.H is None:
            self.H = self.create_H(target_device)
        return self.H

    def create_H(self, target_device):
        return torch.zeros((self.columns, self.columns), dtype=torch.float32,
                           device=self._select_hessian_target_device(target_device))

    def _fallback_quantize(self, strategy: FallbackStrategy, blocksize: int):
        """Apply a lightweight quantization fallback using the requested strategy."""
        maxq = 2 ** self.qcfg.bits - 1
        sigma = 3.0
        effective_group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns
        start_time = time.time()
        smooth_method = getattr(self.fallback, "smooth", None)
        mse_steps = 32
        mse_maxshrink = 0.8
        if isinstance(smooth_method, SmoothMSE):
            mse_steps = smooth_method.steps
            mse_maxshrink = smooth_method.maxshrink

        target_device = self.H.device if self.H is not None else self.module.weight.device
        W = self.clone_module(device=target_device)
        Q = torch.empty_like(W)
        scale_chunks = []
        zero_chunks = []

        for start in range(0, self.columns, effective_group_size):
            end = min(start + effective_group_size, self.columns)
            block = W[:, start:end]

            if isinstance(smooth_method, SmoothMSE):
                dequant, scale, zero = mse_optimal_quant(
                    block,
                    self.qcfg,
                    maxq,
                    steps=mse_steps,
                    maxshrink=mse_maxshrink,
                )
            else:
                block_mod, scale_factor = smooth_block(
                    block,
                    self.fallback,
                    group_size=effective_group_size,
                )
                if strategy == FallbackStrategy.MIDPOINT:
                    w_min = block_mod.min(dim=1, keepdim=True).values
                    w_max = block_mod.max(dim=1, keepdim=True).values
                    mid = (w_max + w_min) / 2.0
                    scale = torch.clamp((w_max - w_min) / maxq, min=1e-8)
                    zero_mid = torch.full_like(scale, maxq / 2.0)
                    q = torch.round((block_mod - mid) / scale + zero_mid)
                    q = torch.clamp(q, 0, maxq)
                    zero = torch.round(zero_mid - (mid / scale))
                    zero = torch.clamp(zero, 0, maxq)
                    dequant = (q - zero) * scale
                elif strategy == FallbackStrategy.MEAN:
                    mean = block_mod.mean(dim=1, keepdim=True)
                    max_dev = torch.max((block_mod - mean).abs(), dim=1, keepdim=True).values
                    max_dev = torch.clamp(max_dev, min=1e-8)
                    scale = (2 * max_dev) / maxq
                    zero_mid = torch.full_like(scale, maxq / 2.0)
                    q = torch.round((block_mod - mean) / scale + zero_mid)
                    q = torch.clamp(q, 0, maxq)
                    zero = torch.round(zero_mid - (mean / scale))
                    zero = torch.clamp(zero, 0, maxq)
                    dequant = (q - zero) * scale
                elif strategy == FallbackStrategy.MEDIAN:
                    median = block_mod.median(dim=1, keepdim=True).values
                    max_dev = torch.max((block_mod - median).abs(), dim=1, keepdim=True).values
                    max_dev = torch.clamp(max_dev, min=1e-8)
                    scale = (2 * max_dev) / maxq
                    zero_mid = torch.full_like(scale, maxq / 2.0)
                    q = torch.round((block_mod - median) / scale + zero_mid)
                    q = torch.clamp(q, 0, maxq)
                    zero = torch.round(zero_mid - (median / scale))
                    zero = torch.clamp(zero, 0, maxq)
                    dequant = (q - zero) * scale
                elif strategy == FallbackStrategy.STDCLIP:
                    mean = block_mod.mean(dim=1, keepdim=True)
                    std = block_mod.std(dim=1, keepdim=True, unbiased=False)
                    std = torch.clamp(std, min=1e-8)
                    lo = mean - sigma * std
                    hi = mean + sigma * std
                    scale = torch.clamp((hi - lo) / maxq, min=1e-8)
                    zero = torch.round(-lo / scale)
                    zero = torch.clamp(zero, 0, maxq)
                    q = torch.round(block_mod / scale + zero)
                    q = torch.clamp(q, 0, maxq)
                    dequant = (q - zero) * scale
                elif strategy == FallbackStrategy.RTN:
                    self.quantizer.find_params(block_mod, weight=True)
                    dequant = self.quantizer.quantize(block_mod)
                    scale = self.quantizer.scale
                    zero = self.quantizer.zero
                else:
                    raise ValueError(f"Unsupported fallback strategy: {strategy}")

                if scale_factor is not None:
                    scale = scale * scale_factor
                    dequant = dequant * scale_factor

            Q[:, start:end] = dequant

            scale_block = scale if scale.dim() > 1 else scale.unsqueeze(1)
            zero_block = zero if zero.dim() > 1 else zero.unsqueeze(1)
            if scale_block.shape[1] > 1:
                scale_block = scale_block.mean(dim=1, keepdim=True)
            if zero_block.shape[1] > 1:
                zero_block = zero_block.mean(dim=1, keepdim=True)
            scale_chunks.append(scale_block)
            zero_chunks.append(zero_block)

        scale = torch.cat(scale_chunks, dim=1)
        zero = torch.cat(zero_chunks, dim=1)

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            Q = Q[:, :valid_cols]
            scale = self.truncate_last_dim(scale, valid_cols)
            zero = self.truncate_last_dim(zero, valid_cols)
        else:
            valid_cols = self.columns

        group_size = effective_group_size if effective_group_size != -1 else self.columns
        g_idx = torch.arange(valid_cols, device=Q.device, dtype=torch.int32) // group_size

        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()

        if Q.shape != self.module.weight.shape:
            Q = Q.reshape(self.module.weight.shape).to(self.module.weight.dtype)
        else:
            Q = Q.to(self.module.weight.dtype)

        Q = Q.to(device=self.module.weight.data.device, non_blocking=False)
        mean_abs_err = (Q - self.module.weight.data).abs().mean().item()
        duration = time.time() - start_time
        avg_loss = f"fallback({strategy.value}): {mean_abs_err:.7f}"
        damp = 0.0

        self.H = None
        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    # FIXME, optimum needs fasterquant, we need to remove it
    def fasterquant(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
    ):
        return self.hf_quantize(blocksize, percdamp, damp_auto_increment, group_size, actorder, static_groups)

    # public api exposed to hf
    def hf_quantize(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
            act_group_aware: Optional[bool] = None,
    ):
        self.qcfg.group_size = group_size
        self.qcfg.damp_percent = percdamp
        self.qcfg.damp_auto_increment = damp_auto_increment
        self.qcfg.desc_act = actorder
        if act_group_aware is not None:
            self.qcfg.act_group_aware = act_group_aware
        self.qcfg._resolve_activation_ordering(actorder, act_group_aware)
        self.qcfg.static_groups = static_groups
        (Q, scale, zero, g_idx, duration, avg_loss, damp_percent, nsamples) = self.quantize(blocksize=blocksize)
        self.module.weight.data = Q
        return scale, zero, g_idx, duration, avg_loss, damp_percent

    @torch.inference_mode()
    def hessian_inverse(self, H: torch.Tensor, qr_R: Optional[torch.Tensor] = None):
        return compute_hessian_inverse(
            H,
            qcfg=self.qcfg,
            nsamples=self.nsamples,
            qr_R=qr_R,
            module_name=self.name,
            uses_qr_factorization=self._uses_qr_factorization(),
        )

    @torch.inference_mode()
    def quantize(
            self,
            blocksize=128,
    ):
        from ..ptq.solvers.gptq_quantizer import GptqQuantizer
        from ..utils.fallback import should_use_fallback

        fallback_requested = should_use_fallback(
            self.fallback,
            float(self.nsamples),
            self.expected_nsamples,
        )
        if not fallback_requested:
            target_device = getattr(self.module, "target_device", None)
            self.finalize_hessian(target_device=target_device)

        wrapper = self._named_module if self._named_module is not None else self.module
        quantizer = GptqQuantizer(
            wrapper,
            qcfg=self.qcfg,
            H=self.H if self.H is not None else self.create_H(getattr(self.module, "target_device", None)),
            nsamples=self.nsamples,
            qr_R=self._qr_R,
            expected_nsamples=self.expected_nsamples,
            fallback=self.fallback,
            module_copy=self.module_copy,
            name=self.name,
        )
        result = quantizer.quantize(blocksize=blocksize)
        self.module_copy = None
        self.H = None
        return (
            result.pack_weight,
            result.q_scales,
            result.q_zeros,
            result.q_g_idx,
            result.duration,
            result.avg_loss,
            result.damp,
            result.nsamples,
        )

    def borrow_materialized_chunk_stats(self, reset: bool = False) -> Dict[str, int]:
        stats = dict(self._borrow_workspace_stats)
        if reset:
            for key in self._borrow_workspace_stats:
                self._borrow_workspace_stats[key] = 0
        return stats

    def _snapshot_borrow_workspace_stats(self, *, context: str) -> None:
        stats = self.borrow_materialized_chunk_stats(reset=True)
        total_requests = int(stats.get("requests", 0) or 0)
        if total_requests == 0:
            return

        materialized_hits = int(stats.get("materialized_hits", 0) or 0)
        materialized_misses = int(stats.get("materialized_misses", 0) or 0)
        staging_hits = int(stats.get("staging_hits", 0) or 0)
        staging_misses = int(stats.get("staging_misses", 0) or 0)
        chunk_rows = self._borrow_workspace_last_chunk_rows
        stage_dtype = self._borrow_workspace_stage_dtype
        stage_dtype_str = str(stage_dtype) if stage_dtype is not None else "n/a"
        hit_rate = materialized_hits / total_requests if total_requests else 0.0

        summary = {
            "context": context,
            "requests": total_requests,
            "materialized_hits": materialized_hits,
            "materialized_misses": materialized_misses,
            "staging_hits": staging_hits,
            "staging_misses": staging_misses,
            "chunk_rows": chunk_rows,
            "staging_dtype": stage_dtype_str,
            "hit_rate": hit_rate,
        }
        self._borrow_workspace_last_summary = summary

        totals = self._borrow_workspace_totals
        totals["requests"] += total_requests
        totals["materialized_hits"] += materialized_hits
        totals["materialized_misses"] += materialized_misses
        totals["staging_hits"] += staging_hits
        totals["staging_misses"] += staging_misses

    def log_workspace_stats(self, *, context: str, reset: bool = True) -> None:
        totals = self._borrow_workspace_totals
        total_requests = int(totals.get("requests", 0) or 0)
        if total_requests == 0:
            if reset:
                self.reset_workspace_stats()
            return

        total_hits = int(totals.get("materialized_hits", 0) or 0)
        total_misses = int(totals.get("materialized_misses", 0) or 0)
        total_hit_rate = total_hits / total_requests if total_requests else 0.0

        last = self._borrow_workspace_last_summary or {}
        last_requests = int(last.get("requests", 0) or 0)
        last_hits = int(last.get("materialized_hits", 0) or 0)
        last_misses = int(last.get("materialized_misses", 0) or 0)
        last_hit_rate = float(last.get("hit_rate", 0.0) or 0.0)
        rows_label = last.get("chunk_rows", "n/a")
        stage_dtype = last.get("staging_dtype", "n/a")

        log.info(
            "GPTQ workspace cache [%s]: module=%s rows=%s staging_dtype=%s "
            "requests=%d hits=%d misses=%d hit_rate=%.2f total_requests=%d "
            "total_hits=%d total_misses=%d total_hit_rate=%.2f",
            context,
            getattr(self, "name", "<unknown>"),
            rows_label,
            stage_dtype,
            last_requests,
            last_hits,
            last_misses,
            last_hit_rate,
            total_requests,
            total_hits,
            total_misses,
            total_hit_rate,
        )

        if reset:
            self.reset_workspace_stats()

    def reset_workspace_stats(self) -> None:
        for key in self._borrow_workspace_stats:
            self._borrow_workspace_stats[key] = 0
        for key in self._borrow_workspace_totals:
            self._borrow_workspace_totals[key] = 0
        self._borrow_workspace_last_summary = None
        self._borrow_workspace_stage_dtype = None
        self._borrow_workspace_last_chunk_rows = None

    def free(self):
        if hasattr(self, "H"):
            del self.H
        del self.quantizer
        if hasattr(self, "module_copy"):
            del self.module_copy

        if self._named_module is not None:
            self._named_module.state.pop("tp_pad_info", None)

        target = getattr(self, "module", None)
        if target is not None:
            del self.module

        # torch_empty_cache(self.device)


__all__ = ["GPTQ"]
