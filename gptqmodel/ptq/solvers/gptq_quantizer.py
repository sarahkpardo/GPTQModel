# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Column-wise GPTQ quantizer operating on precomputed Hessian statistics."""

from __future__ import annotations

import copy
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.nn.modules.conv import _ConvNd

from ...looper.named_module import NamedModule
from ...quantization import QuantizeConfig
from ...quantization.config import FallbackStrategy
from ...quantization.gar import (
    compose_final_perm,
    compute_global_perm,
    compute_local_perms,
    extend_perm_with_tail,
    invert_perm,
)
from ...quantization.qr_gptq_linalg import apply_column_perm_to_qr
from ...ptq.module_shape import get_number_of_rows_and_cols
from ...quantization.quantizer import HF_OPTIMUM, Quantizer
from ...utils.fallback import resolve_fallback_strategy, resolve_threshold, should_use_fallback
from ...utils.logger import setup_logger
from .hessian import compute_hessian_inverse

log = setup_logger()


@dataclass
class GptqQuantizeResult:
    pack_weight: torch.Tensor
    q_scales: torch.Tensor
    q_zeros: torch.Tensor
    q_g_idx: torch.Tensor
    duration: float
    avg_loss: float | str
    damp: float
    nsamples: int


class GptqQuantizer:
    """GPTQ Babai loop from ``(W, H, H_inv)`` without activation capture."""

    def __init__(
        self,
        module: nn.Module,
        *,
        qcfg: QuantizeConfig,
        H: torch.Tensor,
        nsamples: int,
        qr_R: Optional[torch.Tensor] = None,
        expected_nsamples: Optional[float] = None,
        fallback: Any = None,
        module_copy: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
    ) -> None:
        resolved_module = self._resolve_module_source(module)
        self.rows, self.columns = get_number_of_rows_and_cols(resolved_module)
        if isinstance(module, NamedModule):
            self.module = resolved_module
            self.name = name or module.name
            self._named_module = module
        else:
            self.module = resolved_module
            self.name = name or HF_OPTIMUM
            self._named_module = None

        self._original_rows = self.rows
        self._original_columns = self.columns
        pad_info = None
        pad_cols = 0
        if self._named_module is not None:
            pad_info = self._named_module.state.get("tp_pad_info")
        else:
            pad_info = getattr(self.module, "_tp_pad_info", None)
        if isinstance(pad_info, dict):
            pad_cols = max(int(pad_info.get("pad_cols", 0) or 0), 0)
        else:
            pad_info = None
            pad_cols = 0

        self._tp_pad_info = pad_info
        self._tp_pad_cols = pad_cols
        if self._tp_pad_cols:
            self.columns += self._tp_pad_cols

        self.qcfg = qcfg
        self.fallback = fallback
        self.expected_nsamples = expected_nsamples
        self.quantizer = Quantizer(qcfg=self.qcfg, name=self.name)
        self.quantizer.configure(perchannel=True)

        self.H = H.to(device=H.device, dtype=torch.float32)
        self.nsamples = int(nsamples)
        self._qr_R = qr_R.to(device=H.device, dtype=torch.float32) if qr_R is not None else None
        self.module_copy = module_copy

    @staticmethod
    def _resolve_module_source(module: nn.Module) -> nn.Module:
        if isinstance(module, NamedModule):
            quant_source = module.state.get("quant_source_module")
            if isinstance(quant_source, nn.Module):
                return quant_source
            return module.module
        return module

    @staticmethod
    def truncate_last_dim(tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.dim() == 0:
            return tensor
        trim = min(length, tensor.shape[-1])
        if trim == tensor.shape[-1]:
            return tensor
        return tensor.narrow(tensor.dim() - 1, 0, trim).contiguous()

    def clone_module(self, copy: bool = True, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
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

    def log_cpu_fallback(self, stage: str, source_device: torch.device) -> None:
        log.warn(
            "Quantization: Module `%s` -> CUDA OOM during %s on %s; falling back to CPU. "
            "Due to this fallback, the calculation may take much longer than normal.",
            self.name,
            stage,
            source_device,
        )

    def _uses_qr_factorization(self) -> bool:
        return getattr(self.qcfg.hessian, "factorization", "cholesky") == "qr"

    def mock_hessian_inverse(self, H: torch.Tensor):
        damp = self.qcfg.damp_percent
        identity = torch.eye(H.shape[0], dtype=torch.float32, device=H.device)
        return identity, damp

    @torch.inference_mode()
    def quantize(self, blocksize: int = 128) -> GptqQuantizeResult:
        start = time.time()
        target_device = getattr(self.module, "target_device", None)
        result_device = torch.device(self.module.weight.data.device)
        cpu_fallback_used = False

        resolved_strategy = resolve_fallback_strategy(self.fallback)
        fallback_requested = should_use_fallback(
            self.fallback,
            float(self.nsamples),
            self.expected_nsamples,
        )
        threshold_raw, is_percent = resolve_threshold(self.fallback, self.expected_nsamples)
        fallback_configured = threshold_raw is not None

        if fallback_requested:
            from ...quantization.gptq import GPTQ

            legacy = GPTQ.__new__(GPTQ)
            legacy.module = self.module
            legacy.name = self.name
            legacy.qcfg = self.qcfg
            legacy.columns = self.columns
            legacy._original_columns = self._original_columns
            legacy._tp_pad_cols = self._tp_pad_cols
            legacy.quantizer = self.quantizer
            legacy.fallback = self.fallback
            legacy.H = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=result_device)
            q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = legacy._fallback_quantize(
                resolved_strategy,
                blocksize,
            )
            return GptqQuantizeResult(
                pack_weight=q,
                q_scales=scale,
                q_zeros=zero,
                q_g_idx=g_idx,
                duration=duration,
                avg_loss=avg_loss,
                damp=damp,
                nsamples=nsamples,
            )

        use_hessian = True

        if sys.platform == "darwin" and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            raise RuntimeError(
                "For MacOS you must set env `PYTORCH_ENABLE_MPS_FALLBACK=1` before running quantization."
            )

        if self.module_copy is None:
            W = self.clone_module(device=self.H.device)
        else:
            W = self.module_copy.to(device=self.H.device)
            del self.module_copy
            self.module_copy = None

        self.quantizer.find_params(W, weight=True)

        if use_hessian:
            dead = torch.diag(self.H) == 0
            self.H[dead, dead] = 1
            W[:, dead] = 0

        scale: list[torch.Tensor] = []
        zero: list[torch.Tensor] = []
        now_idx = 1
        perm = None
        invperm = None
        final_perm = None
        global_perm = None
        local_perms = None

        if self.qcfg.static_groups:
            groups = []
            for i in range(0, self.columns, self.qcfg.group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + self.qcfg.group_size)], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)
        else:
            groups = []

        if self.qcfg.desc_act and use_hessian:
            perm = torch.argsort(torch.diag(self.H), descending=True)
            try:
                W = W[:, perm]
                self.H = self.H[perm][:, perm]
            except RuntimeError as exc:
                if self.H.device.type != "cuda" or "out of memory" not in str(exc).lower():
                    raise
                self.log_cpu_fallback("Hessian permutation", self.H.device)
                cpu_fallback_used = True
                cpu_device = torch.device("cpu")
                perm = perm.to(device=cpu_device)
                W = W.to(device=cpu_device)[:, perm]
                self.H = self.H.to(device=cpu_device)[perm][:, perm]
                self.quantizer.find_params(W, weight=True)
            invperm = torch.argsort(perm)

        elif self.qcfg.act_group_aware and use_hessian:
            diag_h = torch.diag(self.H)
            local_perms, local_values = compute_local_perms(
                diag_h, self.qcfg.group_size, return_values=True
            )
            global_perm = compute_global_perm(
                diag_h,
                self.qcfg.group_size,
                precomputed_values=local_values,
            )
            del local_values
            final_perm = compose_final_perm(local_perms, global_perm, self.qcfg.group_size)
            final_perm = extend_perm_with_tail(final_perm, self.columns)
            try:
                W = W[:, final_perm]
                self.H = self.H[final_perm][:, final_perm]
            except RuntimeError as exc:
                if self.H.device.type != "cuda" or "out of memory" not in str(exc).lower():
                    raise
                self.log_cpu_fallback("act-group Hessian permutation", self.H.device)
                cpu_fallback_used = True
                cpu_device = torch.device("cpu")
                final_perm = final_perm.to(device=cpu_device)
                W = W.to(device=cpu_device)[:, final_perm]
                self.H = self.H.to(device=cpu_device)[final_perm][:, final_perm]
                self.quantizer.find_params(W, weight=True)

        Hinv = None
        damp = 0.0
        if use_hessian:
            qr_R = self._qr_R
            column_perm = None
            if self.qcfg.desc_act:
                column_perm = perm
            elif self.qcfg.act_group_aware:
                column_perm = final_perm

            if qr_R is not None and column_perm is not None:
                qr_R = apply_column_perm_to_qr(qr_R, column_perm)
            if qr_R is not None:
                qr_R = qr_R.to(device=self.H.device)

            hessian_inverse_fn: Callable = compute_hessian_inverse
            if self.qcfg.mock_quantization:
                hessian_inverse_fn = self.mock_hessian_inverse

            try:
                Hinv, damp = hessian_inverse_fn(
                    self.H,
                    qcfg=self.qcfg,
                    nsamples=self.nsamples,
                    qr_R=qr_R,
                    module_name=self.name,
                    uses_qr_factorization=self._uses_qr_factorization(),
                )
            except RuntimeError as exc:
                if self.H.device.type != "cuda" or "out of memory" not in str(exc).lower():
                    raise
                self.log_cpu_fallback("Hessian inverse", self.H.device)
                cpu_fallback_used = True
                cpu_device = torch.device("cpu")
                self.H = self.H.to(device=cpu_device)
                W = W.to(device=cpu_device)
                self.quantizer.find_params(W, weight=True)
                if qr_R is not None:
                    qr_R = qr_R.to(device=cpu_device)
                Hinv, damp = hessian_inverse_fn(
                    self.H,
                    qcfg=self.qcfg,
                    nsamples=self.nsamples,
                    qr_R=qr_R,
                    module_name=self.name,
                    uses_qr_factorization=self._uses_qr_factorization(),
                )

            if Hinv is None and not fallback_configured:
                raise RuntimeError(
                    f"Quantization: Module `{self.name}` -> Hessian inverse failed after damping "
                    f"recovery (nsamples={self.nsamples}). Increase calibration data or "
                    f"`damp_percent` (last tried={damp:.5f})."
                )

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        if self.qcfg.mock_quantization:
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1
                W1 = W[:, i1:i2]
                Q1 = torch.zeros_like(W1)
                if self.qcfg.group_size != -1:
                    if not self.qcfg.static_groups:
                        group_start_cols = list(range(i1, i2, self.qcfg.group_size))
                        for group_start in group_start_cols:
                            group_end = min(group_start + self.qcfg.group_size, self.columns)
                            if group_start < group_end:
                                self.quantizer.find_params(W[:, group_start:group_end], weight=True)
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                    else:
                        for i in range(count):
                            idx = i1 + i
                            if self.qcfg.desc_act:
                                idx = perm[idx]
                            self.quantizer = groups[idx // self.qcfg.group_size]
                    if scale and zero:
                        latest_scale = scale[-1]
                        latest_zero = zero[-1]
                        if latest_scale.dim() == 1:
                            latest_scale = latest_scale.view(-1, 1)
                        if latest_zero.dim() == 1:
                            latest_zero = latest_zero.view(-1, 1)
                        maxq_val = 2 ** self.qcfg.bits - 1
                        if self.qcfg.sym:
                            Q1 = latest_scale * torch.clamp(
                                torch.round(W1 / latest_scale),
                                -(maxq_val // 2),
                                maxq_val // 2,
                            )
                        else:
                            quantized = torch.clamp(
                                torch.round(W1 / latest_scale) + latest_zero,
                                0,
                                maxq_val,
                            )
                            Q1 = latest_scale * (quantized - latest_zero)
                    else:
                        for i in range(count):
                            w = W1[:, i]
                            q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                            Q1[:, i] = q
                else:
                    maxq_val = 2 ** self.qcfg.bits - 1
                    if hasattr(self.quantizer, "scale") and hasattr(self.quantizer, "zero"):
                        latest_scale = self.quantizer.scale
                        latest_zero = self.quantizer.zero
                        if latest_scale.dim() == 1:
                            latest_scale = latest_scale.view(-1, 1)
                        if latest_zero.dim() == 1:
                            latest_zero = latest_zero.view(-1, 1)
                        if self.qcfg.sym:
                            Q1 = latest_scale * torch.clamp(
                                torch.round(W1 / latest_scale),
                                -(maxq_val // 2),
                                maxq_val // 2,
                            )
                        else:
                            quantized = torch.clamp(
                                torch.round(W1 / latest_scale) + latest_zero,
                                0,
                                maxq_val,
                            )
                            Q1 = latest_scale * (quantized - latest_zero)
                    else:
                        for i in range(count):
                            w = W1[:, i]
                            q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                            Q1[:, i] = q
                Q[:, i1:i2] = Q1
        else:
            effective_block = blocksize
            if Hinv is None and self.qcfg.group_size and self.qcfg.group_size > 0:
                effective_block = self.qcfg.group_size

            for i1 in range(0, self.columns, effective_block):
                i2 = min(i1 + effective_block, self.columns)
                count = i2 - i1
                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1) if Hinv is not None else None
                Losses1 = torch.zeros_like(W1) if Hinv is not None else None
                if Hinv is not None:
                    Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    if Hinv is not None:
                        d = Hinv1[i, i]

                    if self.qcfg.group_size != -1:
                        if not self.qcfg.static_groups:
                            if (i1 + i) % self.qcfg.group_size == 0:
                                self.quantizer.find_params(
                                    W[:, (i1 + i) : (i1 + i + self.qcfg.group_size)],
                                    weight=True,
                                )
                            if ((i1 + i) // self.qcfg.group_size) - now_idx == -1:
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                        else:
                            idx = i1 + i
                            if self.qcfg.desc_act:
                                idx = perm[idx]
                            self.quantizer = groups[idx // self.qcfg.group_size]

                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                    Q1[:, i] = q
                    if Hinv is not None:
                        Losses1[:, i] = (w - q) ** 2 / d**2
                        err1 = (w - q) / d
                        W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                        Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                if Hinv is not None:
                    Losses[:, i1:i2] = Losses1 / 2
                    W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
                del W1, Q1, Err1, Losses1
                if Hinv is not None:
                    del Hinv1

        if Hinv is not None:
            del Hinv
            self._qr_R = None
            if self.nsamples != 0:
                avg_loss = torch.sum(Losses).item() / self.nsamples
                if math.isnan(avg_loss):
                    if fallback_configured:
                        log.info(
                            "Quantization: Failed due to `NaN` loss for `%s`, use mock quantization retry.",
                            self.name,
                        )
                        self.qcfg.mock_quantization = True
                        return self.quantize(blocksize=blocksize)
                    raise ValueError(
                        f"Quantization: Failed due to `NaN` loss for `{self.name}`, "
                        "please try increasing calibration data samples or enable fallback=True"
                    )
            else:
                if fallback_configured:
                    log.warn(
                        "Quantization: Module `%s` -> using fail safe mode. "
                        "Please check if calibration data is sufficient.",
                        self.name,
                    )
                else:
                    log.warn(
                        "Quantization: `%s` is not activated due to model inference logic (MoE)",
                        self.name,
                    )
                avg_loss = f"{resolved_strategy.value} fallback" if fallback_configured else 999999999
        else:
            avg_loss = f"{resolved_strategy.value} fallback" if fallback_configured else 999999999

        del Losses
        del self.H
        del W

        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns

        if self.qcfg.static_groups and self.qcfg.desc_act:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]

        g_idx_tensor = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

        if self.qcfg.desc_act and use_hessian:
            invperm = invperm.to(device=Q.device)
            Q = Q[:, invperm]
            g_idx_tensor = g_idx_tensor[invperm]
            del perm, invperm

        elif self.qcfg.act_group_aware and use_hessian:
            inv_final = invert_perm(final_perm).to(device=Q.device)
            Q = Q[:, inv_final]
            inv_global_perm = invert_perm(global_perm)
            inv_global_perm_list = inv_global_perm.tolist()
            reordered_group_count = len(inv_global_perm_list)
            temp_scale = [scale[i] for i in inv_global_perm_list]
            temp_scale.extend(scale[reordered_group_count:])
            scale = temp_scale
            temp_zero = [zero[i] for i in inv_global_perm_list]
            temp_zero.extend(zero[reordered_group_count:])
            zero = temp_zero
            del final_perm, inv_final, global_perm, inv_global_perm, inv_global_perm_list, local_perms

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            Q = Q[:, :valid_cols]
            g_idx_tensor = g_idx_tensor[:valid_cols]

        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()

        if Q.shape != self.module.weight.shape:
            Q = Q.reshape(self.module.weight.shape).to(self.module.weight.dtype)
        else:
            Q = Q.to(self.module.weight.dtype)

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        q_scales = torch.cat(scale, dim=1)
        q_zeros = torch.cat(zero, dim=1)

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            q_scales = self.truncate_last_dim(q_scales, valid_cols)
            q_zeros = self.truncate_last_dim(q_zeros, valid_cols)

        if cpu_fallback_used and Q.device != result_device:
            log.info(
                "Quantization: Module `%s` -> CPU fallback complete; moving final quantized weights back to %s.",
                self.name,
                result_device,
            )

        Q = Q.to(device=result_device, non_blocking=False)
        duration = time.time() - start

        return GptqQuantizeResult(
            pack_weight=Q,
            q_scales=q_scales,
            q_zeros=q_zeros,
            q_g_idx=g_idx_tensor,
            duration=duration,
            avg_loss=avg_loss,
            damp=damp,
            nsamples=self.nsamples,
        )

    def free(self) -> None:
        del self.quantizer
