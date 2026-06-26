# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pure GPTQ column-wise solver using precomputed Hessian statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from .gptq_quantizer import GptqQuantizer


@dataclass
class GptqSolveResult:
    """Quantized weight tensors produced by :class:`GptqSolver`."""

    pack_weight: torch.Tensor
    q_scales: torch.Tensor
    q_zeros: torch.Tensor
    q_g_idx: torch.Tensor
    avg_loss: float | str
    damp: float
    nsamples: int
    duration: float
    backend: Any


class GptqSolver:
    """Run GPTQ quantization from ``(H, qr_R)`` without inline activation capture."""

    def solve(
        self,
        *,
        module: nn.Module,
        qcfg,
        H: torch.Tensor,
        nsamples: int,
        qr_R: Optional[torch.Tensor] = None,
        expected_nsamples: Optional[float] = None,
        blocksize: int = 128,
    ) -> GptqSolveResult:
        quantizer = GptqQuantizer(
            module,
            qcfg=qcfg,
            H=H,
            nsamples=nsamples,
            qr_R=qr_R,
            expected_nsamples=expected_nsamples,
            fallback=None,
        )
        quantizer.module.weight.data = quantizer.module.weight.data.to(H.device)
        result = quantizer.quantize(blocksize=blocksize)

        return GptqSolveResult(
            pack_weight=result.pack_weight,
            q_scales=result.q_scales,
            q_zeros=result.q_zeros,
            q_g_idx=result.q_g_idx,
            avg_loss=result.avg_loss,
            damp=result.damp,
            nsamples=result.nsamples,
            duration=result.duration,
            backend=quantizer,
        )
