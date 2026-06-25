# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pure GPTQ column-wise solver using precomputed Hessian statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from ...quantization.gptq import GPTQ


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
        gptq = GPTQ(module, qcfg=qcfg)
        gptq.fallback = None
        gptq.expected_nsamples = expected_nsamples
        gptq.quantizer.configure(perchannel=True)
        gptq.module.weight.data = gptq.module.weight.data.to(H.device)

        gptq.H = H.to(device=H.device, dtype=torch.float32)
        gptq.nsamples = int(nsamples)
        gptq._hessian_dirty = False
        if qr_R is not None:
            gptq._qr_R = qr_R.to(device=H.device, dtype=torch.float32)

        wq, q_scales, q_zeros, q_g_idx, duration, avg_loss, damp, out_nsamples = gptq.quantize(
            blocksize=blocksize
        )

        return GptqSolveResult(
            pack_weight=wq,
            q_scales=q_scales,
            q_zeros=q_zeros,
            q_g_idx=q_g_idx,
            avg_loss=avg_loss,
            damp=damp,
            nsamples=out_nsamples,
            duration=duration,
            backend=gptq,
        )
