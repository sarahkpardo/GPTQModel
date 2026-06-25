# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPTQ weight optimizer backend wrapping the existing GPTQ class."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ...quantization.gptq import GPTQ
from ..config import WeightQuantizeTargetConfig
from ..context import ModuleCalibContext, TransformState, WeightQuantState


def _module_label(module: nn.Module) -> str:
    return str(getattr(module, "full_name", getattr(module, "name", module.__class__.__name__)))


class GptqWeightOptimizer:
    """Run standard GPTQ using statistics from ModuleCalibContext when available."""

    requires_calibration = True

    def __init__(self, *, qcfg, target: Optional[WeightQuantizeTargetConfig] = None) -> None:
        self.qcfg = qcfg
        self.target = target

    def optimize(
        self,
        *,
        module: nn.Module,
        ctx: ModuleCalibContext,
        transform: TransformState | None,
        device: torch.device,
        qcfg=None,
        expected_nsamples: Optional[float] = None,
    ) -> WeightQuantState:
        del transform
        active_qcfg = qcfg or self.qcfg
        module_name = _module_label(module)

        if ctx.nsamples <= 0 or ctx.H is None:
            raise ValueError(
                f"GPTQ weight optimizer requires calibration statistics for `{module_name}`, "
                f"observed nsamples={ctx.nsamples}, H={'set' if ctx.H is not None else 'missing'}."
            )

        gptq = GPTQ(module, qcfg=active_qcfg)
        gptq.fallback = None
        gptq.expected_nsamples = expected_nsamples
        gptq.quantizer.configure(perchannel=True)
        gptq.module.weight.data = gptq.module.weight.data.to(device)

        gptq.H = ctx.H.to(device=device, dtype=torch.float32)
        gptq.nsamples = ctx.nsamples
        gptq._hessian_dirty = False
        if ctx.qr_R is not None:
            gptq._qr_R = ctx.qr_R.to(device=device, dtype=torch.float32)

        blocksize = 128
        wq, q_scales, q_zeros, q_g_idx, duration, avg_loss, damp, nsamples = gptq.quantize(
            blocksize=blocksize
        )

        return WeightQuantState(
            q_scales=q_scales,
            q_zeros=q_zeros,
            q_g_idx=q_g_idx,
            pack_weight=wq,
            extra={
                "loss": avg_loss,
                "gptq": gptq,
                "duration": duration,
                "damp": damp,
                "nsamples": nsamples,
            },
        )
