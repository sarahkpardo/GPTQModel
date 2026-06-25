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


class GptqWeightOptimizer:
    """Run standard GPTQ using statistics from ModuleCalibContext when available."""

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
    ) -> WeightQuantState:
        del transform
        gptq = GPTQ(module, qcfg=self.qcfg)
        gptq.module.weight.data = gptq.module.weight.data.to(device)

        if ctx.H is not None and ctx.nsamples > 0:
            gptq.H = ctx.H.to(device=device, dtype=torch.float32)
            gptq.nsamples = ctx.nsamples
            gptq._hessian_dirty = False
            if ctx.qr_R is not None:
                gptq._qr_R = ctx.qr_R.to(device=device, dtype=torch.float32)
        else:
            gptq.finalize_hessian(target_device=device)

        blocksize = 128
        gptq.quantize(blocksize=blocksize)

        named = getattr(gptq, "_named_module", None)
        state = named.state if named is not None else {}

        return WeightQuantState(
            q_scales=state["q_scales"],
            q_zeros=state["q_zeros"],
            q_g_idx=state.get("q_g_idx"),
            pack_weight=state.get("wq"),
            extra={"loss": state.get("quant_loss"), "gptq": gptq},
        )
