# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPTQ weight optimizer backend wrapping the GptqSolver."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import WeightQuantizeTargetConfig
from ..context import ModuleCalibContext, TransformState, WeightQuantState
from ..solvers.gptq import GptqSolver


def _module_label(module: nn.Module) -> str:
    return str(getattr(module, "full_name", getattr(module, "name", module.__class__.__name__)))


class GptqWeightOptimizer:
    """Run standard GPTQ using statistics from ModuleCalibContext when available."""

    requires_calibration = True

    def __init__(self, *, qcfg, target: Optional[WeightQuantizeTargetConfig] = None) -> None:
        self.qcfg = qcfg
        self.target = target
        self._solver = GptqSolver()

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
        active_qcfg = qcfg or self.qcfg
        module_name = _module_label(module)

        if ctx.nsamples <= 0 or ctx.H is None:
            raise ValueError(
                f"GPTQ weight optimizer requires calibration statistics for `{module_name}`, "
                f"observed nsamples={ctx.nsamples}, H={'set' if ctx.H is not None else 'missing'}."
            )

        result = self._solver.solve(
            module=module,
            qcfg=active_qcfg,
            H=ctx.H,
            nsamples=ctx.nsamples,
            qr_R=ctx.qr_R,
            expected_nsamples=expected_nsamples,
        )

        inference_transform = None
        if transform is not None:
            inference_transform = transform.inference
            if inference_transform is None and not transform.bake_weights:
                from ..transforms.registry import build_transform_backend
                from ..config import TransformPrepareConfig

                backend = build_transform_backend(TransformPrepareConfig(method=transform.method))
                inference_transform = backend.get_inference_data(transform)

        return WeightQuantState(
            q_scales=result.q_scales,
            q_zeros=result.q_zeros,
            q_g_idx=result.q_g_idx,
            pack_weight=result.pack_weight,
            extra={
                "loss": result.avg_loss,
                "gptq": result.backend,
                "duration": result.duration,
                "damp": result.damp,
                "nsamples": result.nsamples,
                "inference_transform": inference_transform,
            },
        )
