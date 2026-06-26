# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Library-level PTQ orchestration callable without the module looper."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .config import TransformPrepareConfig, WeightQuantizeTargetConfig, normalize_transform_prepare, resolve_weight_quantize_target
from .context import ModuleCalibContext, TransformState, WeightQuantState
from .optimizers.registry import build_weight_optimizer
from .stats import StatisticsCollector
from .transforms.registry import build_transform_backend
from ..quantization.config import QuantizeConfig


class ModuleQuantizationPipeline:
    """Run statistics → transform → quantize for one module from frozen collectors."""

    def __init__(
        self,
        *,
        qcfg: QuantizeConfig,
        prepare_configs: Optional[List[TransformPrepareConfig]] = None,
        weight_quantize: Optional[WeightQuantizeTargetConfig] = None,
    ) -> None:
        self.qcfg = qcfg
        self.prepare_configs = normalize_transform_prepare(
            prepare_configs or getattr(qcfg, "weight_prepare", None)
        )
        self.weight_quantize = weight_quantize

    def apply_transform(
        self,
        *,
        module: nn.Module,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        ctx: ModuleCalibContext,
        device: torch.device,
    ) -> tuple[torch.Tensor, TransformState]:
        if not self.prepare_configs:
            state = TransformState(method="identity", bake_weights=True)
            ctx.transform = state
            return weight, state

        transform_state = TransformState(method="identity", bake_weights=True)
        qcfg_options = {
            "bits": self.qcfg.bits,
            "group_size": self.qcfg.group_size,
            "sym": self.qcfg.sym,
        }
        for cfg in self.prepare_configs:
            merged_options = {**qcfg_options, **cfg.options}
            merged_cfg = TransformPrepareConfig(
                method=cfg.method,
                mode=cfg.mode,
                bake_weights=cfg.bake_weights,
                options=merged_options,
            )
            backend = build_transform_backend(merged_cfg)
            transform_state = backend.fit(
                weight=weight,
                bias=bias,
                ctx=ctx,
                mode=cfg.mode if cfg.mode in {"standalone", "e2e"} else "standalone",
                device=device,
            )
            inference = backend.get_inference_data(transform_state)
            transform_state.inference = inference
            if transform_state.bake_weights:
                weight = backend.apply_to_weights(weight, transform_state, device=device)
            if transform_state.method != "identity" and ctx.H is not None:
                ctx.H = backend.transform_hessian(ctx.H, transform_state)

        ctx.transform = transform_state
        return weight, transform_state

    def run_module(
        self,
        *,
        module: nn.Module,
        collector: StatisticsCollector,
        ctx: ModuleCalibContext,
        device: torch.device,
        expected_nsamples: Optional[float] = None,
    ) -> WeightQuantState:
        weight = module.weight.data
        bias = getattr(module, "bias", None)
        if bias is not None:
            bias = bias.data

        weight, transform_state = self.apply_transform(
            module=module,
            weight=weight,
            bias=bias,
            ctx=ctx,
            device=device,
        )
        module.weight.data = weight

        optimizer = build_weight_optimizer(
            self.weight_quantize or resolve_weight_quantize_target(self.qcfg),
            self.qcfg,
        )
        return optimizer.optimize(
            module=module,
            ctx=ctx,
            transform=transform_state,
            device=device,
            expected_nsamples=expected_nsamples,
        )
