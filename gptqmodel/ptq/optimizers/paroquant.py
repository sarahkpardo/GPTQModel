# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Passthrough weight optimizer for ParoQuant transform payloads."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import WeightQuantizeTargetConfig
from ..context import ModuleCalibContext, TransformState, WeightQuantState


def _module_label(module: nn.Module) -> str:
    return str(getattr(module, "full_name", getattr(module, "name", module.__class__.__name__)))


class ParoQuantWeightOptimizer:
    """Reuse ParoQuant transform tensors without a second GPTQ pass."""

    requires_calibration = True

    def __init__(self, *, qcfg, target: Optional[WeightQuantizeTargetConfig] = None) -> None:
        del target
        self.qcfg = qcfg

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
        del qcfg, expected_nsamples, device
        module_name = _module_label(module)

        if transform is None or transform.method != "paroquant":
            raise ValueError(
                f"ParoQuant weight optimizer requires a paroquant TransformState for `{module_name}`."
            )

        payload = transform.payload
        required = (
            "pack_weight",
            "q_scales",
            "q_zeros",
            "pairs",
            "theta",
            "channel_scales",
            "pseudo_weight",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(
                f"ParoQuant transform payload for `{module_name}` is missing keys: {missing}."
            )

        pseudo = payload["pseudo_weight"]
        if isinstance(pseudo, torch.Tensor):
            module.weight.data = pseudo.to(device=module.weight.device, dtype=module.weight.dtype)

        pack_weight = payload["pack_weight"]
        q_scales = payload["q_scales"]
        q_zeros = payload["q_zeros"]

        return WeightQuantState(
            q_scales=q_scales,
            q_zeros=q_zeros,
            q_g_idx=None,
            pack_weight=pack_weight if isinstance(pack_weight, torch.Tensor) else None,
            extra={
                "loss": payload.get("train_loss", 0.0),
                "paroquant": True,
                "train_loss": payload.get("train_loss"),
                "val_loss": payload.get("val_loss"),
                "inference_transform": transform.inference,
            },
        )
