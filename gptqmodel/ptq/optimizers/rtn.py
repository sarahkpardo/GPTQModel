# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Round-to-nearest weight optimizer for the PTQ split pipeline."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ...quantization.config import FORMAT, RTNConfig
from ...quantization.rtn import RTN
from ..config import WeightQuantizeTargetConfig
from ..context import ModuleCalibContext, TransformState, WeightQuantState


def rtn_config_from_qcfg(qcfg) -> RTNConfig:
    """Build an RTNConfig from a GPTQ-style QuantizeConfig."""
    return RTNConfig(
        bits=qcfg.bits,
        sym=qcfg.sym,
        group_size=qcfg.group_size,
        format=getattr(qcfg, "format", FORMAT.GPTQ),
        device=getattr(qcfg, "device", None),
        smooth=getattr(qcfg, "smooth", None),
        damp_percent=getattr(qcfg, "damp_percent", None),
        desc_act=getattr(qcfg, "desc_act", False),
        dynamic=getattr(qcfg, "dynamic", None),
        pack_dtype=getattr(qcfg, "pack_dtype", None),
    )


class RtnWeightOptimizer:
    """Weight-only RTN quantizer for protocol-aligned PTQ pipelines."""

    requires_calibration = False

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
    ) -> WeightQuantState:
        del ctx, transform, device
        rtn_cfg = rtn_config_from_qcfg(self.qcfg)
        rtn = RTN(module, rtn_cfg)
        wq, q_scales, q_zeros, q_g_idx, _duration, avg_loss, _damp, _nsamples = rtn.quantize()
        return WeightQuantState(
            q_scales=q_scales,
            q_zeros=q_zeros,
            q_g_idx=q_g_idx,
            pack_weight=wq,
            extra={"loss": avg_loss, "rtn": rtn},
        )
