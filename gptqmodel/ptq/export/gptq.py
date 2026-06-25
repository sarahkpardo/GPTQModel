# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPTQ/Marlin/Triton export — delegates to existing GPTQProcessor packing."""

from __future__ import annotations

import torch.nn as nn

from ...looper.named_module import NamedModule
from ..config import ExportTargetConfig
from ..context import TransformState, WeightQuantState


class GptqExport:
    """No-op export backend; GPTQProcessor.submodule_finalize performs packing."""

    def __init__(self, cfg: ExportTargetConfig) -> None:
        self.cfg = cfg

    def pack_module(
        self,
        *,
        module_name: str,
        submodule: nn.Module,
        transform: TransformState | None,
        weight_quant: WeightQuantState,
        model: nn.Module,
    ) -> None:
        del module_name, submodule, transform, weight_quant, model, self.cfg
