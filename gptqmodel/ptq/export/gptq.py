# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPTQ/Marlin/Triton export — delegates to existing GPTQProcessor packing."""

from __future__ import annotations

import torch.nn as nn

from ...looper.named_module import NamedModule
from ..config import ExportTargetConfig
from ..context import TransformState, WeightQuantState
from ..inference_data import InferenceTransformData
from ..inference_hooks import persist_ptq_inference_buffers, register_activation_pre_hook


class GptqExport:
    """Attach inference transform metadata during GPTQ module packing."""

    def __init__(self, cfg: ExportTargetConfig) -> None:
        self.cfg = cfg

    def pack_module(
        self,
        *,
        module_name: str,
        submodule: nn.Module,
        transform: TransformState | None,
        weight_quant: WeightQuantState,
        inference: InferenceTransformData | None = None,
        model: nn.Module,
    ) -> None:
        del module_name, model, self.cfg
        target = submodule.module if isinstance(submodule, NamedModule) else submodule
        inference_data = inference
        if inference_data is None:
            extra = weight_quant.extra.get("inference_transform")
            if isinstance(extra, InferenceTransformData):
                inference_data = extra
            elif isinstance(extra, dict):
                inference_data = InferenceTransformData.from_dict(extra)
            elif transform is not None and transform.inference is not None:
                inference_data = transform.inference

        if inference_data is None or inference_data.is_identity():
            return

        method = transform.method if transform is not None else inference_data.transform_type
        pad = 0
        if transform is not None and transform.payload:
            pad = int(transform.payload.get("pad", 0) or 0)
        elif inference_data.extra:
            pad = int(inference_data.extra.get("pad", 0) or 0)

        persist_ptq_inference_buffers(
            target,
            inference_data,
            method=method,
            pad=pad,
        )
        register_activation_pre_hook(target, transform, inference=inference_data)
        if isinstance(submodule, NamedModule):
            submodule.state["ptq_inference_transform"] = inference_data.to_dict()
        else:
            setattr(target, "_ptq_inference_transform", inference_data.to_dict())
