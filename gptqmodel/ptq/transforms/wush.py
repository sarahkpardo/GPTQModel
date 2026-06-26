# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""WUSH transform adapter stub."""

from __future__ import annotations

from typing import Callable

import torch

from ..config import TransformPrepareConfig
from ..context import ModuleCalibContext, TransformState
from ..protocols import TransformMode
from .identity import IdentityTransform


class WUSHTransform(IdentityTransform):
    """Placeholder WUSH backend until WUSH math is vendored."""

    def __init__(self, cfg: TransformPrepareConfig) -> None:
        del cfg

    def fit(
        self,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        ctx: ModuleCalibContext,
        mode: TransformMode,
        device: torch.device,
    ) -> TransformState:
        state = super().fit(weight=weight, bias=bias, ctx=ctx, mode=mode, device=device)
        state.method = "wush"
        return state

    def activation_pre_hook(self, state: TransformState) -> Callable[..., None]:
        return super().activation_pre_hook(state)

    def transform_hessian(self, H: torch.Tensor, state: TransformState) -> torch.Tensor:
        del state
        raise NotImplementedError("WUSH transform_hessian is not implemented yet.")

    def get_inference_data(self, state: TransformState) -> "InferenceTransformData":
        from ..inference_data import InferenceTransformData

        del state
        return InferenceTransformData(transform_type="wush")
