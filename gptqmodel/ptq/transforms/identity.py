# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable

import torch

from ..context import ModuleCalibContext, TransformState
from ..inference_data import InferenceTransformData
from ..protocols import TransformMode


class IdentityTransform:
    """No-op transform backend."""

    def fit(
        self,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        ctx: ModuleCalibContext,
        mode: TransformMode,
        device: torch.device,
    ) -> TransformState:
        del weight, bias, ctx, mode, device
        return TransformState(method="identity", bake_weights=True, payload={})

    def apply_to_weights(
        self,
        weight: torch.Tensor,
        state: TransformState,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        del state, device
        return weight

    def transform_hessian(self, H: torch.Tensor, state: TransformState) -> torch.Tensor:
        del state
        return H

    def get_inference_data(self, state: TransformState) -> InferenceTransformData:
        del state
        return InferenceTransformData(transform_type="identity")

    def activation_pre_hook(self, state: TransformState) -> Callable[..., None]:
        del state

        def _noop(*_args, **_kwargs):
            return None

        return _noop
