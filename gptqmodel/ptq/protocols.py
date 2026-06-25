# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Protocol definitions for composable PTQ backends."""

from __future__ import annotations

from typing import Callable, Literal, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

from .context import ModuleCalibContext, TransformState, WeightQuantState


TransformMode = Literal["standalone", "e2e"]


@runtime_checkable
class TransformBackend(Protocol):
    """Learn/fit transforms on weights and/or activations."""

    def fit(
        self,
        *,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        ctx: ModuleCalibContext,
        mode: TransformMode,
        device: torch.device,
    ) -> TransformState: ...

    def apply_to_weights(
        self,
        weight: torch.Tensor,
        state: TransformState,
        *,
        device: torch.device,
    ) -> torch.Tensor: ...

    def activation_pre_hook(self, state: TransformState) -> Callable[..., None]:
        """Return a hook applying T_X at inference when transform is not baked."""
        ...


@runtime_checkable
class WeightOptimizerBackend(Protocol):
    """Find quantized weights given statistics and (possibly transformed) float weights."""

    def optimize(
        self,
        *,
        module: nn.Module,
        ctx: ModuleCalibContext,
        transform: TransformState | None,
        device: torch.device,
    ) -> WeightQuantState: ...


@runtime_checkable
class ExportBackend(Protocol):
    """Pack module tensors for a specific inference runtime."""

    def pack_module(
        self,
        *,
        module_name: str,
        submodule: nn.Module,
        transform: TransformState | None,
        weight_quant: WeightQuantState,
        model: nn.Module,
    ) -> None: ...
