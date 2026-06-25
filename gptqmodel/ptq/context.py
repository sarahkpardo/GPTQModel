# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shared calibration context passed between PTQ pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class TransformState:
    """Learned transform parameters for one module."""

    method: str
    bake_weights: bool = True
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightQuantState:
    """Quantized weight tensors and metadata from a weight optimizer."""

    q_scales: torch.Tensor
    q_zeros: torch.Tensor
    q_g_idx: Optional[torch.Tensor] = None
    pack_weight: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleCalibContext:
    """Per-module calibration statistics and stage outputs.

    ``nsamples`` is the observed activation row count captured for this module
    (used for Hessian normalization). ``expected_calibration_tokens`` is the
    global calibration token total from the dataset (diagnostics only).
    """

    module_name: str
    columns: int
    rows: int
    nsamples: int = 0
    expected_calibration_tokens: Optional[int] = None
    H: Optional[torch.Tensor] = None
    qr_R: Optional[torch.Tensor] = None
    row_buffer: Optional[torch.Tensor] = None
    transform: Optional[TransformState] = None
    weight_quant: Optional[WeightQuantState] = None

    def gram_from_qr(self) -> Optional[torch.Tensor]:
        """Materialize H = R^T R when only the QR factor is stored."""
        if self.qr_R is None:
            return None
        R = self.qr_R.to(dtype=torch.float32)
        return R.transpose(-1, -2) @ R
