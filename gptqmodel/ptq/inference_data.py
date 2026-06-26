# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Inference-time transform payloads separate from offline TransformState."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class InferenceTransformData:
    """Kernel-facing transform metadata for unbaked activation transforms."""

    transform_type: str  # dense | givens | hadamard | factored | identity
    T_X_matrices: Optional[torch.Tensor] = None
    T_X_pairs: Optional[torch.Tensor] = None
    T_X_angles: Optional[torch.Tensor] = None
    T_X_scales: Optional[torch.Tensor] = None
    precision: torch.dtype = torch.float16
    block_size: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "transform_type": self.transform_type,
            "precision": str(self.precision),
            "block_size": int(self.block_size),
        }
        for key in ("T_X_matrices", "T_X_pairs", "T_X_angles", "T_X_scales"):
            tensor = getattr(self, key)
            if tensor is not None:
                payload[key] = tensor.detach().cpu()
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "InferenceTransformData":
        precision = payload.get("precision", torch.float16)
        if isinstance(precision, str):
            if precision == "float16":
                precision = torch.float16
            elif precision == "bfloat16":
                precision = torch.bfloat16
            else:
                precision = torch.float32

        def _tensor(key: str) -> Optional[torch.Tensor]:
            value = payload.get(key)
            return value if isinstance(value, torch.Tensor) else None

        return cls(
            transform_type=str(payload.get("transform_type", "identity")),
            T_X_matrices=_tensor("T_X_matrices"),
            T_X_pairs=_tensor("T_X_pairs"),
            T_X_angles=_tensor("T_X_angles"),
            T_X_scales=_tensor("T_X_scales"),
            precision=precision,
            block_size=int(payload.get("block_size", 0) or 0),
            extra=dict(payload.get("extra") or {}),
        )

    def is_identity(self) -> bool:
        return self.transform_type == "identity"
