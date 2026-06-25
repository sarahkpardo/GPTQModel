# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""FP-Quant / QuTLass export adapter (optional dependency)."""

from __future__ import annotations

import torch.nn as nn

from ..config import ExportTargetConfig
from ..context import TransformState, WeightQuantState


class FpQuantExport:
    """
    Serialize checkpoints compatible with FP-Quant inference_lib / QuTLass.

    When ``fp_quant`` is not installed, stores pseudo-quant metadata only.
    """

    def __init__(self, cfg: ExportTargetConfig) -> None:
        self.cfg = cfg
        self._pseudo = bool(cfg.options.get("pseudoquantization", True))

    def pack_module(
        self,
        *,
        module_name: str,
        submodule: nn.Module,
        transform: TransformState | None,
        weight_quant: WeightQuantState,
        model: nn.Module,
    ) -> None:
        try:
            from fp_quant import FPQuantConfig, replace_with_fp_quant_linear  # type: ignore
        except ImportError:
            if isinstance(submodule, nn.Module):
                meta = getattr(submodule, "_ptq_fpquant_meta", None)
                if meta is None:
                    setattr(
                        submodule,
                        "_ptq_fpquant_meta",
                        {
                            "module_name": module_name,
                            "pseudoquantization": self._pseudo,
                            "transform": None if transform is None else transform.method,
                        },
                    )
            return

        fp_cfg = FPQuantConfig(pseudoquantization=self._pseudo)
        replace_with_fp_quant_linear(model, fp_quant_linear_config=fp_cfg)
        del transform, weight_quant, module_name
