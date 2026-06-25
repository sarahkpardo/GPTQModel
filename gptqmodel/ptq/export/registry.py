# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..config import ExportTargetConfig
from ..protocols import ExportBackend
from .gptq import GptqExport
from .paroquant import ParoQuantExport


def build_export_backend(cfg: ExportTargetConfig) -> ExportBackend:
    fmt = str(cfg.format).strip().lower()
    impl = str(cfg.impl).strip().lower()
    if fmt in {"gptq", "auto"} and impl not in {"paroquant", "fpquant", "qutlass"}:
        return GptqExport(cfg)
    if fmt in {"paroquant", "paro"} or impl == "paroquant":
        return ParoQuantExport(cfg)
    if fmt in {"fpquant", "mxfp4", "nvfp4", "fp4"} or impl in {"fpquant", "qutlass"}:
        from .fpquant import FpQuantExport

        return FpQuantExport(cfg)
    return GptqExport(cfg)
