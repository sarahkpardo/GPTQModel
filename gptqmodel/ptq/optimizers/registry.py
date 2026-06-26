# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..config import WeightQuantizeTargetConfig
from ..protocols import WeightOptimizerBackend
from .gptq import GptqWeightOptimizer
from .paroquant import ParoQuantWeightOptimizer
from .rtn import RtnWeightOptimizer

CALIBRATION_REQUIRED_METHODS = frozenset({"gptq", "gptaq", "foem", "paroquant"})


def weight_optimizer_requires_calibration(cfg: WeightQuantizeTargetConfig) -> bool:
    """Return whether the selected weight optimizer needs calibration statistics."""
    return str(cfg.method).strip().lower() in CALIBRATION_REQUIRED_METHODS


def build_weight_optimizer(cfg: WeightQuantizeTargetConfig, qcfg) -> WeightOptimizerBackend:
    method = str(cfg.method).strip().lower()
    if method in {"gptq", "gptaq", "foem"}:
        return GptqWeightOptimizer(qcfg=qcfg, target=cfg)
    if method in {"paroquant", "paro"}:
        return ParoQuantWeightOptimizer(qcfg=qcfg, target=cfg)
    if method == "rtn":
        return RtnWeightOptimizer(qcfg=qcfg, target=cfg)
    raise ValueError(f"Unsupported weight optimizer `{cfg.method}`.")
