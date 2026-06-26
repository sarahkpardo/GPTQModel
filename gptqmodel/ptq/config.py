# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Protocol-aligned PTQ configuration fragments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..quantization.config import HessianConfig


@dataclass
class TransformPrepareConfig:
    """One transform step in the weight.prepare pipeline."""

    method: str
    mode: str = "standalone"
    bake_weights: bool = True
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Union[Dict[str, Any], "TransformPrepareConfig"]) -> "TransformPrepareConfig":
        if isinstance(payload, TransformPrepareConfig):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("TransformPrepareConfig: expected dict or TransformPrepareConfig.")
        method = str(payload.get("method", "")).strip().lower()
        if not method:
            raise ValueError("TransformPrepareConfig: `method` is required.")
        return cls(
            method=method,
            mode=str(payload.get("mode", "standalone")).strip().lower(),
            bake_weights=bool(payload.get("bake_weights", True)),
            options={
                k: v
                for k, v in payload.items()
                if k not in {"method", "mode", "bake_weights"}
            },
        )


@dataclass
class WeightQuantizeTargetConfig:
    """Weight quantizer selection (maps to protocol weight.quantize)."""

    method: str = "gptq"
    hessian: Optional[HessianConfig] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportTargetConfig:
    """Export/runtime selection (maps to protocol weight.export)."""

    format: str = "gptq"
    impl: str = "default"
    options: Dict[str, Any] = field(default_factory=dict)


def normalize_transform_prepare(
    payload: Optional[Union[TransformPrepareConfig, Dict[str, Any], List[Any]]],
) -> List[TransformPrepareConfig]:
    if payload is None:
        return []
    if isinstance(payload, TransformPrepareConfig):
        return [payload]
    if isinstance(payload, dict):
        return [TransformPrepareConfig.from_dict(payload)]
    if isinstance(payload, list):
        return [TransformPrepareConfig.from_dict(item) for item in payload]
    raise ValueError("GPTQConfig: `weight_prepare` must be a dict, list, or TransformPrepareConfig.")


def resolve_weight_quantize_target(qcfg) -> WeightQuantizeTargetConfig:
    """Resolve the weight quantizer target from a QuantizeConfig."""
    raw = getattr(qcfg, "weight_quantize", None)
    if raw is None:
        method = "gptq"
        if getattr(qcfg, "gptaq", None) is not None:
            method = "gptaq"
        elif getattr(qcfg, "foem", None) is not None:
            method = "foem"
        return WeightQuantizeTargetConfig(method=method)
    if isinstance(raw, WeightQuantizeTargetConfig):
        return raw
    if isinstance(raw, dict):
        payload = dict(raw)
        method = str(payload.pop("method", "gptq")).strip().lower()
        hessian = payload.pop("hessian", None)
        return WeightQuantizeTargetConfig(method=method, hessian=hessian, options=payload)
    raise ValueError("GPTQConfig: `weight_quantize` must be a dict or WeightQuantizeTargetConfig.")


def paro_prepare_from_config(paro, *, mode: str = "standalone") -> TransformPrepareConfig:
    """Build a transform prepare entry from an existing ParoConfig."""
    return TransformPrepareConfig(
        method="paroquant",
        mode=mode,
        bake_weights=True,
        options={
            "krot": paro.krot,
            "opt_rotation_epochs": paro.opt_rotation_epochs,
            "opt_finetune_epochs": 0 if mode == "standalone" else paro.opt_finetune_epochs,
            "opt_train_samples": paro.opt_train_samples,
            "opt_validation_samples": paro.opt_validation_samples,
            "opt_batch_size": paro.opt_batch_size,
            "opt_rotation_lr": paro.opt_rotation_lr,
            "opt_weight_lr": paro.opt_weight_lr,
            "opt_quantizer_lr": paro.opt_quantizer_lr,
            "opt_pair_ratio": paro.opt_pair_ratio,
            "opt_seed": paro.opt_seed,
            "opt_optimizer": paro.opt_optimizer,
            "opt_scope": paro.opt_scope,
            "opt_stage_impl": paro.opt_stage_impl,
            "opt_pair_impl": paro.opt_pair_impl,
            "opt_quantizer_impl": paro.opt_quantizer_impl,
            "opt_fused_rotation": paro.opt_fused_rotation,
            "opt_gradient_checkpointing": paro.opt_gradient_checkpointing,
            "opt_channel_scale_clamp_min": paro.opt_channel_scale_clamp_min,
            "opt_channel_scale_clamp_max": paro.opt_channel_scale_clamp_max,
            "opt_weight_decay": paro.opt_weight_decay,
            "opt_betas": paro.opt_betas,
            "opt_eps": paro.opt_eps,
            "opt_amsgrad": paro.opt_amsgrad,
            "opt_sgd_momentum": paro.opt_sgd_momentum,
            "opt_sgd_dampening": paro.opt_sgd_dampening,
            "opt_sgd_nesterov": paro.opt_sgd_nesterov,
            "opt_best_state_dtype": paro.opt_best_state_dtype,
            "opt_layer_index": 0,
        },
    )
