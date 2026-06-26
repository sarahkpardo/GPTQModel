# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Optional

import torch

from ...quantization.paroquant.optimization import optimize_paroquant_linear
from ..config import TransformPrepareConfig
from ..context import ModuleCalibContext, TransformState
from ..protocols import TransformMode

PAROQUANT_PAYLOAD_KEYS = (
    "pseudo_weight",
    "pack_weight",
    "q_scales",
    "q_zeros",
    "pairs",
    "theta",
    "channel_scales",
    "train_loss",
    "val_loss",
)


def module_seed_from_options(*, module_name: str, options: Dict[str, Any]) -> int:
    """Match legacy ParoQuantProcessor module-scope seed derivation."""
    seed = int(options.get("opt_seed", 0))
    layer_index = int(options.get("opt_layer_index", 0))
    scope = str(options.get("opt_scope", "module"))
    if scope == "module":
        seed_key = str(options.get("opt_module_seed_key", module_name))
    else:
        seed_key = str(options.get("opt_module_seed_key", module_name.rsplit(".", 1)[-1]))
    material = f"{seed}:{layer_index}:{seed_key}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), byteorder="big", signed=False)


def resolve_paroquant_activation_inputs(
    *,
    ctx: ModuleCalibContext,
    weight: torch.Tensor,
) -> torch.Tensor:
    inputs = ctx.row_buffer if ctx.row_buffer is not None else torch.empty(0)
    if inputs.numel() == 0 and ctx.H is not None:
        inputs = torch.empty((0, ctx.columns), dtype=weight.dtype, device=weight.device)
    if inputs.numel() == 0:
        return torch.empty((0, weight.shape[1]), dtype=weight.dtype, device=weight.device)
    inputs = inputs.to(device=weight.device)
    if hasattr(inputs, "is_inference") and inputs.is_inference():
        inputs = inputs.clone()
    return inputs


def resolve_paroquant_quant_params(
    *,
    options: Dict[str, Any],
    module_name: str,
    qcfg=None,
) -> tuple[int, int, bool]:
    bits = int(options.get("bits", 4))
    group_size = int(options.get("group_size", 128))
    sym = bool(options.get("sym", True))
    if qcfg is not None and getattr(qcfg, "dynamic", None):
        bits = int(qcfg.dynamic_get(module_name, "bits", bits))
        group_size = int(qcfg.dynamic_get(module_name, "group_size", group_size))
        sym = bool(qcfg.dynamic_get(module_name, "sym", sym))
    return bits, group_size, sym


def build_paroquant_optimize_kwargs(
    *,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    ctx: ModuleCalibContext,
    cfg: TransformPrepareConfig,
    qcfg=None,
) -> Dict[str, Any]:
    """Single source of truth for kwargs passed to optimize_paroquant_linear."""
    options = dict(cfg.options)
    bits, group_size, sym = resolve_paroquant_quant_params(
        options=options,
        module_name=ctx.module_name,
        qcfg=qcfg,
    )
    inputs = resolve_paroquant_activation_inputs(ctx=ctx, weight=weight)

    finetune_epochs = int(options.get("opt_finetune_epochs", 0))
    if cfg.mode == "e2e" and "opt_finetune_epochs" not in options:
        finetune_epochs = int(options.get("opt_finetune_epochs_default", 10))

    betas = options.get("opt_betas", (0.9, 0.95))
    if isinstance(betas, list):
        betas = tuple(betas)

    return {
        "weight": weight,
        "bias": bias,
        "inputs": inputs,
        "bits": bits,
        "group_size": group_size,
        "sym": sym,
        "krot": int(options.get("krot", 8)),
        "pair_ratio": float(options.get("opt_pair_ratio", 0.5)),
        "train_rows": int(options.get("opt_train_samples", 2048)),
        "val_rows": int(options.get("opt_validation_samples", 64)),
        "batch_size": int(options.get("opt_batch_size", 64)),
        "rotation_epochs": int(options.get("opt_rotation_epochs", 10)),
        "finetune_epochs": finetune_epochs,
        "rotation_lr": float(options.get("opt_rotation_lr", 0.05)),
        "weight_lr": float(options.get("opt_weight_lr", 1e-5)),
        "quantizer_lr": float(options.get("opt_quantizer_lr", 1e-6)),
        "seed": module_seed_from_options(module_name=ctx.module_name, options=options),
        "optimizer_name": str(options.get("opt_optimizer", "adamw")),
        "optimizer_weight_decay": float(options.get("opt_weight_decay", 0.01)),
        "optimizer_betas": (float(betas[0]), float(betas[1])),
        "optimizer_eps": float(options.get("opt_eps", 1e-10)),
        "optimizer_amsgrad": bool(options.get("opt_amsgrad", False)),
        "sgd_momentum": float(options.get("opt_sgd_momentum", 0.0)),
        "sgd_dampening": float(options.get("opt_sgd_dampening", 0.0)),
        "sgd_nesterov": bool(options.get("opt_sgd_nesterov", False)),
        "fused_rotation": bool(options.get("opt_fused_rotation", True)),
        "gradient_checkpointing": bool(options.get("opt_gradient_checkpointing", False)),
        "stage_cudagraph": options.get("opt_stage_cudagraph"),
        "stage_impl": str(options.get("opt_stage_impl", "fast")),
        "pair_impl": str(options.get("opt_pair_impl", "fast")),
        "quantizer_impl": str(options.get("opt_quantizer_impl", "reference")),
        "best_state_dtype": options.get("opt_best_state_dtype", "fp32"),
        "scale_clamp_min": float(options.get("opt_channel_scale_clamp_min", 1e-2)),
        "scale_clamp_max": float(options.get("opt_channel_scale_clamp_max", 1e2)),
    }


def paroquant_result_to_payload(result) -> Dict[str, Any]:
    return {
        "pseudo_weight": result.pseudo_weight.detach().cpu(),
        "pack_weight": result.pack_weight.detach().cpu(),
        "q_scales": result.q_scales.detach().cpu(),
        "q_zeros": result.q_zeros.detach().cpu(),
        "pairs": result.pairs.detach().cpu(),
        "theta": result.theta.detach().cpu(),
        "channel_scales": result.channel_scales.detach().cpu(),
        "train_loss": result.train_loss,
        "val_loss": result.val_loss,
    }


class ParoQuantTransform:
    """ParoQuant rotation/scaling transform as a PTQ prepare backend."""

    def __init__(self, cfg: TransformPrepareConfig) -> None:
        self.cfg = cfg
        self._options = dict(cfg.options)
        self._qcfg = None

    def fit(
        self,
        *,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        ctx: ModuleCalibContext,
        mode: TransformMode,
        device: torch.device,
        qcfg=None,
    ) -> TransformState:
        del mode, device
        active_qcfg = qcfg or self._qcfg
        kwargs = build_paroquant_optimize_kwargs(
            weight=weight,
            bias=bias,
            ctx=ctx,
            cfg=self.cfg,
            qcfg=active_qcfg,
        )
        result = optimize_paroquant_linear(**kwargs)

        return TransformState(
            method="paroquant",
            bake_weights=self.cfg.bake_weights,
            payload=paroquant_result_to_payload(result),
        )

    def apply_to_weights(
        self,
        weight: torch.Tensor,
        state: TransformState,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        pseudo = state.payload.get("pseudo_weight")
        if pseudo is None:
            return weight
        return pseudo.to(device=device, dtype=weight.dtype)

    def transform_hessian(self, H: torch.Tensor, state: TransformState) -> torch.Tensor:
        del state
        return H

    def get_inference_data(self, state: TransformState) -> "InferenceTransformData":
        from ..inference_data import InferenceTransformData

        pairs = state.payload.get("pairs")
        theta = state.payload.get("theta")
        return InferenceTransformData(
            transform_type="givens",
            T_X_pairs=pairs if isinstance(pairs, torch.Tensor) else None,
            T_X_angles=theta if isinstance(theta, torch.Tensor) else None,
            precision=torch.float16,
        )

    def activation_pre_hook(self, state: TransformState) -> Callable[..., None]:
        del state

        def _noop(*_args, **_kwargs):
            return None

        return _noop
