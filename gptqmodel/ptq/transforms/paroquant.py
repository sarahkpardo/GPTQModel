# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from typing import Callable, Optional

import torch

from ...quantization.paroquant.optimization import optimize_paroquant_linear
from ..config import TransformPrepareConfig
from ..context import ModuleCalibContext, TransformState
from ..protocols import TransformMode


class ParoQuantTransform:
    """ParoQuant rotation/scaling transform as a PTQ prepare backend."""

    def __init__(self, cfg: TransformPrepareConfig) -> None:
        self.cfg = cfg
        self._options = dict(cfg.options)

    def _module_seed(self, module_name: str) -> int:
        seed = int(self._options.get("opt_seed", 0))
        material = f"{seed}:{module_name}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), byteorder="big", signed=False)

    def fit(
        self,
        *,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        ctx: ModuleCalibContext,
        mode: TransformMode,
        device: torch.device,
    ) -> TransformState:
        del mode
        bits = int(self._options.get("bits", 4))
        group_size = int(self._options.get("group_size", 128))
        sym = bool(self._options.get("sym", True))
        inputs = ctx.row_buffer if ctx.row_buffer is not None else torch.empty(0)
        if inputs.numel() == 0 and ctx.H is not None:
            inputs = torch.empty((0, ctx.columns), dtype=weight.dtype, device=weight.device)

        finetune_epochs = int(self._options.get("opt_finetune_epochs", 0))
        if self.cfg.mode == "e2e" and "opt_finetune_epochs" not in self._options:
            finetune_epochs = int(self._options.get("opt_finetune_epochs_default", 10))

        result = optimize_paroquant_linear(
            weight=weight,
            bias=bias,
            inputs=inputs.to(device=weight.device) if inputs.numel() else inputs,
            bits=bits,
            group_size=group_size,
            sym=sym,
            krot=int(self._options.get("krot", 8)),
            pair_ratio=float(self._options.get("opt_pair_ratio", 0.5)),
            train_rows=int(self._options.get("opt_train_samples", 2048)),
            val_rows=int(self._options.get("opt_validation_samples", 64)),
            batch_size=int(self._options.get("opt_batch_size", 64)),
            rotation_epochs=int(self._options.get("opt_rotation_epochs", 10)),
            finetune_epochs=finetune_epochs,
            rotation_lr=float(self._options.get("opt_rotation_lr", 0.05)),
            weight_lr=float(self._options.get("opt_weight_lr", 1e-5)),
            quantizer_lr=float(self._options.get("opt_quantizer_lr", 1e-6)),
            seed=self._module_seed(ctx.module_name),
            optimizer_name=str(self._options.get("opt_optimizer", "adamw")),
            fused_rotation=bool(self._options.get("opt_fused_rotation", True)),
            gradient_checkpointing=bool(self._options.get("opt_gradient_checkpointing", False)),
            stage_impl=str(self._options.get("opt_stage_impl", "fast")),
            pair_impl=str(self._options.get("opt_pair_impl", "fast")),
            quantizer_impl=str(self._options.get("opt_quantizer_impl", "reference")),
            scale_clamp_min=float(self._options.get("opt_channel_scale_clamp_min", 1e-2)),
            scale_clamp_max=float(self._options.get("opt_channel_scale_clamp_max", 1e2)),
        )

        return TransformState(
            method="paroquant",
            bake_weights=self.cfg.bake_weights,
            payload={
                "pseudo_weight": result.pseudo_weight.detach().cpu(),
                "pack_weight": result.pack_weight.detach().cpu(),
                "q_scales": result.q_scales.detach().cpu(),
                "q_zeros": result.q_zeros.detach().cpu(),
                "pairs": result.pairs.detach().cpu(),
                "theta": result.theta.detach().cpu(),
                "channel_scales": result.channel_scales.detach().cpu(),
                "train_loss": result.train_loss,
                "val_loss": result.val_loss,
            },
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

    def activation_pre_hook(self, state: TransformState) -> Callable[..., None]:
        del state

        def _noop(*_args, **_kwargs):
            return None

        return _noop
