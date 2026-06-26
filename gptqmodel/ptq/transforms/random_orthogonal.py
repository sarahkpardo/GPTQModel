# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Fixed random orthogonal transform per block (QuIP-style incoherence baseline)."""

from __future__ import annotations

from typing import Callable

import torch

from ..config import TransformPrepareConfig
from ..context import ModuleCalibContext, TransformState
from ..protocols import TransformMode
from .block_dense import (
    apply_block_transform_to_activation,
    apply_block_transform_to_hessian,
    apply_block_transform_to_weight,
    build_inference_transform_data,
    generate_random_orthogonal_blocks,
    pad_columns,
    resolve_weight_layout,
)
from .paroquant import module_seed_from_options


def _resolve_block_size(options: dict, input_columns: int) -> int:
    block_size = int(options.get("block_size", options.get("group_size", 128)))
    if block_size <= 0:
        raise ValueError(f"block_size/group_size must be positive, got {block_size}.")
    if input_columns % block_size != 0:
        padded, pad, _ = pad_columns(input_columns, block_size)
        if pad > 0:
            options.setdefault("allow_pad", True)
            options["pad"] = pad
            options["padded_columns"] = padded
    return block_size


def _resolve_inference_dtype(options: dict) -> torch.dtype:
    raw = options.get("inference_precision", "float16")
    if isinstance(raw, torch.dtype):
        return raw
    raw = str(raw).strip().lower()
    if raw in {"float16", "fp16", "torch.float16"}:
        return torch.float16
    if raw in {"bfloat16", "bf16", "torch.bfloat16"}:
        return torch.bfloat16
    if raw in {"float32", "fp32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported inference_precision `{raw}`.")


class RandomOrthogonalTransform:
    """Closed-form Haar-random orthogonal ``Q`` per input block via QR."""

    def __init__(self, cfg: TransformPrepareConfig) -> None:
        self.cfg = cfg

    def fit(
        self,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        ctx: ModuleCalibContext,
        mode: TransformMode,
        device: torch.device,
    ) -> TransformState:
        del bias, mode
        options = dict(self.cfg.options)
        input_columns = int(ctx.columns)
        if input_columns <= 0:
            raise ValueError(
                f"RandomOrthogonalTransform requires positive ctx.columns for `{ctx.module_name}`, "
                f"observed {input_columns}."
            )
        weight_layout = resolve_weight_layout(weight, input_columns)
        block_size = _resolve_block_size(options, input_columns)
        padded, pad, num_blocks = pad_columns(input_columns, block_size)
        seed = module_seed_from_options(module_name=ctx.module_name, options=options)
        inference_dtype = _resolve_inference_dtype(options)

        t_w_blocks, t_x_matrices = generate_random_orthogonal_blocks(
            block_size,
            num_blocks,
            seed=seed,
            device=device,
            inference_dtype=inference_dtype,
        )

        payload = {
            "group_size": block_size,
            "block_size": block_size,
            "seed": seed,
            "pad": pad,
            "padded_columns": padded,
            "input_columns": input_columns,
            "weight_layout": weight_layout,
            "T_W_blocks": t_w_blocks,
            "T_X_matrices": t_x_matrices,
            "inference_precision": "float16",
        }
        return TransformState(
            method="random_orthogonal",
            bake_weights=self.cfg.bake_weights,
            payload=payload,
        )

    def apply_to_weights(
        self,
        weight: torch.Tensor,
        state: TransformState,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        del device
        payload = state.payload
        return apply_block_transform_to_weight(
            weight,
            payload["T_W_blocks"],
            block_size=int(payload["group_size"]),
            pad=int(payload.get("pad", 0)),
            input_columns=int(payload.get("input_columns", weight.shape[1])),
            weight_layout=str(payload.get("weight_layout", "linear")),
        )

    def transform_hessian(self, H: torch.Tensor, state: TransformState) -> torch.Tensor:
        payload = state.payload
        return apply_block_transform_to_hessian(
            H,
            payload["T_W_blocks"],
            block_size=int(payload["group_size"]),
            pad=int(payload.get("pad", 0)),
        )

    def get_inference_data(self, state: TransformState):
        payload = state.payload
        inference_dtype = _resolve_inference_dtype(
            {"inference_precision": payload.get("inference_precision", "float16")}
        )
        t_x = payload.get("T_X_matrices")
        if t_x is None:
            return build_inference_transform_data(
                torch.empty(int(payload["group_size"]), int(payload["group_size"]), 0),
                block_size=int(payload["group_size"]),
                precision=inference_dtype,
            )
        return build_inference_transform_data(
            t_x,
            block_size=int(payload["group_size"]),
            precision=inference_dtype,
        )

    def activation_pre_hook(self, state: TransformState) -> Callable[..., None]:
        payload = state.payload
        t_x_matrices = payload["T_X_matrices"]
        block_size = int(payload["group_size"])
        pad = int(payload.get("pad", 0))
        inference_dtype = _resolve_inference_dtype(
            {"inference_precision": payload.get("inference_precision", "float16")}
        )

        def _hook(_module, args, kwargs):
            if not args and "input" not in kwargs:
                return None
            x = args[0] if args else kwargs.get("input")
            if not isinstance(x, torch.Tensor):
                return None
            original_columns = x.shape[-1]
            transformed = apply_block_transform_to_activation(
                x,
                t_x_matrices.to(device=x.device, dtype=inference_dtype),
                block_size=block_size,
                pad=pad,
                original_columns=original_columns,
            )
            if args:
                new_args = (transformed, *args[1:])
                return new_args, kwargs
            kwargs["input"] = transformed
            return args, kwargs

        return _hook
