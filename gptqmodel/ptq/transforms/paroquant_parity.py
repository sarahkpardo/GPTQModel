# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for ParoQuant transform parity testing."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from ...quantization.paroquant.optimization import optimize_paroquant_linear
from ..config import TransformPrepareConfig
from ..context import ModuleCalibContext, TransformState
from .paroquant import (
    PAROQUANT_PAYLOAD_KEYS,
    ParoQuantTransform,
    build_paroquant_optimize_kwargs,
    paroquant_result_to_payload,
)


def make_paroquant_prepare_config(
    *,
    options: Optional[Dict[str, Any]] = None,
    mode: str = "standalone",
) -> TransformPrepareConfig:
    merged = {
        "krot": 2,
        "opt_rotation_epochs": 1,
        "opt_finetune_epochs": 0,
        "opt_train_samples": 128,
        "opt_validation_samples": 32,
        "opt_batch_size": 32,
        "opt_seed": 0,
        "bits": 4,
        "group_size": 4,
        "sym": True,
    }
    if options:
        merged.update(options)
    return TransformPrepareConfig(method="paroquant", mode=mode, bake_weights=True, options=merged)


def make_calibration_context(
    *,
    module_name: str,
    columns: int,
    rows: int,
    inputs: torch.Tensor,
) -> ModuleCalibContext:
    return ModuleCalibContext(
        module_name=module_name,
        columns=columns,
        rows=rows,
        nsamples=int(inputs.shape[0]) if inputs.numel() else 0,
        row_buffer=inputs.detach().clone(),
    )


def run_paroquant_reference(
    *,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    ctx: ModuleCalibContext,
    cfg: TransformPrepareConfig,
    qcfg=None,
) -> Dict[str, Any]:
    """Direct optimize_paroquant_linear using transform-equivalent kwargs."""
    kwargs = build_paroquant_optimize_kwargs(
        weight=weight,
        bias=bias,
        ctx=ctx,
        cfg=cfg,
        qcfg=qcfg,
    )
    result = optimize_paroquant_linear(**kwargs)
    return paroquant_result_to_payload(result)


def run_paroquant_transform(
    *,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    ctx: ModuleCalibContext,
    cfg: TransformPrepareConfig,
    qcfg=None,
) -> TransformState:
    backend = ParoQuantTransform(cfg)
    return backend.fit(
        weight=weight,
        bias=bias,
        ctx=ctx,
        mode=cfg.mode if cfg.mode in {"standalone", "e2e"} else "standalone",
        device=weight.device,
        qcfg=qcfg,
    )


def payload_from_state(state: TransformState) -> Dict[str, Any]:
    return dict(state.payload)


def compare_paroquant_payloads(
    reference: Dict[str, Any],
    subject: Dict[str, Any],
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    keys: Iterable[str] = PAROQUANT_PAYLOAD_KEYS,
) -> Tuple[bool, List[str]]:
    """Return (all_match, human-readable diff messages)."""
    diffs: List[str] = []
    for key in keys:
        ref_value = reference.get(key)
        sub_value = subject.get(key)
        if isinstance(ref_value, torch.Tensor):
            if not isinstance(sub_value, torch.Tensor):
                diffs.append(f"{key}: subject missing tensor")
                continue
            if ref_value.shape != sub_value.shape:
                diffs.append(f"{key}: shape {tuple(sub_value.shape)} != {tuple(ref_value.shape)}")
                continue
            if not torch.allclose(ref_value, sub_value, atol=atol, rtol=rtol):
                max_diff = (ref_value.to(torch.float32) - sub_value.to(torch.float32)).abs().max().item()
                diffs.append(f"{key}: max_abs_diff={max_diff:.6e}")
        elif isinstance(ref_value, (int, float)):
            if not isinstance(sub_value, (int, float)):
                diffs.append(f"{key}: subject type {type(sub_value)} != numeric")
                continue
            if abs(float(ref_value) - float(sub_value)) > max(atol, rtol * max(abs(float(ref_value)), 1.0)):
                diffs.append(f"{key}: {sub_value} != {ref_value}")
        elif ref_value != sub_value:
            diffs.append(f"{key}: {sub_value!r} != {ref_value!r}")
    return len(diffs) == 0, diffs


def tensor_fingerprint(t: Optional[torch.Tensor]) -> str:
    if t is None or not isinstance(t, torch.Tensor):
        return "none"
    flat = t.detach().to(torch.float32).cpu().contiguous().view(-1)
    return f"shape={tuple(t.shape)} norm={flat.norm().item():.6f}"


def make_synthetic_linear(
    *,
    in_features: int,
    out_features: int,
    seed: int,
) -> tuple[nn.Linear, torch.Tensor]:
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    inputs = torch.randn(160, in_features, dtype=torch.float32)
    return linear, inputs
