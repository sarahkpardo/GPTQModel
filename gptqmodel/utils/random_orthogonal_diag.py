# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Diagnostics for random_orthogonal PTQ: quant logs, hook audit, output MSE."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn

from ..models.writer import QUANT_LOG_DAMP, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES, PROCESS_LOG_NAME
from ..ptq.config import TransformPrepareConfig, WeightQuantizeTargetConfig
from ..ptq.context import ModuleCalibContext
from ..ptq.pipeline import ModuleQuantizationPipeline
from ..ptq.transforms.block_dense import (
    apply_block_transform_to_activation,
    apply_block_transform_to_weight,
    pad_columns,
)
from ..ptq.transforms.random_orthogonal import RandomOrthogonalTransform
from ..quantization.config import QuantizeConfig

WeightPrepareMode = Literal["identity", "random_orthogonal"]


@dataclass
class QuantLogSummary:
    mean_loss: float | None
    max_damp: float | None
    modules_with_elevated_damp: int
    module_count: int
    per_module: list[dict[str, str | float | None]]


@dataclass
class LayerDiagResult:
    weight_prepare: str
    gptq_loss: float | str | None
    damp: float | None
    pre_quant_matmul_rel_err: float
    post_quant_output_mse: float
    hessian_trace: float | None
    hessian_trace_transformed: float | None


@dataclass
class HookAuditResult:
    modules_with_t_x_buffers: int
    modules_with_active_hooks: int
    modules_with_pad: list[tuple[str, int]]


def summarize_quant_log(
    quantize_result: dict[str, list[dict[str, str]]],
    *,
    base_damp: float = 0.01,
    damp_epsilon: float = 1e-6,
) -> QuantLogSummary:
    """Summarize per-module loss/damp from ``GPTQModel.quantize()`` log output."""
    per_module: list[dict[str, str | float | None]] = []
    losses: list[float] = []
    damp_values: list[float] = []
    elevated = 0

    for layer_key, entries in quantize_result.items():
        for entry in entries:
            if entry.get(PROCESS_LOG_NAME) == "statistics":
                continue
            module = entry.get("module") or layer_key
            raw_loss = entry.get(QUANT_LOG_LOSS, entry.get("loss", ""))
            raw_damp = entry.get(QUANT_LOG_DAMP, entry.get("damp", ""))
            loss_val: float | None = None
            damp_val: float | None = None
            if raw_loss not in {"", "unknown", None}:
                try:
                    loss_val = float(raw_loss)
                    losses.append(loss_val)
                except (TypeError, ValueError):
                    pass
            if raw_damp not in {"", None}:
                try:
                    damp_val = float(raw_damp)
                    damp_values.append(damp_val)
                    if damp_val > base_damp + damp_epsilon:
                        elevated += 1
                except (TypeError, ValueError):
                    pass
            per_module.append(
                {
                    "layer": str(layer_key),
                    "module": str(module),
                    "loss": loss_val if loss_val is not None else str(raw_loss),
                    "damp": damp_val,
                    "nsamples": entry.get(QUANT_LOG_NSAMPLES),
                }
            )

    return QuantLogSummary(
        mean_loss=(sum(losses) / len(losses)) if losses else None,
        max_damp=max(damp_values) if damp_values else None,
        modules_with_elevated_damp=elevated,
        module_count=len(per_module),
        per_module=per_module,
    )


def audit_ptq_hooks(model: nn.Module) -> HookAuditResult:
    """Count T_X buffers, active pre-hooks, and modules with nonzero pad."""
    buffers = 0
    hooked = 0
    padded: list[tuple[str, int]] = []
    for name, module in model.named_modules():
        t_x = getattr(module, "ptq_t_x_matrices", None)
        if isinstance(t_x, torch.Tensor) and t_x.numel() > 0:
            buffers += 1
            if module._forward_pre_hooks:
                hooked += 1
            pad = int(getattr(module, "ptq_t_x_pad", torch.tensor(0)).item())
            if pad > 0:
                padded.append((name, pad))
    return HookAuditResult(
        modules_with_t_x_buffers=buffers,
        modules_with_active_hooks=hooked,
        modules_with_pad=padded,
    )


def inverse_block_transform_to_weight(
    weight: torch.Tensor,
    t_w_blocks: Sequence[torch.Tensor],
    *,
    block_size: int,
    pad: int,
    input_columns: int,
    weight_layout: str = "linear",
) -> torch.Tensor:
    """Map baked weights back to original input axis: ``W = W_baked @ Q`` per block."""
    device = weight.device
    dtype = weight.dtype
    blocks = [block.to(device=device, dtype=dtype) for block in t_w_blocks]
    padded, _, num_blocks = pad_columns(input_columns, block_size)

    if weight_layout == "linear":
        out_features = weight.shape[0]
        if pad > 0:
            weight_work = torch.cat(
                [weight, torch.zeros(out_features, pad, device=device, dtype=dtype)],
                dim=1,
            )
        else:
            weight_work = weight.clone()
        result = weight_work.clone()
        for block_idx in range(num_blocks):
            sl = slice(block_idx * block_size, (block_idx + 1) * block_size)
            q = blocks[block_idx]
            result[:, sl] = result[:, sl] @ q
        return result[:, :input_columns]

    if weight_layout == "conv1d":
        in_features = weight.shape[0]
        if pad > 0:
            weight_work = torch.cat(
                [weight, torch.zeros(pad, weight.shape[1], device=device, dtype=dtype)],
                dim=0,
            )
        else:
            weight_work = weight.clone()
        result = weight_work.clone()
        for block_idx in range(num_blocks):
            sl = slice(block_idx * block_size, (block_idx + 1) * block_size)
            q = blocks[block_idx]
            result[sl, :] = q.T @ result[sl, :]
        return result[:input_columns, :]

    raise ValueError(f"Unsupported weight_layout `{weight_layout}`.")


def _linear_forward(x: torch.Tensor, weight: torch.Tensor, layout: str) -> torch.Tensor:
    x_f = x.float()
    w_f = weight.float()
    if layout == "conv1d":
        return x_f @ w_f
    return x_f @ w_f.T


def _stack_t_x(t_w_blocks: Sequence[torch.Tensor], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.stack([block.T.to(device=device, dtype=dtype) for block in t_w_blocks], dim=2)


def pre_quant_matmul_rel_error(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    baked: torch.Tensor,
    t_w_blocks: Sequence[torch.Tensor],
    block_size: int,
    pad: int,
    weight_layout: str,
) -> float:
    baseline = _linear_forward(x, weight, weight_layout)
    t_x = _stack_t_x(t_w_blocks, dtype=x.dtype, device=x.device)
    x_tx = apply_block_transform_to_activation(
        x,
        t_x,
        block_size=block_size,
        pad=pad,
        original_columns=x.shape[-1],
    )
    transformed = _linear_forward(x_tx, baked, weight_layout)
    denom = baseline.norm().item()
    if denom <= 0:
        return 0.0
    return (transformed - baseline).norm().item() / denom


def post_quant_output_mse(
    *,
    x: torch.Tensor,
    weight_fp: torch.Tensor,
    weight_q: torch.Tensor,
    weight_layout: str,
    transform_state=None,
) -> float:
    target = _linear_forward(x, weight_fp, weight_layout)
    if transform_state is None or transform_state.method == "identity":
        approx = _linear_forward(x, weight_q, weight_layout)
    else:
        payload = transform_state.payload
        block_size = int(payload["group_size"])
        pad = int(payload.get("pad", 0))
        input_columns = int(payload.get("input_columns", weight_fp.shape[1 if weight_layout == "linear" else 0]))
        layout = str(payload.get("weight_layout", weight_layout))
        t_w_blocks = payload["T_W_blocks"]
        t_x = payload.get("T_X_matrices")
        if t_x is None:
            t_x = _stack_t_x(t_w_blocks, dtype=x.dtype, device=x.device)
        else:
            t_x = t_x.to(device=x.device, dtype=x.dtype)
        x_tx = apply_block_transform_to_activation(
            x,
            t_x,
            block_size=block_size,
            pad=pad,
            original_columns=x.shape[-1],
        )
        approx = _linear_forward(x_tx, weight_q, layout)
    return (target - approx).pow(2).mean().item()


def run_layer_pipeline(
    *,
    module: nn.Linear,
    ctx: ModuleCalibContext,
    qcfg: QuantizeConfig,
    weight_prepare: WeightPrepareMode,
    opt_seed: int = 42,
    inference_precision: str = "float16",
) -> tuple[LayerDiagResult, torch.Tensor, object | None]:
    """Run identity or random_orthogonal pipeline on a module with frozen context."""
    weight_fp = module.weight.data.clone()
    bias = getattr(module, "bias", None)
    bias_data = bias.data.clone() if bias is not None else None
    work = nn.Linear(module.in_features, module.out_features, bias=bias is not None)
    work.weight.data = weight_fp.clone()
    if bias_data is not None:
        work.bias.data = bias_data.clone()

    ctx_copy = copy.deepcopy(ctx)
    if ctx_copy.H is not None:
        ctx_copy.H = ctx_copy.H.clone()
    if ctx_copy.qr_R is not None:
        ctx_copy.qr_R = ctx_copy.qr_R.clone()

    hessian_trace = float(ctx_copy.H.trace().item()) if ctx_copy.H is not None else None

    prepare_cfg = TransformPrepareConfig(
        method=weight_prepare,
        options=(
            {
                "group_size": qcfg.group_size,
                "opt_seed": opt_seed,
                "inference_precision": inference_precision,
            }
            if weight_prepare == "random_orthogonal"
            else {}
        ),
    )
    pipeline = ModuleQuantizationPipeline(
        qcfg=qcfg,
        prepare_configs=[prepare_cfg],
        weight_quantize=WeightQuantizeTargetConfig(method="gptq"),
    )
    result = pipeline.run_module(
        module=work,
        collector=None,  # type: ignore[arg-type]  # ctx already finalized
        ctx=ctx_copy,
        device=torch.device("cpu"),
    )
    transform_state = ctx_copy.transform
    weight_q = result.pack_weight
    if weight_q is None:
        weight_q = work.weight.data.clone()
    else:
        weight_q = weight_q.detach().clone()

    x = ctx_copy.row_buffer
    if x is None or x.numel() == 0:
        x = torch.randn(min(256, max(32, ctx_copy.nsamples)), module.in_features)

    hessian_trace_t = None
    pre_err = 0.0
    if transform_state is not None and transform_state.method == "random_orthogonal":
        payload = transform_state.payload
        backend = RandomOrthogonalTransform(
            TransformPrepareConfig(method="random_orthogonal", options={"group_size": qcfg.group_size})
        )
        baked = backend.apply_to_weights(weight_fp, transform_state, device=torch.device("cpu"))
        pre_err = pre_quant_matmul_rel_error(
            x=x,
            weight=weight_fp,
            baked=baked,
            t_w_blocks=payload["T_W_blocks"],
            block_size=int(payload["group_size"]),
            pad=int(payload.get("pad", 0)),
            weight_layout=str(payload.get("weight_layout", "linear")),
        )
        if ctx_copy.H is not None:
            h_prime = backend.transform_hessian(ctx.H.clone(), transform_state)
            hessian_trace_t = float(h_prime.trace().item())

    mse = post_quant_output_mse(
        x=x,
        weight_fp=weight_fp,
        weight_q=weight_q,
        weight_layout="linear",
        transform_state=transform_state,
    )

    damp = result.extra.get("damp")
    damp_f = float(damp) if damp is not None else None
    return (
        LayerDiagResult(
            weight_prepare=weight_prepare,
            gptq_loss=result.extra.get("loss"),
            damp=damp_f,
            pre_quant_matmul_rel_err=pre_err,
            post_quant_output_mse=mse,
            hessian_trace=hessian_trace,
            hessian_trace_transformed=hessian_trace_t,
        ),
        weight_q,
        transform_state,
    )


def measure_activation_drift(
    fp16_module: nn.Module,
    quant_module: nn.Module,
    input_ids: torch.Tensor,
    *,
    probe_module_names: Sequence[str],
) -> dict[str, float]:
    """Compare intermediate activations (first tensor output) vs FP16 on one forward."""
    fp16_tensors: dict[str, torch.Tensor] = {}
    quant_tensors: dict[str, torch.Tensor] = {}

    def _make_hook(store: dict[str, torch.Tensor], name: str):
        def _hook(_mod, _inp, out):
            if isinstance(out, torch.Tensor):
                store[name] = out.detach()
            elif isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                store[name] = out[0].detach()

        return _hook

    fp16_handles = []
    quant_handles = []
    fp16_named = dict(fp16_module.named_modules())
    quant_named = dict(quant_module.named_modules())
    for name in probe_module_names:
        if name in fp16_named:
            fp16_handles.append(fp16_named[name].register_forward_hook(_make_hook(fp16_tensors, name)))
        if name in quant_named:
            quant_handles.append(quant_named[name].register_forward_hook(_make_hook(quant_tensors, name)))

    device = input_ids.device
    with torch.inference_mode():
        fp16_module(input_ids)
        quant_module(input_ids)

    for handle in fp16_handles + quant_handles:
        handle.remove()

    drift: dict[str, float] = {}
    for name in probe_module_names:
        fp = fp16_tensors.get(name)
        qt = quant_tensors.get(name)
        if fp is None or qt is None:
            continue
        fp_f = fp.float()
        qt_f = qt.float()
        denom = fp_f.norm().item()
        if denom <= 0:
            drift[name] = 0.0
        else:
            drift[name] = (fp_f - qt_f).norm().item() / denom
    return drift
