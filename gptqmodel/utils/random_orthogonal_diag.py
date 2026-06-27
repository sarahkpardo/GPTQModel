# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Diagnostics for random_orthogonal PTQ: quant logs, hook audit, output MSE."""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class KernelTypeAudit:
    torch_linear: int
    marlin_linear: int
    other_quant_linear: int
    plain_linear: int
    sample_torch_linear: list[str]
    sample_marlin_linear: list[str]
    sample_plain_linear: list[str]
    sample_other_quant_linear: list[str]


def resolve_model_param_dtype(model: nn.Module) -> str:
    """Return the dtype string of the first floating-point parameter, if any."""
    for param in model.parameters():
        if param.is_floating_point():
            return str(param.dtype)
    return "unknown"


def sample_torchlinear_qweight_device(model: nn.Module) -> str | None:
    """Return the device string of the first TorchLinear qweight buffer, if any."""
    from ..nn_modules.qlinear.torch import TorchLinear

    for module in model.modules():
        if isinstance(module, TorchLinear):
            qweight = getattr(module, "qweight", None)
            if isinstance(qweight, torch.Tensor):
                return str(qweight.device)
    return None


def _resolve_embedding_modules(
    model: nn.Module,
) -> tuple[nn.Module | None, nn.Module | None, str | None, str | None]:
    """Return input/output embedding modules and their module paths when discoverable."""
    input_embed = None
    output_embed = None
    input_name = None
    output_name = None

    if hasattr(model, "get_input_embeddings"):
        try:
            input_embed = model.get_input_embeddings()
        except Exception:
            input_embed = None
    if hasattr(model, "get_output_embeddings"):
        try:
            output_embed = model.get_output_embeddings()
        except Exception:
            output_embed = None

    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if input_embed is None and name.endswith("embed_tokens"):
            input_embed = module
            input_name = name
        if output_embed is None and name.endswith("lm_head"):
            output_embed = module
            output_name = name

    return input_embed, output_embed, input_name, output_name


def _tensor_storage_key(tensor: torch.Tensor) -> tuple[int, int]:
    return (tensor.untyped_storage().data_ptr(), tensor.storage_offset())


@torch.no_grad()
def audit_tied_weight_aliasing(model: nn.Module) -> dict[str, object]:
    """Audit embed/lm_head aliasing for reload debugging."""
    config = getattr(model, "config", None)
    tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
    input_embed, output_embed, input_name, output_name = _resolve_embedding_modules(model)

    input_weight = getattr(input_embed, "weight", None) if input_embed is not None else None
    output_weight = getattr(output_embed, "weight", None) if output_embed is not None else None

    same_param_object = (
        input_weight is not None
        and output_weight is not None
        and input_weight is output_weight
    )
    same_storage = False
    if input_weight is not None and output_weight is not None:
        same_storage = _tensor_storage_key(input_weight) == _tensor_storage_key(output_weight)

    max_abs_diff: float | None = None
    if (
        input_weight is not None
        and output_weight is not None
        and input_weight.shape == output_weight.shape
        and not same_storage
    ):
        max_abs_diff = float((input_weight - output_weight).abs().max().item())

    tied_keys = getattr(model, "_tied_weights_keys", None)
    if tied_keys is None:
        tied_keys_repr: object = None
    elif isinstance(tied_keys, dict):
        tied_keys_repr = dict(tied_keys)
    else:
        tied_keys_repr = list(tied_keys)

    hf_device_map = getattr(model, "hf_device_map", None)

    return {
        "tie_word_embeddings": tie_word_embeddings,
        "input_embed_module": input_name,
        "output_embed_module": output_name,
        "input_embed_found": input_embed is not None,
        "output_embed_found": output_embed is not None,
        "same_param_object": same_param_object if input_weight is not None and output_weight is not None else None,
        "same_storage": same_storage if input_weight is not None and output_weight is not None else None,
        "aliased_as_expected": (
            same_storage
            if tie_word_embeddings and input_weight is not None and output_weight is not None
            else None
        ),
        "max_abs_diff_untied": max_abs_diff,
        "input_weight_shape": list(input_weight.shape) if input_weight is not None else None,
        "output_weight_shape": list(output_weight.shape) if output_weight is not None else None,
        "input_weight_device": str(input_weight.device) if input_weight is not None else None,
        "output_weight_device": str(output_weight.device) if output_weight is not None else None,
        "tied_weights_keys": tied_keys_repr,
        "hf_device_map_present": hf_device_map is not None,
        "hf_device_map_sample": dict(list(hf_device_map.items())[:8]) if isinstance(hf_device_map, dict) else None,
    }


def list_checkpoint_tensor_keys(checkpoint_path: str | Path) -> set[str]:
    """Return tensor names stored in a GPTQModel checkpoint directory or safetensors file."""
    from safetensors import safe_open

    path = Path(checkpoint_path)
    keys: set[str] = set()
    files: list[Path]
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
    elif path.suffix == ".safetensors":
        files = [path]
    else:
        return keys

    for file_path in files:
        with safe_open(str(file_path), framework="pt", device="cpu") as reader:
            keys.update(reader.keys())
    return keys


def _model_checkpoint_key_names(model: nn.Module) -> set[str]:
    names = {name for name, _ in model.named_parameters()}
    for name, buffer in model.named_buffers():
        module_path, leaf = name.rsplit(".", 1) if "." in name else ("", name)
        module = model.get_submodule(module_path) if module_path else model
        non_persistent = getattr(module, "_non_persistent_buffers_set", set())
        if leaf in non_persistent:
            continue
        names.add(name)
    return names


def audit_checkpoint_load_keys(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    max_list: int = 20,
) -> dict[str, object]:
    """Compare checkpoint tensor names against the loaded model state."""
    checkpoint_keys = list_checkpoint_tensor_keys(checkpoint_path)
    model_keys = _model_checkpoint_key_names(model)
    missing_from_checkpoint = sorted(model_keys - checkpoint_keys)
    unexpected_in_checkpoint = sorted(checkpoint_keys - model_keys)

    def _has_suffix(keys: Sequence[str], suffix: str) -> bool:
        return any(key == suffix or key.endswith(f".{suffix}") for key in keys)

    embed_keys = [key for key in checkpoint_keys if key.endswith("embed_tokens.weight")]
    lm_head_keys = [key for key in checkpoint_keys if key.endswith("lm_head.weight")]

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_key_count": len(checkpoint_keys),
        "model_key_count": len(model_keys),
        "missing_from_checkpoint_count": len(missing_from_checkpoint),
        "missing_from_checkpoint_sample": missing_from_checkpoint[:max_list],
        "unexpected_in_checkpoint_count": len(unexpected_in_checkpoint),
        "unexpected_in_checkpoint_sample": unexpected_in_checkpoint[:max_list],
        "embed_tokens_weight_in_checkpoint": bool(embed_keys),
        "embed_tokens_weight_keys": embed_keys[:max_list],
        "lm_head_weight_in_checkpoint": bool(lm_head_keys),
        "lm_head_weight_keys": lm_head_keys[:max_list],
        "lm_head_weight_missing_from_checkpoint": _has_suffix(missing_from_checkpoint, "lm_head.weight"),
        "embed_tokens_weight_missing_from_checkpoint": _has_suffix(
            missing_from_checkpoint,
            "embed_tokens.weight",
        ),
    }


@torch.no_grad()
def compare_logits_tensors(
    pre_logits: torch.Tensor,
    post_logits: torch.Tensor,
    *,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> dict[str, object]:
    """Compare two captured logits tensors (e.g. pre-reload vs post-reload)."""
    ref = pre_logits.float()
    cand = post_logits.float()
    if ref.shape != cand.shape:
        return {
            "shape_match": False,
            "pre_logits_shape": list(pre_logits.shape),
            "post_logits_shape": list(post_logits.shape),
            "mean_rel_error": None,
            "max_abs_error": None,
            "mean_abs_error": None,
            "allclose": False,
        }

    diff = (cand - ref).abs()
    denom = ref.abs().mean().clamp(min=1e-6)
    return {
        "shape_match": True,
        "pre_logits_shape": list(pre_logits.shape),
        "post_logits_shape": list(post_logits.shape),
        "mean_rel_error": float(diff.mean().item() / denom.item()),
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "allclose": bool(torch.allclose(cand, ref, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
    }


def audit_quant_kernel_types(model: nn.Module, *, sample_limit: int = 5) -> KernelTypeAudit:
    """Count quant kernel module types in a model graph."""
    from ..nn_modules.qlinear import BaseQuantLinear
    from ..nn_modules.qlinear.marlin import MarlinLinear
    from ..nn_modules.qlinear.torch import TorchLinear

    torch_names: list[str] = []
    marlin_names: list[str] = []
    plain_names: list[str] = []
    other_names: list[str] = []

    torch_count = 0
    marlin_count = 0
    plain_count = 0
    other_count = 0

    for name, module in model.named_modules():
        if isinstance(module, TorchLinear):
            torch_count += 1
            if len(torch_names) < sample_limit:
                torch_names.append(name)
        elif isinstance(module, MarlinLinear):
            marlin_count += 1
            if len(marlin_names) < sample_limit:
                marlin_names.append(name)
        elif isinstance(module, BaseQuantLinear):
            other_count += 1
            if len(other_names) < sample_limit:
                other_names.append(f"{name}:{type(module).__name__}")
        elif isinstance(module, nn.Linear):
            plain_count += 1
            if len(plain_names) < sample_limit:
                plain_names.append(name)

    return KernelTypeAudit(
        torch_linear=torch_count,
        marlin_linear=marlin_count,
        other_quant_linear=other_count,
        plain_linear=plain_count,
        sample_torch_linear=torch_names,
        sample_marlin_linear=marlin_names,
        sample_plain_linear=plain_names,
        sample_other_quant_linear=other_names,
    )


@torch.no_grad()
def compare_inmem_reload_dequant(
    inmem_model: nn.Module,
    reloaded_model: nn.Module,
    *,
    max_modules: int = 3,
    module_names: Sequence[str] | None = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> dict[str, object]:
    """Compare dequantized weights in-memory vs after checkpoint reload."""
    from ..nn_modules.qlinear.torch import TorchLinear

    comparisons: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []

    inmem_modules = [
        (name, module)
        for name, module in inmem_model.named_modules()
        if isinstance(module, TorchLinear)
    ]
    if module_names is not None:
        name_set = set(module_names)
        inmem_modules = [(name, module) for name, module in inmem_modules if name in name_set]
    elif max_modules >= 0:
        inmem_modules = inmem_modules[:max_modules]

    for name, inmem_q in inmem_modules:
        try:
            reloaded_q = reloaded_model.get_submodule(name)
        except (AttributeError, ModuleNotFoundError) as exc:
            mismatches.append({"module": name, "error": str(exc)})
            continue
        if not isinstance(reloaded_q, TorchLinear):
            mismatches.append(
                {
                    "module": name,
                    "error": f"reloaded type {type(reloaded_q).__name__}, expected TorchLinear",
                }
            )
            continue

        buffers = inmem_q.list_buffers()
        eval_device = buffers[0].device if buffers else next(inmem_q.parameters()).device
        reloaded_q = reloaded_q.to(eval_device)
        w_inmem = inmem_q.dequantize_weight().float().cpu()
        w_reload = reloaded_q.dequantize_weight().float().cpu()
        diff = (w_inmem - w_reload).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        denom = float(w_inmem.abs().mean().clamp(min=1e-6).item())
        match = bool(torch.allclose(w_inmem, w_reload, rtol=rtol, atol=atol))
        comparisons.append(
            {
                "module": name,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "mean_rel_diff": mean_abs / denom,
                "match": match,
            }
        )

    return {
        "modules_compared": len(comparisons),
        "all_match": all(bool(row["match"]) for row in comparisons) if comparisons else False,
        "comparisons": comparisons,
        "mismatches": mismatches,
    }


def clear_torchlinear_inference_state(model: nn.Module) -> int:
    """Reset TorchLinear eval caches before inference or dequant comparisons."""
    from ..nn_modules.qlinear.torch import TorchLinear

    count = 0
    for module in model.modules():
        if isinstance(module, TorchLinear):
            module.clear_weight_cache()
            if hasattr(module, "_stream_reset_cache"):
                module._stream_reset_cache()
            count += 1
    return count


def _eager_torchlinear_dequant(module) -> torch.Tensor:
    """Dequantize without eval caches or compiled wrappers."""
    from ..nn_modules.qlinear.torch import TorchLinear

    if not isinstance(module, TorchLinear):
        raise TypeError(f"Expected TorchLinear, got {type(module)!r}")
    module.clear_weight_cache()
    if hasattr(module, "_stream_reset_cache"):
        module._stream_reset_cache()
    num_itr = module.g_idx.shape[0] // module.in_features
    return TorchLinear.__bases__[0].dequantize_weight(module, num_itr=num_itr)


@torch.no_grad()
def compare_inmem_reload_dequant_eager(
    inmem_model: nn.Module,
    reloaded_model: nn.Module,
    **kwargs,
) -> dict[str, object]:
    """Like ``compare_inmem_reload_dequant`` but uses eager parent dequant on both sides."""
    from ..nn_modules.qlinear.torch import TorchLinear

    result = compare_inmem_reload_dequant(inmem_model, reloaded_model, **kwargs)
    comparisons: list[dict[str, object]] = []
    inmem_modules = [
        (name, module)
        for name, module in inmem_model.named_modules()
        if isinstance(module, TorchLinear)
    ]
    max_modules = kwargs.get("max_modules", 3)
    rtol = kwargs.get("rtol", 1e-2)
    atol = kwargs.get("atol", 1e-2)

    for name, inmem_q in inmem_modules[:max_modules]:
        try:
            reloaded_q = reloaded_model.get_submodule(name)
        except (AttributeError, ModuleNotFoundError):
            continue
        if not isinstance(reloaded_q, TorchLinear):
            continue
        buffers = inmem_q.list_buffers()
        eval_device = buffers[0].device if buffers else next(inmem_q.parameters()).device
        reloaded_q = reloaded_q.to(eval_device)
        w_inmem = _eager_torchlinear_dequant(inmem_q).float().cpu()
        w_reload = _eager_torchlinear_dequant(reloaded_q).float().cpu()
        diff = (w_inmem - w_reload).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        denom = float(w_inmem.abs().mean().clamp(min=1e-6).item())
        comparisons.append(
            {
                "module": name,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "mean_rel_diff": mean_abs / denom,
                "match": bool(torch.allclose(w_inmem, w_reload, rtol=rtol, atol=atol)),
            }
        )

    if comparisons:
        result["eager_comparisons"] = comparisons
        result["eager_all_match"] = all(bool(row["match"]) for row in comparisons)
    return result


def sample_torchlinear_modules_by_layer(
    model: nn.Module,
    *,
    proj: str = "q_proj",
    layers_prefix: str = "model.layers",
) -> list[str]:
    """Return one ``TorchLinear`` module path per transformer layer (default: q_proj)."""
    from ..nn_modules.qlinear.torch import TorchLinear

    layer_pattern = re.compile(rf"^{re.escape(layers_prefix)}\.(\d+)\.")
    layer_to_name: dict[int, str] = {}
    suffix = f".self_attn.{proj}"
    for name, module in model.named_modules():
        if not isinstance(module, TorchLinear):
            continue
        match = layer_pattern.match(name)
        if match is None or not name.endswith(suffix):
            continue
        layer_to_name[int(match.group(1))] = name
    return [layer_to_name[index] for index in sorted(layer_to_name)]


def list_torchlinear_module_names(model: nn.Module) -> list[str]:
    """Return all ``TorchLinear`` module paths in ``named_modules`` order."""
    from ..nn_modules.qlinear.torch import TorchLinear

    return [name for name, module in model.named_modules() if isinstance(module, TorchLinear)]


def _summarize_dequant_compare(result: dict[str, object]) -> dict[str, object]:
    comparisons = result.get("comparisons") or []
    mismatches = [row for row in comparisons if not bool(row.get("match"))]
    first_mismatch = mismatches[0]["module"] if mismatches else None
    worst_mean_rel_diff = 0.0
    for row in comparisons:
        rel = row.get("mean_rel_diff")
        if isinstance(rel, (int, float)):
            worst_mean_rel_diff = max(worst_mean_rel_diff, float(rel))
    return {
        "modules_compared": result.get("modules_compared", 0),
        "all_match": result.get("all_match", False),
        "first_mismatch": first_mismatch,
        "worst_mean_rel_diff": worst_mean_rel_diff,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "load_errors": result.get("mismatches", []),
    }


@torch.no_grad()
def compare_inmem_reload_dequant_layers(
    inmem_model: nn.Module,
    reloaded_model: nn.Module,
    *,
    all_modules: bool = False,
    proj: str = "q_proj",
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> dict[str, object]:
    """Compare in-memory vs reload dequant parity across layers or all TorchLinear modules."""
    if all_modules:
        module_names = list_torchlinear_module_names(inmem_model)
        tier = "all_modules"
    else:
        module_names = sample_torchlinear_modules_by_layer(inmem_model, proj=proj)
        tier = "per_layer"

    raw = compare_inmem_reload_dequant(
        inmem_model,
        reloaded_model,
        module_names=module_names,
        max_modules=-1,
        rtol=rtol,
        atol=atol,
    )
    summary = _summarize_dequant_compare(raw)
    summary["tier"] = tier
    summary["module_names_sample"] = module_names[:5]
    summary["layers_compared"] = len(module_names) if tier == "per_layer" else None
    summary["modules_compared"] = raw.get("modules_compared", 0)
    return summary


def capture_torchlinear_env_snapshot() -> dict[str, object]:
    """Snapshot TorchLinear runtime env flags relevant to reload forward audits."""

    def _flag(name: str, default: str = "0") -> bool:
        return os.environ.get(name, default) not in {"0", "false", "False"}

    return {
        "gptq_torch_disable_compile": _flag("GPTQ_TORCH_DISABLE_COMPILE"),
        "gptq_torch_triton_dequant": (
            None
            if os.environ.get("GPTQ_TORCH_TRITON_DEQUANT") is None
            else _flag("GPTQ_TORCH_TRITON_DEQUANT")
        ),
        "gptq_torch_streaming": _flag("GPTQ_TORCH_STREAMING"),
        "gptq_torch_cache_weights": _flag("GPTQ_TORCH_CACHE_WEIGHTS"),
        "gptq_torch_lookahead": _flag("GPTQ_TORCH_LOOKAHEAD"),
    }


def _tensor_rel_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    ref = reference.float()
    cand = candidate.float()
    if ref.shape != cand.shape:
        return {
            "mean_rel_error": float("nan"),
            "max_abs_error": float("nan"),
            "mean_abs_error": float("nan"),
            "allclose": False,
        }
    diff = (cand - ref).abs()
    denom = ref.abs().mean().clamp(min=1e-6)
    return {
        "mean_rel_error": float(diff.mean().item() / denom.item()),
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "allclose": False,
    }


@torch.no_grad()
def measure_torchlinear_forward_vs_dequant(
    module: nn.Module,
    x: torch.Tensor,
    *,
    use_eager_parent: bool = False,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> dict[str, object]:
    """Compare ``TorchLinear.forward`` output vs matmul with dequantized weights."""
    from ..nn_modules.qlinear.torch import TorchLinear

    if not isinstance(module, TorchLinear):
        raise TypeError(f"Expected TorchLinear, got {type(module)!r}")

    module.eval()
    out_fwd = module(x)
    num_itr = module.g_idx.shape[0] // x.shape[-1]
    if use_eager_parent:
        weights = _eager_torchlinear_dequant(module)
    else:
        weights = module.dequantize_weight(num_itr=num_itr)
    weights = weights.to(device=x.device, dtype=x.dtype)
    x_flat = x.reshape(-1, x.shape[-1])
    out_dq = torch.matmul(x_flat, weights).reshape(*x.shape[:-1], module.out_features)
    bias = getattr(module, "bias", None)
    if bias is not None:
        out_dq = out_dq + bias.to(device=out_dq.device, dtype=out_dq.dtype)

    stats = _tensor_rel_error(out_fwd, out_dq)
    stats["allclose"] = bool(torch.allclose(out_fwd, out_dq, rtol=rtol, atol=atol))
    stats["rtol"] = rtol
    stats["atol"] = atol
    stats["use_eager_parent"] = use_eager_parent
    return stats


def _extract_module_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor | None:
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    for key in ("x", "hidden_states", "input"):
        value = kwargs.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


@torch.no_grad()
def capture_torchlinear_inputs(
    model: nn.Module,
    module_names: Sequence[str],
    batch: dict[str, torch.Tensor],
    *,
    use_autocast: bool = True,
) -> dict[str, torch.Tensor]:
    """Run one forward and capture inputs to selected ``TorchLinear`` modules."""
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(name: str):
        def _hook(_module, args, kwargs):
            value = _extract_module_input(args, kwargs)
            if value is not None:
                captured[name] = value.detach()

        return _hook

    for name in module_names:
        module = model.get_submodule(name)
        handles.append(module.register_forward_pre_hook(_make_hook(name), with_kwargs=True))

    model.eval()
    device = next(model.parameters()).device
    batch_on_device = {key: value.to(device) for key, value in batch.items()}
    with torch.amp.autocast("cuda", enabled=use_autocast and device.type == "cuda"):
        model(**batch_on_device)

    for handle in handles:
        handle.remove()
    return captured


@torch.no_grad()
def audit_torchlinear_forward_vs_dequant(
    model: nn.Module,
    module_names: Sequence[str],
    batch: dict[str, torch.Tensor],
    *,
    eager: bool = False,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> dict[str, object]:
    """Measure forward-vs-dequant gap for modules using real activations from one forward."""
    inputs = capture_torchlinear_inputs(model, module_names, batch)
    per_module: dict[str, object] = {}
    missing_inputs: list[str] = []
    for name in module_names:
        x = inputs.get(name)
        if x is None:
            missing_inputs.append(name)
            continue
        module = model.get_submodule(name)
        per_module[name] = measure_torchlinear_forward_vs_dequant(
            module,
            x,
            use_eager_parent=eager,
            rtol=rtol,
            atol=atol,
        )

    rel_errors = [
        float(row["mean_rel_error"])
        for row in per_module.values()
        if isinstance(row, dict) and isinstance(row.get("mean_rel_error"), (int, float))
    ]
    worst_module = None
    worst_rel = 0.0
    for name, row in per_module.items():
        if not isinstance(row, dict):
            continue
        rel = row.get("mean_rel_error")
        if isinstance(rel, (int, float)) and float(rel) >= worst_rel:
            worst_rel = float(rel)
            worst_module = name

    return {
        "modules_audited": len(per_module),
        "missing_inputs": missing_inputs,
        "worst_mean_rel_error": worst_rel if rel_errors else None,
        "worst_module": worst_module,
        "all_match": all(
            bool(row.get("allclose"))
            for row in per_module.values()
            if isinstance(row, dict)
        )
        if per_module
        else False,
        "per_module": per_module,
    }


@torch.no_grad()
def compare_forward_vs_dequant_pre_post(
    inmem_model: nn.Module,
    reloaded_model: nn.Module,
    module_names: Sequence[str],
    batch: dict[str, torch.Tensor],
    *,
    eager: bool = False,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> dict[str, object]:
    """Compare forward-vs-dequant gaps before and after reload on the same modules."""
    pre = audit_torchlinear_forward_vs_dequant(
        inmem_model,
        module_names,
        batch,
        eager=eager,
        rtol=rtol,
        atol=atol,
    )
    post = audit_torchlinear_forward_vs_dequant(
        reloaded_model,
        module_names,
        batch,
        eager=eager,
        rtol=rtol,
        atol=atol,
    )
    gap_delta: dict[str, float | None] = {}
    for name in module_names:
        pre_row = pre.get("per_module", {}).get(name, {})
        post_row = post.get("per_module", {}).get(name, {})
        pre_rel = pre_row.get("mean_rel_error") if isinstance(pre_row, dict) else None
        post_rel = post_row.get("mean_rel_error") if isinstance(post_row, dict) else None
        if isinstance(pre_rel, (int, float)) and isinstance(post_rel, (int, float)):
            gap_delta[name] = float(post_rel) - float(pre_rel)
        else:
            gap_delta[name] = None

    return {
        "pre_inmem": pre,
        "post_reload": post,
        "gap_delta": gap_delta,
    }


def default_forward_audit_module_names(model: nn.Module) -> list[str]:
    """Sample modules for forward-vs-dequant audit (layer 0 q/k/v + mid/deep q_proj)."""
    all_names = set(list_torchlinear_module_names(model))
    per_layer = sample_torchlinear_modules_by_layer(model, proj="q_proj")
    names: list[str] = []
    for suffix in ("q_proj", "k_proj", "v_proj"):
        candidate = f"model.layers.0.self_attn.{suffix}"
        if candidate in all_names:
            names.append(candidate)
    for index in (4, 13, 27):
        for name in per_layer:
            if name.startswith(f"model.layers.{index}."):
                names.append(name)
                break
    if per_layer and per_layer[-1] not in names:
        names.append(per_layer[-1])
    return list(dict.fromkeys(names))


def default_hidden_state_probe_names(model: nn.Module, *, layers_prefix: str = "model.layers") -> list[str]:
    """Return decoder-layer probe names at early/mid/deep positions plus final norm."""
    layer_pattern = re.compile(rf"^{re.escape(layers_prefix)}\.(\d+)$")
    layer_indices: list[int] = []
    for name, _module in model.named_modules():
        match = layer_pattern.match(name)
        if match is not None:
            layer_indices.append(int(match.group(1)))
    layer_indices = sorted(set(layer_indices))
    picks = [index for index in (0, 4, 13, 27) if index in layer_indices]
    if layer_indices and layer_indices[-1] not in picks:
        picks.append(layer_indices[-1])
    probes = [f"{layers_prefix}.{index}" for index in sorted(set(picks))]
    norm_name = f"{layers_prefix.rsplit('.', 1)[0]}.norm" if "." in layers_prefix else "model.norm"
    try:
        model.get_submodule(norm_name)
        probes.append(norm_name)
    except (AttributeError, ModuleNotFoundError):
        pass
    return probes


@torch.no_grad()
def capture_probe_activations(
    model: nn.Module,
    input_batch: dict[str, torch.Tensor],
    probe_names: Sequence[str],
    *,
    use_autocast: bool = True,
) -> dict[str, torch.Tensor]:
    """Capture module outputs for hidden-state probes during one forward."""
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(name: str):
        def _hook(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                captured[name] = out.detach()
            elif isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                captured[name] = out[0].detach()

        return _hook

    for name in probe_names:
        module = model.get_submodule(name)
        handles.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in input_batch.items()}
    with torch.amp.autocast("cuda", enabled=use_autocast and device.type == "cuda"):
        model(**batch)

    for handle in handles:
        handle.remove()
    return captured


@torch.no_grad()
def compare_hidden_states_pre_post(
    pre_model: nn.Module,
    post_model: nn.Module,
    input_batch: dict[str, torch.Tensor],
    probe_names: Sequence[str],
    *,
    threshold: float = 0.05,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> dict[str, object]:
    """Compare hidden-state probe activations between pre-reload and post-reload models."""
    pre_acts = capture_probe_activations(pre_model, input_batch, probe_names)
    post_acts = capture_probe_activations(post_model, input_batch, probe_names)
    probes: dict[str, object] = {}
    first_diverged_probe = None
    for name in probe_names:
        pre = pre_acts.get(name)
        post = post_acts.get(name)
        if pre is None or post is None:
            probes[name] = {
                "missing_pre": pre is None,
                "missing_post": post is None,
                "mean_rel_error": None,
                "allclose": False,
            }
            if first_diverged_probe is None:
                first_diverged_probe = name
            continue
        stats = _tensor_rel_error(pre, post)
        stats["allclose"] = bool(torch.allclose(pre, post, rtol=rtol, atol=atol))
        stats["rtol"] = rtol
        stats["atol"] = atol
        probes[name] = stats
        rel = stats["mean_rel_error"]
        if (
            first_diverged_probe is None
            and isinstance(rel, (int, float))
            and float(rel) > threshold
        ):
            first_diverged_probe = name

    all_match = all(
        isinstance(row, dict) and bool(row.get("allclose"))
        for row in probes.values()
    )
    return {
        "probes": probes,
        "first_diverged_probe": first_diverged_probe,
        "hidden_states_match_pre_post": all_match,
        "threshold": threshold,
    }


def _has_submodule(model: nn.Module, name: str) -> bool:
    try:
        model.get_submodule(name)
        return True
    except (AttributeError, ModuleNotFoundError):
        return False


def default_fine_hidden_state_probe_names(
    model: nn.Module,
    *,
    layer_index: int = 0,
    layers_prefix: str = "model.layers",
) -> list[str]:
    """Return finer layer-0 probes (embed + sub-blocks) that exist on ``model``."""
    prefix = f"{layers_prefix}.{layer_index}"
    candidates = [
        "model.embed_tokens",
        f"{prefix}.input_layernorm",
        f"{prefix}.self_attn",
        f"{prefix}.post_attention_layernorm",
        f"{prefix}.mlp",
        prefix,
    ]
    return [name for name in candidates if _has_submodule(model, name)]


def merge_probe_names(*sequences: Sequence[str]) -> list[str]:
    """Merge probe name lists preserving order and removing duplicates."""
    merged: list[str] = []
    seen: set[str] = set()
    for sequence in sequences:
        for name in sequence:
            if name not in seen:
                seen.add(name)
                merged.append(name)
    return merged


def audit_accelerate_dispatch(model: nn.Module) -> dict[str, object]:
    """Snapshot accelerate dispatch state: ``hf_device_map`` and hook counts."""
    hf_device_map = getattr(model, "hf_device_map", None)
    modules_with_hf_hook: list[str] = []
    align_devices_hook_count = 0
    try:
        from accelerate.hooks import AlignDevicesHook
    except ImportError:
        AlignDevicesHook = None  # type: ignore[misc, assignment]

    for name, module in model.named_modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        modules_with_hf_hook.append(name)
        if AlignDevicesHook is not None and isinstance(hook, AlignDevicesHook):
            align_devices_hook_count += 1

    return {
        "hf_device_map": dict(hf_device_map) if isinstance(hf_device_map, dict) else hf_device_map,
        "hf_device_map_present": hf_device_map is not None,
        "align_devices_hook_count": align_devices_hook_count,
        "modules_with_hf_hook_count": len(modules_with_hf_hook),
        "modules_with_hf_hook": modules_with_hf_hook[:32],
    }


def strip_accelerate_dispatch_hooks(model: nn.Module) -> int:
    """Remove accelerate hooks so forwards use plain parameter devices."""
    from accelerate.hooks import remove_hook_from_module, remove_hook_from_submodules

    before = audit_accelerate_dispatch(model)
    remove_hook_from_submodules(model)
    remove_hook_from_module(model, recurse=False)
    if hasattr(model, "config") and getattr(model.config, "tie_word_embeddings", False):
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    after = audit_accelerate_dispatch(model)
    return int(before.get("modules_with_hf_hook_count", 0)) - int(
        after.get("modules_with_hf_hook_count", 0)
    )


def apply_plain_eval_placement(
    model: nn.Module,
    device: torch.device | str,
) -> dict[str, object]:
    """Strip dispatch hooks, move model to ``device``, and clear TorchLinear caches."""
    dispatch_before = audit_accelerate_dispatch(model)
    strip_accelerate_dispatch_hooks(model)
    model.to(device)
    cleared = clear_torchlinear_inference_state(model)
    dispatch_after = audit_accelerate_dispatch(model)
    param_device = str(next(model.parameters()).device)
    return {
        "dispatch_before": dispatch_before,
        "dispatch_after": dispatch_after,
        "param_device": param_device,
        "torchlinear_caches_cleared": cleared,
    }


PplEvalVariant = Literal["skip_second_to", "single_to", "double_to"]


def prepare_model_for_ppl_eval_variant(
    model: Any,
    eval_device: torch.device | str,
    variant: PplEvalVariant,
) -> torch.device:
    """Apply one post-reload eval placement variant (mirrors benchmark prep)."""
    eval_device = torch.device(eval_device)
    if variant == "skip_second_to":
        clear_torchlinear_inference_state(model.model)
    elif variant == "single_to":
        model.to(eval_device)
        clear_torchlinear_inference_state(model.model)
    elif variant == "double_to":
        model.to(eval_device)
        clear_torchlinear_inference_state(model.model)
        model.to(eval_device)
        clear_torchlinear_inference_state(model.model)
    else:
        raise ValueError(f"Unknown PPL eval variant: {variant!r}")
    return next(model.model.parameters()).device


def prepare_model_for_ppl_eval_variants() -> dict[str, str]:
    """Return descriptions of supported post-reload eval placement ablations."""
    return {
        "skip_second_to": "No extra model.to() after reload; only clear TorchLinear caches.",
        "single_to": "One model.to(eval_device) plus cache clear (benchmark default).",
        "double_to": "Two model.to(eval_device) calls with cache clear between each.",
    }


def reload_gptq_checkpoint_mirror_pre_reload(
    checkpoint_path: str,
    *,
    eval_device: torch.device | str,
    load_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Reload on CPU then plain ``.to(eval_device)`` to mirror pre-reload placement."""
    from .. import GPTQModel

    kwargs = dict(load_kwargs or {})
    kwargs["device"] = "cpu"
    model = GPTQModel.load(checkpoint_path, **kwargs)
    dispatch_before = audit_accelerate_dispatch(model.model)
    placement = apply_plain_eval_placement(model.model, eval_device)
    return {
        "model": model,
        "dispatch_before": dispatch_before,
        "placement": placement,
    }


def _logits_within_threshold(
    logits_delta: dict[str, object] | None,
    *,
    threshold: float,
) -> bool | None:
    if not isinstance(logits_delta, dict):
        return None
    rel = logits_delta.get("mean_rel_error")
    if not isinstance(rel, (int, float)):
        return None
    return float(rel) <= threshold


def compare_reload_path_ablations(
    *,
    reference_logits_delta: dict[str, object] | None,
    ablations: dict[str, dict[str, object]],
    logits_threshold: float = 0.05,
) -> dict[str, object]:
    """Summarize reload path ablation outcomes vs a broken baseline delta."""
    baseline_rel = None
    if isinstance(reference_logits_delta, dict):
        rel = reference_logits_delta.get("mean_rel_error")
        if isinstance(rel, (int, float)):
            baseline_rel = float(rel)

    rows: dict[str, object] = {}
    for name, payload in ablations.items():
        logits_delta = payload.get("logits_pre_vs_ablation") or payload.get("logits_delta")
        row = {
            "logits_pre_vs_ablation": logits_delta,
            "hidden_state_pre_vs_ablation": payload.get("hidden_state_pre_vs_ablation")
            or payload.get("hidden_state_delta"),
            "dispatch_audit": payload.get("dispatch_audit"),
            "placement": payload.get("placement"),
            "ppl": payload.get("ppl"),
            "forward_vs_dequant": payload.get("forward_vs_dequant"),
            "logits_match_pre_reload": _logits_within_threshold(
                logits_delta,
                threshold=logits_threshold,
            ),
        }
        if baseline_rel is not None and isinstance(logits_delta, dict):
            rel = logits_delta.get("mean_rel_error")
            if isinstance(rel, (int, float)):
                row["logits_improved_vs_baseline"] = float(rel) < baseline_rel
        rows[name] = row

    single_rel = None
    double_rel = None
    single_row = rows.get("skip_second_to") or rows.get("single_to")
    double_row = rows.get("double_to")
    if isinstance(single_row, dict):
        ld = single_row.get("logits_pre_vs_ablation")
        if isinstance(ld, dict) and isinstance(ld.get("mean_rel_error"), (int, float)):
            single_rel = float(ld["mean_rel_error"])
    if isinstance(double_row, dict):
        ld = double_row.get("logits_pre_vs_ablation")
        if isinstance(ld, dict) and isinstance(ld.get("mean_rel_error"), (int, float)):
            double_rel = float(ld["mean_rel_error"])

    return {
        "logits_threshold": logits_threshold,
        "baseline_logits_mean_rel_error": baseline_rel,
        "ablations": rows,
        "mirror_pre_reload_fixes_logits": _ablation_fixes(rows, "mirror_pre_reload", logits_threshold),
        "strip_hooks_fixes_logits": _ablation_fixes(rows, "strip_hooks_only", logits_threshold),
        "skip_second_to_fixes_logits": _ablation_fixes(rows, "skip_second_to", logits_threshold),
        "double_to_worsens_logits": (
            single_rel is not None
            and double_rel is not None
            and double_rel > single_rel + 1e-6
        ),
    }


def _ablation_fixes(
    rows: dict[str, object],
    key: str,
    threshold: float,
) -> bool | None:
    row = rows.get(key)
    if not isinstance(row, dict):
        return None
    fixed = row.get("logits_match_pre_reload")
    if isinstance(fixed, bool):
        return fixed
    return _logits_within_threshold(row.get("logits_pre_vs_ablation"), threshold=threshold)


def summarize_reload_forward_audit(
    *,
    dequant_per_layer: dict[str, object] | None,
    dequant_all_modules: dict[str, object] | None,
    forward_vs_dequant: dict[str, object] | None,
    hidden_state_pre_vs_post: dict[str, object] | None,
    device_map_ablation: dict[str, object] | None = None,
    path_ablation: dict[str, object] | None = None,
    fine_hidden_state_pre_vs_post: dict[str, object] | None = None,
    forward_gap_threshold: float = 0.05,
    hidden_threshold: float = 0.05,
    logits_threshold: float = 0.05,
) -> dict[str, object]:
    """Summarize reload forward audit results into at-a-glance verdict fields."""
    packed_weights_ok = bool(dequant_per_layer and dequant_per_layer.get("all_match"))
    if dequant_all_modules is not None:
        packed_weights_ok = packed_weights_ok and bool(dequant_all_modules.get("all_match"))

    pre_fwd = (forward_vs_dequant or {}).get("pre_inmem", {})
    post_fwd = (forward_vs_dequant or {}).get("post_reload_layerwise", {})
    if isinstance(pre_fwd, dict):
        pre_worst = pre_fwd.get("worst_mean_rel_error")
        forward_matches_dequant_pre = (
            pre_worst is None or float(pre_worst) <= forward_gap_threshold
        )
    else:
        forward_matches_dequant_pre = None
    if isinstance(post_fwd, dict):
        post_worst = post_fwd.get("worst_mean_rel_error")
        forward_matches_dequant_post = (
            post_worst is None or float(post_worst) <= forward_gap_threshold
        )
    else:
        forward_matches_dequant_post = None

    hidden_states_match = None
    first_diverged = None
    if hidden_state_pre_vs_post is not None:
        hidden_states_match = bool(hidden_state_pre_vs_post.get("hidden_states_match_pre_post"))
        first_diverged = hidden_state_pre_vs_post.get("first_diverged_probe")

    first_fine_diverged = None
    if fine_hidden_state_pre_vs_post is not None:
        first_fine_diverged = fine_hidden_state_pre_vs_post.get("first_diverged_probe")

    ruled_out: list[str] = []
    if packed_weights_ok:
        ruled_out.append("layerwise_packed_weight_checkpoint_corruption")
    if forward_matches_dequant_pre and forward_matches_dequant_post:
        ruled_out.append("torchlinear_forward_not_equal_dequant_matmul")
    if hidden_states_match:
        ruled_out.append("hidden_state_graph_divergence_pre_vs_post")

    flat_fixed = None
    if device_map_ablation is not None:
        flat_fixed = device_map_ablation.get("flat_map_fixes_logits_or_ppl")

    mirror_pre_reload_fixes_logits = None
    strip_hooks_fixes_logits = None
    skip_second_to_fixes_logits = None
    double_to_worsens_logits = None
    if path_ablation is not None:
        mirror_pre_reload_fixes_logits = path_ablation.get("mirror_pre_reload_fixes_logits")
        strip_hooks_fixes_logits = path_ablation.get("strip_hooks_fixes_logits")
        skip_second_to_fixes_logits = path_ablation.get("skip_second_to_fixes_logits")
        double_to_worsens_logits = path_ablation.get("double_to_worsens_logits")
        if mirror_pre_reload_fixes_logits is True:
            ruled_out.append("cuda_side_load_checkpoint_in_model_only")
        if strip_hooks_fixes_logits is True:
            ruled_out.append("align_devices_hook_dispatch_only")
        if skip_second_to_fixes_logits is True:
            ruled_out.append("second_eval_model_to_corruption")

    return {
        "packed_weights_ok": packed_weights_ok,
        "forward_matches_dequant_pre": forward_matches_dequant_pre,
        "forward_matches_dequant_post": forward_matches_dequant_post,
        "hidden_states_match_pre_post": hidden_states_match,
        "first_diverged_probe": first_diverged,
        "first_fine_diverged_probe": first_fine_diverged,
        "flat_map_ablation_fixes_issue": flat_fixed,
        "mirror_pre_reload_fixes_logits": mirror_pre_reload_fixes_logits,
        "strip_hooks_fixes_logits": strip_hooks_fixes_logits,
        "skip_second_to_fixes_logits": skip_second_to_fixes_logits,
        "double_to_worsens_logits": double_to_worsens_logits,
        "likely_causes_ruled_out": ruled_out,
    }


@torch.no_grad()
def measure_identity_torchlinear_mse(
    fp16_module: nn.Linear,
    quant_module: nn.Module,
    *,
    device: torch.device,
    batch_size: int = 4,
    seq_len: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    """Compare FP16 linear output vs packed TorchLinear on random activations."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(batch_size, seq_len, fp16_module.in_features, device=device, generator=generator)
    x = x.to(dtype=next(fp16_module.parameters()).dtype)

    fp16_out = torch.nn.functional.linear(x, fp16_module.weight, fp16_module.bias)
    quant_out = quant_module(x)

    diff = (fp16_out - quant_out).abs()
    denom = fp16_out.abs().mean().clamp(min=1e-6)
    return {
        "mse": float(torch.mean((fp16_out - quant_out) ** 2).item()),
        "mean_abs_error": float(diff.mean().item()),
        "max_abs_error": float(diff.max().item()),
        "mean_rel_error": float((diff.mean() / denom).item()),
    }


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
            t_x = _stack_t_x(t_w_blocks, dtype=torch.float32, device=x.device)
        else:
            t_x = t_x.to(device=x.device, dtype=torch.float32)
        x_tx = apply_block_transform_to_activation(
            x.float(),
            t_x,
            block_size=block_size,
            pad=pad,
            original_columns=x.shape[-1],
        ).to(dtype=x.dtype)
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
