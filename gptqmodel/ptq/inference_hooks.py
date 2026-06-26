# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Attach inference-time activation transforms to torch modules."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

from .config import TransformPrepareConfig
from .context import TransformState
from .inference_data import InferenceTransformData
from .transforms.block_dense import build_dense_activation_hook
from .transforms.registry import build_transform_backend

PTQ_BUFFER_NAMES = (
    "ptq_t_x_matrices",
    "ptq_t_x_block_size",
    "ptq_t_x_pad",
    "ptq_transform_method_bytes",
)


def resolve_inference_transform(
    transform: TransformState | None,
    *,
    inference: InferenceTransformData | None = None,
) -> InferenceTransformData:
    if inference is not None:
        return inference
    if transform is not None and transform.inference is not None:
        return transform.inference
    if transform is None or transform.method == "identity":
        return InferenceTransformData(transform_type="identity")
    backend = build_transform_backend(TransformPrepareConfig(method=transform.method))
    return backend.get_inference_data(transform)


def _encode_method_name(method: str) -> torch.Tensor:
    return torch.tensor(list(method.encode("utf-8")), dtype=torch.uint8)


def _decode_method_name(buffer: torch.Tensor) -> str:
    if buffer is None or buffer.numel() == 0:
        return "identity"
    return bytes(buffer.tolist()).decode("utf-8")


def persist_ptq_inference_buffers(
    module: nn.Module,
    inference: InferenceTransformData,
    *,
    method: str,
    pad: int = 0,
) -> None:
    """Register checkpoint-persistent buffers for online ``T_X`` rehydration."""
    if inference.is_identity():
        return

    t_x = inference.T_X_matrices
    if t_x is None:
        return

    module.register_buffer("ptq_t_x_matrices", t_x.detach().cpu(), persistent=True)
    module.register_buffer(
        "ptq_t_x_block_size",
        torch.tensor(int(inference.block_size), dtype=torch.int32),
        persistent=True,
    )
    module.register_buffer(
        "ptq_t_x_pad",
        torch.tensor(int(pad), dtype=torch.int32),
        persistent=True,
    )
    module.register_buffer(
        "ptq_transform_method_bytes",
        _encode_method_name(method),
        persistent=True,
    )


def _inference_from_buffers(module: nn.Module) -> InferenceTransformData | None:
    t_x = getattr(module, "ptq_t_x_matrices", None)
    if not isinstance(t_x, torch.Tensor) or t_x.numel() == 0:
        return None

    block_size = int(getattr(module, "ptq_t_x_block_size", torch.tensor(0)).item())
    pad = int(getattr(module, "ptq_t_x_pad", torch.tensor(0)).item())
    method = _decode_method_name(getattr(module, "ptq_transform_method_bytes", None))
    transform_type = "dense" if method in {"random_orthogonal", "rand_ortho", "quip_incoherence"} else method
    precision = t_x.dtype if t_x.is_floating_point() else torch.float16
    return InferenceTransformData(
        transform_type=transform_type,
        T_X_matrices=t_x,
        block_size=block_size,
        precision=precision,
        extra={"pad": pad, "method": method},
    )


def _checkpoint_weight_files(checkpoint_path: str) -> list[str]:
    path = Path(checkpoint_path)
    if path.name.endswith(".safetensors.index.json"):
        import json

        with path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        base = path.parent
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        return [str(base / shard_name) for shard_name in shard_names if shard_name]
    if path.is_dir():
        return [str(file_path) for file_path in sorted(path.glob("*.safetensors"))]
    return [str(path)]


def load_ptq_inference_buffers_from_checkpoint(model: nn.Module, checkpoint_path: str) -> int:
    """Load ``ptq_*`` buffers saved in a quantized checkpoint into ``TorchLinear`` modules."""
    from safetensors import safe_open

    module_tensors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for file_path in _checkpoint_weight_files(checkpoint_path):
        with safe_open(file_path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                for name in PTQ_BUFFER_NAMES:
                    suffix = f".{name}"
                    if not key.endswith(suffix):
                        continue
                    prefix = key[: -len(suffix)]
                    module_tensors[prefix][name] = reader.get_tensor(key)
                    break

    loaded = 0
    for prefix, tensors in module_tensors.items():
        if "ptq_t_x_matrices" not in tensors:
            continue
        try:
            module = model.get_submodule(prefix)
        except (AttributeError, ModuleNotFoundError):
            continue
        for name, tensor in tensors.items():
            existing = getattr(module, name, None)
            if isinstance(existing, torch.Tensor):
                existing.copy_(tensor)
            else:
                module.register_buffer(name, tensor, persistent=True)
        loaded += 1
    return loaded


def build_activation_pre_hook(
    inference: InferenceTransformData,
    *,
    transform: TransformState | None = None,
) -> Callable[..., None]:
    """Build a forward pre-hook from inference metadata."""
    if inference.transform_type == "dense" or inference.T_X_matrices is not None:
        pad = int(inference.extra.get("pad", 0) or 0)
        return build_dense_activation_hook(inference, pad=pad)

    method = transform.method if transform is not None else inference.transform_type
    backend = build_transform_backend(TransformPrepareConfig(method=method))
    if transform is not None:
        return backend.activation_pre_hook(transform)
    return backend.activation_pre_hook(TransformState(method=method, inference=inference))


def register_activation_pre_hook(
    module: nn.Module,
    transform: TransformState | None,
    *,
    inference: InferenceTransformData | None = None,
) -> Optional[Callable[..., None]]:
    """Register online ``T_X`` when the inference payload is non-identity.

    ``bake_weights=True`` only means ``T_W`` was applied offline; ``T_X`` still
    runs at inference to preserve the bilinear inner product.
    """
    inference_data = resolve_inference_transform(transform, inference=inference)
    if inference_data.is_identity():
        return None

    hook = build_activation_pre_hook(inference_data, transform=transform)
    module.register_forward_pre_hook(hook, with_kwargs=True)
    setattr(module, "_ptq_inference_transform", inference_data.to_dict())
    return hook


def rehydrate_ptq_inference_hooks(model: nn.Module) -> int:
    """Re-register ``T_X`` hooks from persistent buffers after checkpoint load."""
    restored = 0
    for module in model.modules():
        inference = _inference_from_buffers(module)
        if inference is None:
            continue
        if module._forward_pre_hooks:
            continue
        register_activation_pre_hook(module, transform=None, inference=inference)
        restored += 1
    return restored
