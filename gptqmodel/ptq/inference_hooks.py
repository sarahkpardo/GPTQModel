# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Attach inference-time activation transforms to torch modules."""

from __future__ import annotations

from typing import Callable, Optional

import torch.nn as nn

from .context import TransformState
from .inference_data import InferenceTransformData
from .transforms.registry import build_transform_backend
from .config import TransformPrepareConfig


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

    method = transform.method if transform is not None else inference_data.transform_type
    backend = build_transform_backend(TransformPrepareConfig(method=method))
    hook = backend.activation_pre_hook(transform or TransformState(method=method))
    module.register_forward_pre_hook(hook, with_kwargs=True)
    setattr(module, "_ptq_inference_transform", inference_data.to_dict())
    return hook
