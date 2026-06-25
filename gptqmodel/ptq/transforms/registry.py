# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..config import TransformPrepareConfig
from ..protocols import TransformBackend
from .identity import IdentityTransform


def build_transform_backend(cfg: TransformPrepareConfig) -> TransformBackend:
    method = cfg.method.strip().lower()
    if method in {"identity", "none"}:
        return IdentityTransform()
    if method in {"paroquant", "paro"}:
        from .paroquant import ParoQuantTransform

        return ParoQuantTransform(cfg)
    if method == "wush":
        from .wush import WUSHTransform

        return WUSHTransform(cfg)
    raise ValueError(f"Unsupported transform method `{cfg.method}`.")
