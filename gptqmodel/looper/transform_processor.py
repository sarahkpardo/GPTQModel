# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Deprecated transform processor — replaced by SequentialPTQProcessor."""

from __future__ import annotations

from .ptq_keys import PTQ_TRANSFORM_KEY

__all__ = ["PTQ_TRANSFORM_KEY", "TransformProcessor"]


class TransformProcessor:
    """Removed. Use SequentialPTQProcessor via build_gpt_quantizer_processors()."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "TransformProcessor was removed; use SequentialPTQProcessor "
            "(capture → transform → quantize per module)."
        )
