# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Deprecated statistics processor — replaced by SequentialPTQProcessor."""

from __future__ import annotations

from .ptq_keys import PTQ_CONTEXT_KEY, PTQ_STATS_KEY

__all__ = ["PTQ_CONTEXT_KEY", "PTQ_STATS_KEY", "StatisticsProcessor"]


class StatisticsProcessor:
    """Removed. Use SequentialPTQProcessor via build_gpt_quantizer_processors()."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "StatisticsProcessor was removed; use SequentialPTQProcessor "
            "(capture → transform → quantize per module)."
        )
