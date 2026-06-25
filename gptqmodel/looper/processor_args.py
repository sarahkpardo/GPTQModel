# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shared keyword filtering for calibration vs quantizer LoopProcessors."""

from __future__ import annotations

from typing import Any, Dict

CALIBRATION_PROCESSOR_KEYS = frozenset({
    "tokenizer",
    "qcfg",
    "calibration",
    "prepare_dataset_func",
    "calibration_concat_size",
    "calibration_sort",
    "calibration_concat_separator",
    "batch_size",
})

QUANTIZER_ONLY_KEYS = frozenset({
    "calculate_w_wq_diff",
    "require_fwd",
})


def calibration_processor_kwargs(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return kwargs accepted by statistics/transform calibration processors."""
    return {k: v for k, v in args.items() if k in CALIBRATION_PROCESSOR_KEYS}


def quantizer_processor_kwargs(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return calibration kwargs plus quantizer-only flags."""
    return {
        **calibration_processor_kwargs(args),
        **{k: v for k, v in args.items() if k in QUANTIZER_ONLY_KEYS},
    }


def build_gpt_quantizer_processors(qcfg, args, preprocessors):
    """Build the paper-aligned sequential PTQ processor for GPTQ method."""
    from ..looper.sequential_ptq_processor import SequentialPTQProcessor
    from ..ptq.config import resolve_weight_quantize_target

    quant_args = quantizer_processor_kwargs(args)
    target = resolve_weight_quantize_target(qcfg)

    return preprocessors + [
        SequentialPTQProcessor(**quant_args, weight_quantize=target),
    ]


def native_processor_kwargs(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return kwargs for NativeProcessor, excluding quantizer-only flags."""
    filtered = dict(args)
    for key in QUANTIZER_ONLY_KEYS:
        filtered.pop(key, None)
    return filtered
