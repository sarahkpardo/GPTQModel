# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import Optional

from ..ptq.config import WeightQuantizeTargetConfig
from .quantizer_processor import QuantizerProcessor, clone_gptq_config_for_module

__all__ = ["GPTQProcessor", "QuantizerProcessor", "clone_gptq_config_for_module"]


class GPTQProcessor(QuantizerProcessor):
    """Deprecated inline GPTQ processor; use SequentialPTQProcessor via build_gpt_quantizer_processors."""

    def __init__(
        self,
        tokenizer,
        qcfg,
        calibration,
        prepare_dataset_func,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        require_fwd: bool = True,
        calculate_w_wq_diff: bool = False,
        calibration_concat_separator: Optional[str] = None,
        weight_quantize: WeightQuantizeTargetConfig | None = None,
        capture_mode: str | None = None,
    ):
        if weight_quantize is None:
            weight_quantize = WeightQuantizeTargetConfig(method="gptq")
        if capture_mode is None:
            capture_mode = "inline"

        if capture_mode == "inline":
            import warnings

            warnings.warn(
                "GPTQProcessor inline capture is deprecated; the default GPTQ pipeline "
                "uses SequentialPTQProcessor (capture → transform → quantize per module).",
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            prepare_dataset_func=prepare_dataset_func,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            batch_size=batch_size,
            require_fwd=require_fwd,
            calculate_w_wq_diff=calculate_w_wq_diff,
            calibration_concat_separator=calibration_concat_separator,
            weight_quantize=weight_quantize,
            capture_mode=capture_mode,
        )
