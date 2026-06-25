# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Capture per-module calibration statistics for the PTQ pipeline."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import (
    DTYPE_SIZE_COLUMN,
    MODULE_FEATURE_COLUMN,
    ExecutionConfig,
    LoopProcessor,
)
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models.writer import (
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_NAME,
    QUANT_LOG_NSAMPLES,
)
from ..ptq.calibration_coverage import (
    describe_calibration_coverage,
    expected_calibration_tokens,
    format_coverage_stat,
)
from ..ptq.stats import StatisticsCollector
from ..quantization.config import QuantizeConfig
from ..quantization.gptq import get_number_of_rows_and_cols
from ..utils.logger import setup_logger

log = setup_logger()

PTQ_STATS_KEY = "ptq_stats_collector"
PTQ_CONTEXT_KEY = "ptq_calib_context"


class StatisticsProcessor(LoopProcessor):
    """Forward-hook processor that streams activation statistics into CPU-backed collectors."""

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
        calibration,
        prepare_dataset_func,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        calibration_concat_separator: Optional[str] = None,
    ):
        kwargs = dict(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            prepare_dataset_func=prepare_dataset_func,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            calibration_concat_separator=calibration_concat_separator,
            batch_size=batch_size,
            execution_config=ExecutionConfig(
                require_fwd=True,
                fwd_replay_after_process=True,
                subset_forward_early_stop=True,
            ),
        )
        super().__init__(**kwargs)
        self._collectors: Dict[str, StatisticsCollector] = {}
        self.preserve_batch_keep_mask = True

    def preprocess(self, module: NamedModule, **kwargs):
        del kwargs
        if self.qcfg.dynamic_get(layer_name=module.full_name) is False:
            return
        _, columns = get_number_of_rows_and_cols(module)
        row_budget = getattr(self.qcfg.hessian, "row_buffer_max_rows", None)
        collector = StatisticsCollector(
            columns=columns,
            hessian=self.qcfg.hessian,
            row_buffer_max_rows=row_budget,
        )
        self._collectors[module.name] = collector
        module.state[PTQ_STATS_KEY] = collector

    def is_skipped(self, module: NamedModule) -> bool:
        return module.name not in self._collectors

    def has_captured_input_ids(self, name: str) -> bool:
        collector = self._collectors.get(name)
        return collector is not None and collector.nsamples > 0

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook(_module, inp: Tuple[torch.Tensor, ...], _out: torch.Tensor):
            collector = self._collectors.get(name)
            if collector is None or not inp:
                return

            inp_tensor = inp[0]
            keep_mask = getattr(getattr(self, "_mask_tls", None), "value", None)

            if (
                torch.is_tensor(inp_tensor)
                and torch.is_tensor(keep_mask)
                and inp_tensor.dim() >= 3
                and keep_mask.ndim == 2
                and keep_mask.shape[:2] == inp_tensor.shape[:2]
            ):
                for sample_index, sample_keep in enumerate(keep_mask):
                    if not bool(sample_keep.any().item()):
                        continue
                    sample_inp = inp_tensor[sample_index : sample_index + 1, sample_keep, :].contiguous()
                    collector.add_batch(sample_inp.detach())
            else:
                collector.add_batch(inp_tensor.detach())

        return hook

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        del device, subset, previous_subset, subset_index, subset_total
        collector = self._collectors.get(module.name)
        if collector is None:
            return
        rows, _ = get_number_of_rows_and_cols(module)
        expected_tokens = expected_calibration_tokens(self)
        ctx = collector.to_context(
            module_name=module.full_name,
            rows=rows,
            expected_calibration_tokens=expected_tokens,
        )
        observed_rows = ctx.nsamples
        if observed_rows <= 0:
            raise ValueError(
                f"Statistics capture collected 0 activation rows for `{module.full_name}` "
                f"(expected_calibration_tokens={expected_tokens}, observed_rows=0). "
                f"Ensure the module receives calibration forwards before PTQ quantization."
            )
        if observed_rows != expected_tokens:
            log.warn(describe_calibration_coverage(
                module_name=module.full_name,
                observed_rows=observed_rows,
                expected_tokens=expected_tokens,
            ))

        module.state[PTQ_CONTEXT_KEY] = ctx

        stat = {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_NSAMPLES: f"{observed_rows}",
            "expected_calibration_tokens": str(expected_tokens),
            "coverage": format_coverage_stat(
                observed_rows=observed_rows,
                expected_tokens=expected_tokens,
            ),
        }
        with self.lock:
            self.log.append(stat)
        self.log_new_row(stat)

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        del model, kwargs
        collector = self._collectors.pop(module.name, None)
        if collector is not None:
            collector.free()
        module.state.pop(PTQ_STATS_KEY, None)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        del processor_index
        if self.calibration_dataset is None:
            raise ValueError("StatisticsProcessor requires a calibration dataset.")
        return True

    def name(self) -> str:
        return "statistics"
