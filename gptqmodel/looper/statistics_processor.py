# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Capture per-module calibration statistics for the PTQ pipeline."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import ExecutionConfig, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..ptq.stats import StatisticsCollector
from ..quantization.config import QuantizeConfig
from ..quantization.gptq import get_number_of_rows_and_cols

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
                fwd_replay_after_process=False,
                subset_forward_early_stop=True,
            ),
        )
        super().__init__(**kwargs)
        self._collectors: Dict[str, StatisticsCollector] = {}

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

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook(_module, inp, _out):
            collector = self._collectors.get(name)
            if collector is None:
                return
            if not inp:
                return
            collector.add_batch(inp[0].detach())

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
        ctx = collector.to_context(module_name=module.full_name, rows=rows)
        module.state[PTQ_CONTEXT_KEY] = ctx

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
