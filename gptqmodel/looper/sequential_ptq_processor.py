# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Paper-aligned PTQ processor: per-module capture → transform → quantize in one layer pass."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import (
    DTYPE_SIZE_COLUMN,
    MODULE_FEATURE_COLUMN,
    ExecutionConfig,
    LoopProcessor,
)
from ..looper.named_module import NamedModule
from ..looper.quantizer_processor import QuantizerProcessor, clone_gptq_config_for_module
from ..looper.statistics_processor import PTQ_CONTEXT_KEY, PTQ_STATS_KEY
from ..looper.transform_processor import PTQ_TRANSFORM_KEY
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
from ..ptq.config import TransformPrepareConfig, WeightQuantizeTargetConfig, normalize_transform_prepare
from ..ptq.context import ModuleCalibContext, TransformState
from ..ptq.pipeline import ModuleQuantizationPipeline
from ..ptq.stats import StatisticsCollector
from ..quantization.config import QuantizeConfig
from ..quantization.gptq import get_number_of_rows_and_cols
from ..utils.device import get_device
from ..utils.looper_helpers import normalize_device_like
from ..utils.logger import setup_logger

log = setup_logger()


class SequentialPTQProcessor(QuantizerProcessor):
    """Interleaved statistics capture, transform, and quantize per subset (Chen et al. ordering)."""

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
        calibration,
        prepare_dataset_func,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        require_fwd: bool = True,
        calculate_w_wq_diff: bool = False,
        calibration_concat_separator: Optional[str] = None,
        weight_quantize: WeightQuantizeTargetConfig | None = None,
        prepare_configs: Optional[List[TransformPrepareConfig]] = None,
    ):
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
            capture_mode="none",
        )
        self.execution_config = ExecutionConfig(
            require_fwd=True,
            fwd_replay_after_process=True,
            subset_forward_early_stop=True,
        )
        self.prepare_configs = normalize_transform_prepare(
            prepare_configs or getattr(qcfg, "weight_prepare", None)
        )
        self._pipeline = ModuleQuantizationPipeline(
            qcfg=qcfg,
            prepare_configs=self.prepare_configs,
            weight_quantize=weight_quantize,
        )
        self._collectors: Dict[str, StatisticsCollector] = {}
        self._collectors_by_module_id: Dict[int, StatisticsCollector] = {}
        self._collectors_by_short_name: Dict[str, StatisticsCollector] = {}
        self._paroquant_prewarm_done = False

    def _maybe_prewarm_paroquant(self) -> None:
        if self._paroquant_prewarm_done:
            return
        paro_cfg = next(
            (cfg for cfg in self.prepare_configs if cfg.method in {"paroquant", "paro"}),
            None,
        )
        if paro_cfg is None:
            return

        from ..utils.paroquant import prewarm_paroquant_rotation_extension

        opts = paro_cfg.options
        device = normalize_device_like(self.qcfg.device)
        if device is None and torch.cuda.is_available():
            device = torch.device("cuda")
        prewarmed = prewarm_paroquant_rotation_extension(
            fused_rotation=bool(opts.get("opt_fused_rotation", True)),
            group_size=int(opts.get("group_size", self.qcfg.group_size)),
            krot=int(opts.get("krot", 8)),
            device=device,
        )
        if not prewarmed:
            log.info(
                "SequentialPTQProcessor: ParoQuant fused rotation prewarm skipped "
                f"(group_size={opts.get('group_size', self.qcfg.group_size)}, "
                f"krot={opts.get('krot', 8)}); reference rotation will be used during fit."
            )
        self._paroquant_prewarm_done = True

    @staticmethod
    def _collector_key(module: NamedModule) -> str:
        return module.full_name

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        del kwargs
        self._maybe_prewarm_paroquant()
        qcfg_clone = clone_gptq_config_for_module(
            self.qcfg,
            module.full_name,
            fallback=fallback,
        )
        if qcfg_clone is None:
            return

        self.qcfg_dynamic = qcfg_clone
        self._split_modules[module.name] = True

        _, columns = get_number_of_rows_and_cols(module)
        row_budget = getattr(self.qcfg.hessian, "row_buffer_max_rows", None)
        collector = StatisticsCollector(
            columns=columns,
            hessian=self.qcfg.hessian,
            row_buffer_max_rows=row_budget,
        )
        collector_key = self._collector_key(module)
        self._collectors[collector_key] = collector
        self._collectors_by_module_id[id(module.module)] = collector
        self._collectors_by_short_name[module.name] = collector
        module.state[PTQ_STATS_KEY] = collector

        if not self.prepare_configs:
            module.state[PTQ_TRANSFORM_KEY] = TransformState(method="identity", bake_weights=True)
        else:
            module.state["_ptq_prepare_pending"] = True

    def is_skipped(self, module: NamedModule) -> bool:
        return module.name not in self._split_modules

    def has_captured_input_ids(self, name: str) -> bool:
        if name not in self._split_modules:
            return False
        collector = self._collectors_by_short_name.get(name)
        if collector is not None:
            return collector.nsamples > 0
        for key, candidate in self._collectors.items():
            if key == name or key.endswith(f".{name}"):
                return candidate.nsamples > 0
        return False

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook(_module, inp: Tuple[torch.Tensor, ...], _out: torch.Tensor):
            collector = self._collectors_by_module_id.get(id(_module))
            if collector is None:
                collector = self._collectors_by_short_name.get(name)
            if collector is None:
                module_key = getattr(_module, "module_name", None)
                if module_key:
                    collector = self._collectors.get(module_key)
            if collector is None:
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

    def _finalize_stats_context(self, module: NamedModule) -> ModuleCalibContext:
        collector = self._collectors_by_module_id.get(id(module.module))
        if collector is None:
            collector = self._collectors.get(self._collector_key(module))
        if collector is None:
            raise ValueError(f"Sequential PTQ missing statistics collector for `{module.full_name}`.")

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
            log.warn(
                describe_calibration_coverage(
                    module_name=module.full_name,
                    observed_rows=observed_rows,
                    expected_tokens=expected_tokens,
                )
            )

        module.state[PTQ_CONTEXT_KEY] = ctx

        stat = {
            PROCESS_LOG_NAME: "statistics",
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
        return ctx

    def _apply_transform(self, module: NamedModule, ctx: ModuleCalibContext, device: torch.device | None):
        if not self.prepare_configs:
            return

        expected_tokens = expected_calibration_tokens(self)
        if ctx.nsamples <= 0:
            raise ValueError(
                f"Transform stage received empty calibration statistics for `{module.full_name}` "
                f"(observed_rows={ctx.nsamples}, expected_calibration_tokens={expected_tokens})."
            )

        opt_device = device or get_device(module.module) or get_device(module)
        bias = getattr(module.module, "bias", None)
        if bias is not None:
            bias = bias.data
        weight, transform_state = self._pipeline.apply_transform(
            module=module.module,
            weight=module.weight.data,
            bias=bias,
            ctx=ctx,
            device=opt_device,
        )
        module.weight.data = weight
        module.state[PTQ_TRANSFORM_KEY] = transform_state
        module.state.pop("_ptq_prepare_pending", None)
        module.state[PTQ_CONTEXT_KEY] = ctx

    def _transform_requires_grad(self) -> bool:
        return any(cfg.method not in {"identity", "none"} for cfg in self.prepare_configs)

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        del subset, previous_subset, subset_index, subset_total
        ctx = self._finalize_stats_context(module)
        if self._transform_requires_grad():
            with torch.inference_mode(False), torch.enable_grad():
                self._apply_transform(module, ctx, device)
        else:
            self._apply_transform(module, ctx, device)
        self._process_split(module, device=device)

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        self._collectors_by_module_id.pop(id(module.module), None)
        self._collectors_by_short_name.pop(module.name, None)
        collector = self._collectors.pop(self._collector_key(module), None)
        if collector is not None:
            collector.free()
        module.state.pop(PTQ_STATS_KEY, None)
        super().submodule_finalize(module, model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        del processor_index
        if self.calibration_dataset is None:
            raise ValueError("SequentialPTQProcessor requires a calibration dataset.")
        return True

    def name(self) -> str:
        return "sequential-ptq"
