# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Apply configured transform backends before weight quantization."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import ExecutionConfig, LoopProcessor
from ..looper.named_module import NamedModule
from ..looper.statistics_processor import PTQ_CONTEXT_KEY
from ..models import BaseQModel
from ..ptq.config import TransformPrepareConfig, normalize_transform_prepare, resolve_calibration_nsamples
from ..ptq.context import ModuleCalibContext, TransformState
from ..ptq.transforms.registry import build_transform_backend
from ..quantization.config import QuantizeConfig
from ..utils.device import get_device

PTQ_TRANSFORM_KEY = "ptq_transform_state"


class TransformProcessor(LoopProcessor):
    """Run weight.prepare transforms using captured calibration context."""

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
        calibration,
        prepare_dataset_func,
        calibration_concat_size: Optional[int],
        calibration_sort: Optional[str],
        batch_size: int,
        prepare_configs: Optional[List[TransformPrepareConfig]] = None,
        calibration_concat_separator: Optional[str] = None,
    ):
        super().__init__(
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
        self.prepare_configs = normalize_transform_prepare(
            prepare_configs or getattr(qcfg, "weight_prepare", None)
        )
        self.preserve_batch_keep_mask = True

    def set_calibration_dataset(self, calibration_dataset):
        self.calibration_dataset = calibration_dataset
        self.total_calibration_tokens = LoopProcessor._compute_total_tokens(calibration_dataset)

    def preprocess(self, module: NamedModule, **kwargs):
        del kwargs
        if self.qcfg.dynamic_get(layer_name=module.full_name) is False:
            return
        if not self.prepare_configs:
            module.state[PTQ_TRANSFORM_KEY] = TransformState(method="identity", bake_weights=True)
            return
        module.state["_ptq_prepare_pending"] = True

    def is_skipped(self, module: NamedModule) -> bool:
        if not self.prepare_configs:
            return False
        return not module.state.get("_ptq_prepare_pending", False)

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        del name

        def _noop(*_args, **_kwargs):
            return None

        return _noop

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
        if not self.prepare_configs:
            return
        ctx: ModuleCalibContext | None = module.state.get(PTQ_CONTEXT_KEY)
        expected_nsamples = resolve_calibration_nsamples(self.qcfg, self)
        if ctx is None:
            raise ValueError(
                f"Transform stage missing calibration context for `{module.full_name}` "
                f"(configured nsamples={expected_nsamples})."
            )
        if ctx.nsamples <= 0:
            raise ValueError(
                f"Transform stage received empty calibration statistics for `{module.full_name}` "
                f"(observed nsamples={ctx.nsamples}, configured nsamples={expected_nsamples})."
            )

        weight = module.weight.data
        bias = getattr(module.module, "bias", None)
        if bias is not None:
            bias = bias.data
        opt_device = device or get_device(module.module)

        transform_state = TransformState(method="identity", bake_weights=True)
        qcfg_options = {
            "bits": self.qcfg.bits,
            "group_size": self.qcfg.group_size,
            "sym": self.qcfg.sym,
        }
        for cfg in self.prepare_configs:
            merged_options = {**qcfg_options, **cfg.options}
            merged_cfg = TransformPrepareConfig(
                method=cfg.method,
                mode=cfg.mode,
                bake_weights=cfg.bake_weights,
                options=merged_options,
            )
            backend = build_transform_backend(merged_cfg)
            transform_state = backend.fit(
                weight=weight,
                bias=bias,
                ctx=ctx,
                mode=cfg.mode if cfg.mode in {"standalone", "e2e"} else "standalone",
                device=opt_device,
            )
            if transform_state.bake_weights:
                weight = backend.apply_to_weights(weight, transform_state, device=opt_device)
                module.weight.data = weight

        module.state[PTQ_TRANSFORM_KEY] = transform_state
        module.state.pop("_ptq_prepare_pending", None)
        ctx.transform = transform_state
        module.state[PTQ_CONTEXT_KEY] = ctx

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        del module, model, kwargs

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        del processor_index
        return False

    def name(self) -> str:
        return "transform"
