# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Generic weight quantizer processor for split PTQ capture (SequentialPTQProcessor)."""

from __future__ import annotations

import copy
import threading
import time
from typing import Callable, Dict, Literal, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import DTYPE_SIZE_COLUMN, ExecutionConfig, MODULE_FEATURE_COLUMN, LoopProcessor
from ..looper.named_module import NamedModule
from ..looper.ptq_keys import PTQ_CONTEXT_KEY, PTQ_TRANSFORM_KEY
from ..models import BaseQModel
from ..models._const import CPU
from ..models.writer import (
    PROCESS_LOG_FWD_TIME,
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_NAME,
    PROCESS_LOG_TIME,
    PROCESS_USED_MEMORY,
    QUANT_LOG_DAMP,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
)
from ..ptq.calibration_coverage import expected_calibration_tokens
from ..ptq.config import WeightQuantizeTargetConfig, resolve_weight_quantize_target
from ..ptq.optimizers.registry import build_weight_optimizer, weight_optimizer_requires_calibration
from ..quantization.config import FOEMConfig, GPTAQConfig, HessianConfig, METHOD, QuantizeConfig, resolve_quant_format
from ..utils.device import get_device
from ..utils.fallback import normalize_fallback
from ..utils.logger import log_time_block, setup_logger
from ..utils.model import create_quant_module, find_modules, pack_module
from ..utils.module_locks import parent_module_lock
log = setup_logger()
lock = threading.Lock()

CaptureMode = Literal["none"]


def clone_gptq_config_for_module(
    qcfg: QuantizeConfig,
    module_full_name: str,
    *,
    fallback=None,
) -> Optional[QuantizeConfig]:
    """Clones and applies per-module GPTQ dynamic overrides, or skips the module."""

    if qcfg.dynamic_get(layer_name=module_full_name) is False:
        return None

    qcfg_clone = copy.deepcopy(qcfg)

    if qcfg.dynamic is not None:
        qcfg_clone.bits = qcfg.dynamic_get(module_full_name, "bits", qcfg_clone.bits)
        qcfg_clone.sym = qcfg.dynamic_get(module_full_name, "sym", qcfg_clone.sym)
        qcfg_clone.mse = qcfg.dynamic_get(module_full_name, "mse", qcfg_clone.mse)

        qcfg_clone.group_size = qcfg.dynamic_get(module_full_name, "group_size", qcfg_clone.group_size)
        desc_act_override = qcfg.dynamic_get(module_full_name, "desc_act", None)
        if desc_act_override is not None:
            qcfg_clone.desc_act = desc_act_override
        act_group_aware_override = qcfg.dynamic_get(module_full_name, "act_group_aware", None)
        if act_group_aware_override is not None:
            qcfg_clone.act_group_aware = act_group_aware_override
        qcfg_clone.damp_percent = qcfg.dynamic_get(module_full_name, "damp_percent", qcfg_clone.damp_percent)
        qcfg_clone.static_groups = qcfg.dynamic_get(module_full_name, "static_groups", qcfg_clone.static_groups)
        fallback_override = qcfg.dynamic_get(module_full_name, "fallback", None)
        if fallback_override is not None:
            qcfg_clone.fallback = normalize_fallback(fallback_override, qcfg_clone.fallback)
        hessian_override = qcfg.dynamic_get(module_full_name, "hessian", None)
        if hessian_override is not None:
            if isinstance(hessian_override, dict):
                qcfg_clone.hessian = HessianConfig(**hessian_override)
            elif isinstance(hessian_override, HessianConfig):
                qcfg_clone.hessian = hessian_override
            else:
                raise ValueError("QuantizeConfig: dynamic `hessian` must be a HessianConfig or dict.")
        gptaq_override = qcfg.dynamic_get(module_full_name, "gptaq", None)
        foem_override = qcfg.dynamic_get(module_full_name, "foem", None)
        if gptaq_override is not None:
            if isinstance(gptaq_override, dict):
                qcfg_clone.gptaq = GPTAQConfig(**gptaq_override)
            elif isinstance(gptaq_override, GPTAQConfig):
                qcfg_clone.gptaq = gptaq_override
            else:
                raise ValueError("QuantizeConfig: dynamic `gptaq` must be a GPTAQConfig or dict.")
        if foem_override is not None:
            if isinstance(foem_override, dict):
                qcfg_clone.foem = FOEMConfig(**foem_override)
            elif isinstance(foem_override, FOEMConfig):
                qcfg_clone.foem = foem_override
            else:
                raise ValueError("QuantizeConfig: dynamic `foem` must be a FOEMConfig or dict.")

        qcfg_clone._resolve_activation_ordering(desc_act_override, act_group_aware_override)

    qcfg_clone.fallback = normalize_fallback(fallback, qcfg_clone.fallback)
    return qcfg_clone


class QuantizerProcessor(LoopProcessor):
    """Quantize module weights via a selectable WeightOptimizerBackend."""

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
        capture_mode: CaptureMode = "none",
    ):
        if capture_mode != "none":
            raise ValueError(
                "QuantizerProcessor only supports capture_mode='none'; "
                "use SequentialPTQProcessor for the production GPTQ pipeline."
            )
        self.capture_mode = capture_mode
        target = weight_quantize or resolve_weight_quantize_target(qcfg)
        self.weight_quantize = target

        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            calibration_concat_separator=calibration_concat_separator,
            prepare_dataset_func=prepare_dataset_func,
            batch_size=batch_size,
            execution_config=ExecutionConfig(
                require_fwd=False,
                fwd_replay_after_process=True,
                subset_forward_early_stop=False,
            ),
        )

        self.optimizer = build_weight_optimizer(target, qcfg)
        self.calculate_w_wq_diff = calculate_w_wq_diff
        self.avg_losses = []
        self.preserve_batch_keep_mask = True
        self._split_modules: Dict[str, bool] = {}

    def set_calibration_dataset(self, calibration_dataset):
        self.calibration_dataset = calibration_dataset
        self.total_calibration_tokens = LoopProcessor._compute_total_tokens(calibration_dataset)

    def preprocess(self, module: NamedModule, fallback=None, **kwargs):
        del kwargs
        qcfg_clone = clone_gptq_config_for_module(
            self.qcfg,
            module.full_name,
            fallback=fallback,
        )
        if qcfg_clone is None:
            return

        self.qcfg_dynamic = qcfg_clone
        self._split_modules[module.name] = True

    def is_skipped(self, module: NamedModule) -> bool:
        return module.name not in self._split_modules

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
        self._process_split(module, device=device)

    def _process_split(self, module: NamedModule, device: torch.device = None):
        base_title = f"Quantizing {module.name} in layer"
        self.draw_progress(base_title)

        ctx = module.state.get(PTQ_CONTEXT_KEY)
        if ctx is None:
            raise ValueError(f"PTQ split pipeline missing calibration context for `{module.full_name}`.")

        if weight_optimizer_requires_calibration(self.weight_quantize):
            if ctx.nsamples <= 0:
                expected_tokens = expected_calibration_tokens(self)
                raise ValueError(
                    f"Quantizer `{self.weight_quantize.method}` requires calibration statistics "
                    f"for `{module.full_name}`, but observed_rows={ctx.nsamples} "
                    f"(expected_calibration_tokens={expected_tokens})."
                )
            if ctx.H is None:
                raise ValueError(
                    f"Quantizer `{self.weight_quantize.method}` requires a Hessian for "
                    f"`{module.full_name}`, but calibration context has H=None."
                )

        transform = module.state.get(PTQ_TRANSFORM_KEY)
        opt_device = device or get_device(module.module)
        if opt_device is None:
            opt_device = get_device(module)

        start = time.perf_counter()
        weight_quant = self.optimizer.optimize(
            module=module,
            ctx=ctx,
            transform=transform,
            device=opt_device,
            qcfg=self.qcfg_dynamic or self.qcfg,
            expected_nsamples=expected_calibration_tokens(self),
        )
        duration = time.perf_counter() - start

        backend = weight_quant.extra.get("gptq") or weight_quant.extra.get("rtn")
        if backend is not None and hasattr(backend, "free"):
            backend.free()

        wq = weight_quant.pack_weight
        if wq is None:
            raise ValueError(f"Weight optimizer `{self.weight_quantize.method}` did not return pack_weight.")

        avg_loss = weight_quant.extra.get("loss", "unknown")
        damp_percent = getattr(self.qcfg_dynamic or self.qcfg, "damp_percent", 0.0) or 0.0
        nsamples = ctx.nsamples

        self._finalize_module_quant(
            module,
            wq=wq,
            q_scales=weight_quant.q_scales,
            q_zeros=weight_quant.q_zeros,
            q_g_idx=weight_quant.q_g_idx,
            duration=duration,
            avg_loss=avg_loss,
            damp_percent=damp_percent,
            nsamples=nsamples,
            weight_quant_extra=weight_quant.extra,
        )

    def _preserve_pseudo_weight_for_replay(self, weight_quant_extra=None) -> bool:
        """Keep dequantized pseudo weights on nn.Linear modules until ParoQuant export."""
        extra = weight_quant_extra or {}
        if extra.get("paroquant"):
            return True
        export_payload = getattr(self.qcfg, "weight_export", None)
        if not export_payload:
            return False
        export_format = str(export_payload.get("format", "")).strip().lower()
        return export_format in {"paroquant", "paro"}

    def _finalize_module_quant(
        self,
        module: NamedModule,
        *,
        wq,
        q_scales,
        q_zeros,
        q_g_idx,
        duration,
        avg_loss,
        damp_percent,
        nsamples,
        workspace_summary=None,
        workspace_totals=None,
        weight_quant_extra=None,
    ):
        module.stream_state_payload_to_cpu(
            {
                "q_scales": q_scales,
                "q_zeros": q_zeros,
                "q_g_idx": q_g_idx,
            },
        )
        del q_scales, q_zeros, q_g_idx

        with self.lock:
            self.durations.append(duration)
            if isinstance(avg_loss, (int, float)):
                self.avg_losses.append(avg_loss)
            self.module_names.append(f"layer-{module.layer_index}-{module.name}")

        if isinstance(avg_loss, str):
            loss_display = avg_loss
        else:
            loss_display = f"{avg_loss:.10f}" if isinstance(avg_loss, (int, float)) else "unknown"

        stat = {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_LOSS: loss_display,
            QUANT_LOG_NSAMPLES: f"{nsamples}",
            QUANT_LOG_DAMP: f"{damp_percent:.5f}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
            PROCESS_USED_MEMORY: self.device_memory_report(),
        }

        if workspace_summary:
            requests = int(workspace_summary.get("requests", 0) or 0)
            if requests:
                hit_rate = float(workspace_summary.get("hit_rate", 0.0) or 0.0)
                chunk_rows = workspace_summary.get("chunk_rows")
                stat["workspace_cache_requests"] = str(requests)
                stat["workspace_cache_hit_rate"] = f"{hit_rate:.1%}"
                stat["workspace_stage_dtype"] = workspace_summary.get("staging_dtype", "")
                if chunk_rows is not None:
                    stat["workspace_chunk_rows"] = str(chunk_rows)
        if workspace_totals:
            total_requests = int(workspace_totals.get("requests", 0) or 0)
            if total_requests:
                cumulative_hit_rate = (
                    float(workspace_totals.get("materialized_hits", 0) or 0.0) / total_requests
                )
                stat["workspace_total_requests"] = str(total_requests)
                stat["workspace_total_hit_rate"] = f"{cumulative_hit_rate:.1%}"

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        with self.lock:
            self.log.append(stat)

        self.log_new_row(stat)

        if self.calculate_w_wq_diff and not self._preserve_pseudo_weight_for_replay(weight_quant_extra):
            w_wq_diff = module.weight.data.to(dtype=torch.float32) - wq.to(dtype=torch.float32)
            with self.lock:
                module.state.update({"w_wq_diff": w_wq_diff})
                module.state.update({"wq": wq})

        if not self._preserve_pseudo_weight_for_replay(weight_quant_extra):
            module.weight.data = wq

        if weight_quant_extra is not None:
            with self.lock:
                module.state["ptq_weight_quant_extra"] = dict(weight_quant_extra)

    def _register_ptq_inference_hooks(
        self,
        module: NamedModule,
        qmodule: Module,
    ) -> None:
        transform = module.state.get(PTQ_TRANSFORM_KEY)
        if transform is None:
            return

        from ..ptq.inference_data import InferenceTransformData
        from ..ptq.inference_hooks import persist_ptq_inference_buffers, register_activation_pre_hook

        extra = module.state.pop("ptq_weight_quant_extra", None) or {}
        inference = transform.inference
        if inference is None:
            raw = extra.get("inference_transform")
            if isinstance(raw, InferenceTransformData):
                inference = raw
            elif isinstance(raw, dict):
                inference = InferenceTransformData.from_dict(raw)

        resolved = inference
        if resolved is None and transform is not None:
            from ..ptq.config import TransformPrepareConfig
            from ..ptq.transforms.registry import build_transform_backend

            backend = build_transform_backend(TransformPrepareConfig(method=transform.method))
            resolved = backend.get_inference_data(transform)

        if resolved is not None and not resolved.is_identity():
            pad = 0
            if transform.payload:
                pad = int(transform.payload.get("pad", 0) or 0)
            persist_ptq_inference_buffers(
                qmodule,
                resolved,
                method=transform.method,
                pad=pad,
            )

        register_activation_pre_hook(qmodule, transform, inference=inference)
        if resolved is not None and not resolved.is_identity():
            module.state["ptq_inference_transform"] = resolved.to_dict()
        module.state.pop(PTQ_TRANSFORM_KEY, None)

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        export_payload = getattr(self.qcfg, "weight_export", None)
        if export_payload:
            from ..ptq.config import ExportTargetConfig
            from ..ptq.context import WeightQuantState
            from ..ptq.export.gptq import GptqExport
            from ..ptq.export.registry import build_export_backend

            export_cfg = ExportTargetConfig(
                format=str(export_payload.get("format", "gptq")),
                impl=str(export_payload.get("impl", "default")),
                options={k: v for k, v in export_payload.items() if k not in {"format", "impl"}},
            )
            export_backend = build_export_backend(export_cfg)
            if not isinstance(export_backend, GptqExport):
                transform = module.state.get(PTQ_TRANSFORM_KEY)
                weight_quant = WeightQuantState(
                    q_scales=module.state.get("q_scales", torch.empty(0)),
                    q_zeros=module.state.get("q_zeros", torch.empty(0)),
                    q_g_idx=module.state.get("q_g_idx"),
                )
                export_backend.pack_module(
                    module_name=module.full_name,
                    submodule=module,
                    transform=transform,
                    weight_quant=weight_quant,
                    model=model,
                )
                module.state.pop(PTQ_TRANSFORM_KEY, None)
                return

        module.stream_sync()
        with self.lock:
            if self.calculate_w_wq_diff:
                module.weight.data = module.state.pop("wq").to(CPU)

            module.state.pop("w", None)
            module.state.pop("w_wq_diff", None)

            q_zeros = module.state.pop("q_zeros").clone()
            q_scales = module.state.pop("q_scales").clone()
            q_g_idx = module.state.pop("q_g_idx").clone()

        assert q_zeros.device == CPU
        assert q_scales.device == CPU
        assert q_g_idx.device == CPU

        layers = find_modules(model.model)
        module_label = getattr(module, "full_name", getattr(module, "name", ""))
        parent_key = getattr(module, "full_name", getattr(module, "name", None))

        timer = getattr(model, "quant_region_timer", None)

        create_start = time.perf_counter() if timer is not None else None
        with log_time_block(
            "create_quant_module",
            logger=log,
            module_name=module_label,
        ):
            with parent_module_lock(parent_key):
                create_quant_module(
                    name=module.full_name,
                    linear_cls=model.qlinear_kernel,
                    bits=self.qcfg.runtime_bits,
                    desc_act=self.qcfg.desc_act,
                    dynamic=self.qcfg.dynamic,
                    group_size=self.qcfg.group_size,
                    module=model.model,
                    submodule=module,
                    sym=self.qcfg.sym,
                    device=self.qcfg.device,
                    lm_head_name=model.lm_head,
                    pack_dtype=self.qcfg.pack_dtype,
                    format=resolve_quant_format(self.qcfg.format, self.qcfg.method),
                    register_buffers=False,
                )
        if timer is not None and create_start is not None:
            timer.record(
                "submodule_finalize_create",
                time.perf_counter() - create_start,
                source=module_label,
            )

        qModules = {
            name: submodule
            for name, submodule in find_modules(model.model, [model.qlinear_kernel]).items()
            if name == module.full_name
        }
        pack_start = time.perf_counter() if timer is not None else None
        with log_time_block(
            "pack",
            logger=log,
            module_name=module_label,
        ):
            with parent_module_lock(parent_key):
                packer_label = pack_module(
                    name=module.full_name,
                    qModules=qModules,
                    q_scales=q_scales,
                    q_zeros=q_zeros,
                    q_g_idx=q_g_idx,
                    layers=layers,
                    quant_linear_cls=model.qlinear_kernel,
                    lock=self.lock,
                    quantize_config=self.qcfg,
                )
        if timer is not None and pack_start is not None:
            timer.record(
                "submodule_finalize_pack",
                time.perf_counter() - pack_start,
                source=f"{module_label} [{packer_label or 'module.pack_original'}]",
            )

        qmodule = qModules.get(module.full_name)
        if qmodule is not None:
            self._register_ptq_inference_hooks(module, qmodule)

        with self.lock:
            self.result_pop(module.full_name)

        del q_scales, q_zeros, q_g_idx
        module.unregister_parameter("weight")

    def finalize(self, model: BaseQModel, **kwargs):
        model.quantized = True
        model.quantize_config.method = METHOD.GPTQ
        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        del processor_index
        return False

    def name(self) -> str:
        qcfg = self.qcfg_dynamic if self.qcfg_dynamic is not None else self.qcfg
        method = str(self.weight_quantize.method).strip().lower()
        if method == "rtn":
            return "rtn"
        if qcfg.gptaq is not None:
            return "gptaq"
        if qcfg.foem is not None:
            return "foem"
        return "gptq"

    def _release_host_buffers(self, *tensors: torch.Tensor) -> None:
        _ = tensors

    def has_captured_input_ids(self, name: str) -> bool:
        if name not in self._split_modules:
            return False
        collectors = getattr(self, "_collectors", None)
        if isinstance(collectors, dict) and collectors:
            by_short = getattr(self, "_collectors_by_short_name", None)
            if isinstance(by_short, dict):
                collector = by_short.get(name)
                if collector is not None:
                    return collector.nsamples > 0
            for key, collector in collectors.items():
                if key == name or key.endswith(f".{name}"):
                    return collector.nsamples > 0
            return False
        return True
