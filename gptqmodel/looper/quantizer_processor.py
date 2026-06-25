# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Generic weight quantizer processor with inline or split PTQ capture modes."""

from __future__ import annotations

import copy
import threading
import time
from typing import Callable, Dict, Literal, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import DTYPE_SIZE_COLUMN, ExecutionConfig, MODULE_FEATURE_COLUMN, LoopProcessor
from ..looper.named_module import NamedModule
from ..looper.statistics_processor import PTQ_CONTEXT_KEY
from ..looper.transform_processor import PTQ_TRANSFORM_KEY
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
from ..ptq.config import WeightQuantizeTargetConfig, resolve_weight_quantize_target
from ..ptq.optimizers.registry import build_weight_optimizer
from ..quantization import FOEM, GPTAQ, GPTQ
from ..quantization.config import FOEMConfig, GPTAQConfig, HessianConfig, METHOD, QuantizeConfig, resolve_quant_format
from ..utils.device import get_device
from ..utils.fallback import normalize_fallback
from ..utils.logger import log_time_block, setup_logger
from ..utils.model import create_quant_module, find_modules, pack_module
from ..utils.module_locks import parent_module_lock
from ..utils.torch import HAS_NPU

log = setup_logger()
lock = threading.Lock()

CaptureMode = Literal["none", "inline"]


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
        capture_mode: CaptureMode = "inline",
    ):
        self.capture_mode = capture_mode
        inline = capture_mode == "inline"
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
                require_fwd=require_fwd if inline else False,
                fwd_replay_after_process=inline,
                subset_forward_early_stop=inline,
            ),
        )

        self.optimizer = None if inline else build_weight_optimizer(target, qcfg)
        self.calculate_w_wq_diff = calculate_w_wq_diff
        self.avg_losses = []
        self.preserve_batch_keep_mask = True
        self._split_modules: Dict[str, bool] = {}

    def set_calibration_dataset(self, calibration_dataset):
        if self.capture_mode == "none":
            # PTQ split: statistics/transform stages own calibration; only inherit input cache.
            return
        raise NotImplementedError("QuantizerProcessor's calibration_dataset cannot be modified")

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

        if self.capture_mode == "none":
            self._split_modules[module.name] = True
            return

        if qcfg_clone.gptaq is not None:
            tmp = GPTAQ(module=module, qcfg=qcfg_clone)
        elif qcfg_clone.foem is not None:
            tmp = FOEM(module=module, qcfg=qcfg_clone)
        else:
            tmp = GPTQ(module=module, qcfg=qcfg_clone)
            tmp.fallback = qcfg_clone.fallback
            tmp.expected_nsamples = getattr(self, "total_calibration_tokens", None)

        tmp.quantizer.configure(perchannel=True)
        self.tasks[module.name] = tmp

    def is_skipped(self, module: NamedModule) -> bool:
        if self.capture_mode == "none":
            return module.name not in self._split_modules
        t = self.tasks.get(module.name, False)
        return t is False

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        if self.capture_mode == "none":

            def _noop(*_args, **_kwargs):
                return None

            return _noop

        def tmp(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            g = self.tasks[name]  # noqa: F821
            batch_idx = self.current_batch_index()
            inp_tensor = inp[0]
            keep_mask = getattr(getattr(self, "_mask_tls", None), "value", None)

            if (
                torch.is_tensor(inp_tensor)
                and torch.is_tensor(keep_mask)
                and inp_tensor.dim() >= 3
                and keep_mask.ndim == 2
                and keep_mask.shape[:2] == inp_tensor.shape[:2]
            ):
                out_tensor = out if torch.is_tensor(out) else None
                for sample_index, sample_keep in enumerate(keep_mask):
                    if not bool(sample_keep.any().item()):
                        continue

                    sample_inp = inp_tensor[sample_index : sample_index + 1, sample_keep, :].contiguous()
                    if out_tensor is not None and out_tensor.dim() >= 3 and out_tensor.shape[:2] == inp_tensor.shape[:2]:
                        sample_out = out_tensor[sample_index : sample_index + 1, sample_keep, :].contiguous()
                    else:
                        sample_out = out
                    g.add_batch(sample_inp.data, sample_out.data, batch_index=batch_idx)  # noqa: F821
            else:
                g.add_batch(inp_tensor.data, out.data, batch_index=batch_idx)  # noqa: F821
            del inp, out

        return tmp

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
        if self.capture_mode == "none":
            self._process_split(module, device=device)
        else:
            self._process_inline(module, device=device)

    def _process_split(self, module: NamedModule, device: torch.device = None):
        base_title = f"Quantizing {module.name} in layer"
        self.draw_progress(base_title)

        ctx = module.state.get(PTQ_CONTEXT_KEY)
        if ctx is None:
            raise ValueError(f"PTQ split pipeline missing calibration context for `{module.full_name}`.")

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
        )

    def _process_inline(self, module: NamedModule, device: torch.device = None):
        base_title = f"Quantizing {module.name} in layer"
        self.draw_progress(base_title)

        with self.lock:
            g = self.tasks[module.name]

        expected_device = getattr(module, "target_device", None)
        if expected_device is None:
            expected_device = getattr(module.module, "target_device", None)
        if expected_device is None:
            expected_device = get_device(module.module)

        if expected_device is not None:
            expected_device = torch.device(expected_device)

            module_weight = getattr(module.module, "weight", None)
            if module_weight is not None:
                assert module_weight.device == expected_device, (
                    f"Module '{module.full_name}' weight device {module_weight.device} does not match "
                    f"assigned target device {expected_device}."
                )
                assert module_weight.data.device == expected_device, (
                    f"Module '{module.full_name}' weight.data device {module_weight.data.device} does not match "
                    f"assigned target device {expected_device}."
                )

            g_module = getattr(g, "module", None)
            g_weight = getattr(g_module, "weight", None) if g_module is not None else None
            if g_weight is not None:
                assert g_weight.device == expected_device, (
                    f"GPTQ task for module '{module.full_name}' expected device {expected_device}, "
                    f"but found weight on {g_weight.device}."
                )
                assert g_weight.data.device == expected_device, (
                    f"GPTQ task for module '{module.full_name}' weight.data on {g_weight.data.device} "
                    f"does not match target device {expected_device}."
                )

            g_h = getattr(g, "H", None)
            if g_h is not None:
                assert torch.device(g_h.device) == expected_device, (
                    f"GPTQ Hessian tensor for '{module.full_name}' lives on {g_h.device}, expected {expected_device}."
                )

            if expected_device.type == "cuda" and torch.cuda.is_available():
                current_cuda_device = torch.device("cuda", torch.cuda.current_device())
                assert current_cuda_device == expected_device, (
                    f"CUDA thread context {current_cuda_device} does not match expected device {expected_device} "
                    f"while processing '{module.full_name}'."
                )
            if expected_device.type == "npu" and HAS_NPU:
                current_npu_device = torch.device("npu", torch.npu.current_device())
                assert current_npu_device == expected_device, (
                    f"NPU thread context {current_npu_device} does not match expected device {expected_device} "
                    f"while processing '{module.full_name}'."
                )

        wq, q_scales, q_zeros, q_g_idx, duration, avg_loss, damp_percent, nsamples = g.quantize()

        workspace_summary = getattr(g, "_borrow_workspace_last_summary", None)
        workspace_totals = getattr(g, "_borrow_workspace_totals", None)

        self._finalize_module_quant(
            module,
            wq=wq,
            q_scales=q_scales,
            q_zeros=q_zeros,
            q_g_idx=q_g_idx,
            duration=duration,
            avg_loss=avg_loss,
            damp_percent=damp_percent,
            nsamples=nsamples,
            workspace_summary=workspace_summary,
            workspace_totals=workspace_totals,
        )

        g.log_workspace_stats(context="gptq_process")

        with self.lock:
            self.tasks[module.name].free()

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

        if self.calculate_w_wq_diff:
            w_wq_diff = module.weight.data.to(dtype=torch.float32) - wq.to(dtype=torch.float32)
            with self.lock:
                module.state.update({"w_wq_diff": w_wq_diff})
                module.state.update({"wq": wq})

        module.weight.data = wq

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
        if self.capture_mode == "none":
            return False
        if self.calibration_dataset is None:
            raise ValueError("QuantizerProcessor's calibration_dataset must be provided.")
        return True

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
        if self.capture_mode == "none":
            return name in self._split_modules
        return self.tasks[name].fwd_counter > 0
