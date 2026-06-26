# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""ParoQuant runtime export via ParoLinear packing."""

from __future__ import annotations

import threading

import torch.nn as nn

from ...looper.named_module import NamedModule
from ...models import BaseQModel
from ...nn_modules.qlinear.paroquant import ParoLinear
from ...quantization.config import FORMAT, METHOD, resolve_quant_format
from ...utils.model import create_quant_module, find_modules, move_to, pack_module
from ...utils.module_locks import parent_module_lock
from ..config import ExportTargetConfig
from ..context import TransformState, WeightQuantState
from ..inference_data import InferenceTransformData

_PACK_LOCK = threading.Lock()


class ParoQuantExport:
    def __init__(self, cfg: ExportTargetConfig) -> None:
        self.cfg = cfg

    def pack_module(
        self,
        *,
        module_name: str,
        submodule: nn.Module,
        transform: TransformState | None,
        weight_quant: WeightQuantState,
        inference: InferenceTransformData | None = None,
        model: nn.Module,
    ) -> None:
        del inference, weight_quant
        if transform is None or transform.method != "paroquant":
            raise ValueError("ParoQuantExport requires a paroquant TransformState.")
        if not isinstance(submodule, NamedModule):
            raise TypeError("ParoQuantExport expects a NamedModule wrapper.")
        if not isinstance(model, BaseQModel):
            raise TypeError("ParoQuantExport expects a BaseQModel instance.")

        payload = transform.payload
        pack_weight = payload["pack_weight"].clone()
        q_zeros = payload["q_zeros"].clone()
        q_scales = payload["q_scales"].clone()
        pairs = payload["pairs"].clone()
        theta = payload["theta"].clone()
        channel_scales = payload["channel_scales"].clone()

        submodule.weight.data = move_to(pack_weight, device=submodule.weight.device)
        qcfg = model.quantize_config
        format_code = resolve_quant_format(qcfg.format, qcfg.method)
        layers = find_modules(model.model)
        with parent_module_lock(submodule.full_name):
            create_quant_module(
                name=submodule.full_name,
                linear_cls=ParoLinear,
                bits=qcfg.runtime_bits,
                desc_act=qcfg.desc_act,
                dynamic=qcfg.dynamic,
                group_size=qcfg.group_size,
                module=model.model,
                submodule=submodule,
                sym=qcfg.sym,
                device=qcfg.device,
                lm_head_name=model.lm_head,
                pack_dtype=qcfg.pack_dtype,
                format=format_code,
                register_buffers=False,
                init_kwargs=qcfg.quant_linear_init_kwargs(),
            )

        qmodules = {
            name: mod
            for name, mod in find_modules(model.model, [ParoLinear]).items()
            if name == submodule.full_name
        }
        with parent_module_lock(submodule.full_name):
            pack_module(
                name=submodule.full_name,
                qModules=qmodules,
                q_scales=q_scales,
                q_zeros=q_zeros,
                q_g_idx=None,
                layers=layers,
                quant_linear_cls=ParoLinear,
                lock=_PACK_LOCK,
                quantize_config=qcfg,
            )

        qmodule = qmodules[submodule.full_name]
        if not isinstance(qmodule, ParoLinear):
            raise TypeError(
                f"Expected `{submodule.full_name}` to be packed as ParoLinear, got `{type(qmodule).__name__}`."
            )
        qmodule.pairs.copy_(pairs.to(device=qmodule.pairs.device, dtype=qmodule.pairs.dtype))
        qmodule.theta.copy_(theta.to(device=qmodule.theta.device, dtype=qmodule.theta.dtype))
        qmodule.channel_scales.copy_(
            channel_scales.to(device=qmodule.channel_scales.device, dtype=qmodule.channel_scales.dtype)
        )
        qmodule.post_init()
        model.quantize_config.method = METHOD.PARO
        model.quantize_config.format = FORMAT.PAROQUANT
