# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""MoE helpers for GPTQ benchmarking scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gptqmodel import GPTQModel, QuantizeConfig


def is_moe_gptq_model(model: Any) -> bool:
    """Return True when the GPTQModel class declares dynamic expert modules."""
    model_cls = model if isinstance(model, type) else model.__class__
    return getattr(model_cls, "dynamic_expert_index", None) is not None


def apply_moe_routing_override(qcfg: QuantizeConfig, *, enabled: bool = True) -> QuantizeConfig:
    """Attach ExpertsRoutingOverride when benchmarking MoE models."""
    if not enabled or qcfg.moe is not None:
        return qcfg

    from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig

    qcfg.moe = MoEConfig(routing=ExpertsRoutingOverride())
    return qcfg


def _expected_quant_module_names(gptq_model: GPTQModel) -> list[str]:
    qcfg = gptq_model.quantize_config
    if qcfg is None:
        return []
    layer_modules = gptq_model.simple_layer_modules(gptq_model.model.config, qcfg)
    names: list[str] = []
    for block in layer_modules:
        for name in block:
            if ":!" in name or ":?" in name:
                continue
            names.append(name.split(":!", 1)[0].split(":?", 1)[0])
    return names


def _discover_layer_modules(gptq_model: GPTQModel) -> dict[str, Any]:
    from defuser.modeling.replace_modules import materialize_model
    from gptqmodel.utils.model import find_modules, get_layers_with_prefixes

    extract_layers = gptq_model.extract_layers_node()
    layers, _ = get_layers_with_prefixes(gptq_model.model, extract_layers)
    if not layers:
        return {}

    layer0 = layers[0]
    materialize_model(layer0)
    return find_modules(layer0)


def _maybe_defuse_experts(gptq_model: GPTQModel) -> bool:
    import defuser

    converted = defuser.convert_model(gptq_model.model, cleanup_original=False)
    defuser_paths = getattr(gptq_model.__class__, "defuser_module_paths", ()) or ()
    for module_path in defuser_paths:
        from gptqmodel.utils.model import get_module_by_name

        try:
            module = get_module_by_name(gptq_model.model, module_path)
        except ValueError:
            continue
        converted = defuser.convert_model(module, cleanup_original=False) or converted
    return bool(converted)


def preflight_moe_expert_modules(gptq_model: GPTQModel) -> None:
    """Ensure defused expert linears exist before MoE PTQ begins."""
    if not is_moe_gptq_model(gptq_model):
        return

    expected = _expected_quant_module_names(gptq_model)
    expert_expected = [name for name in expected if ".experts." in name]
    if not expert_expected:
        return

    found = _discover_layer_modules(gptq_model)
    missing = [name for name in expert_expected if name not in found]
    if missing:
        _maybe_defuse_experts(gptq_model)
        found = _discover_layer_modules(gptq_model)
        missing = [name for name in expert_expected if name not in found]

    if not missing:
        return

    expert_found = sorted(key for key in found if "expert" in key.lower())
    raise RuntimeError(
        "MoE expert modules from the model definition were not found in the loaded checkpoint. "
        "GPTQModel expects defused per-expert linears such as "
        "`mlp.experts.0.gate_proj`. Ensure `defuser` is installed/up to date and pass "
        "`MoEConfig(routing=ExpertsRoutingOverride())` when quantizing MoE models. "
        f"Missing sample: {missing[:6]}. "
        f"Found expert-related modules sample: {expert_found[:12]}"
    )


def configure_moe_quantize_config(gptq_model: GPTQModel, qcfg: QuantizeConfig) -> QuantizeConfig:
    """Apply MoE routing override and validate expert module layout."""
    if not is_moe_gptq_model(gptq_model):
        return qcfg
    apply_moe_routing_override(qcfg)
    preflight_moe_expert_modules(gptq_model)
    return qcfg
