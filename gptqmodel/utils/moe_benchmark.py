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


def resolve_moe_model_class(model_id: str, *, trust_remote_code: bool = False):
    from gptqmodel.models.auto import check_and_get_model_definition

    return check_and_get_model_definition(model_id, trust_remote_code=trust_remote_code)


def moe_quantize_load_kwargs(model_id: str, *, trust_remote_code: bool = False) -> dict[str, object]:
    """Return ``QuantizeConfig`` overrides needed for reliable MoE quantization."""
    model_cls = resolve_moe_model_class(model_id, trust_remote_code=trust_remote_code)
    if not getattr(model_cls, "dynamic_expert_index", None):
        return {}
    # Lazy turtle shells keep fused/meta expert blocks until layer materialization,
    # which breaks module discovery and MoE PTQ on small benchmark models.
    return {"offload_to_disk": False}


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


def _ensure_defused_moe_layout(gptq_model: GPTQModel) -> bool:
    import defuser
    from defuser.modeling.replace_modules import materialize_model
    from gptqmodel.models.loader import _convert_model_with_defuser

    model_type = getattr(gptq_model.model.config, "model_type", None)
    if model_type:
        defuser.replace_fused_blocks(model_type)
    converted = _convert_model_with_defuser(
        gptq_model.__class__,
        gptq_model.model,
        cleanup_original=False,
    )
    materialize_model(gptq_model.model)
    return bool(converted)


def _discover_layer_modules(gptq_model: GPTQModel) -> dict[str, Any]:
    from defuser.modeling.replace_modules import materialize_model
    from gptqmodel.utils.model import find_modules, get_layers_with_prefixes

    materialize_model(gptq_model.model)
    extract_layers = gptq_model.extract_layers_node()
    layers, _ = get_layers_with_prefixes(gptq_model.model, extract_layers)
    if not layers:
        return {}

    layer0 = layers[0]
    materialize_model(layer0)
    return find_modules(layer0)


def preflight_moe_expert_modules(gptq_model: GPTQModel) -> list[str]:
    """Return missing expert module names, or [] when layout looks usable."""
    if not is_moe_gptq_model(gptq_model):
        return []

    if getattr(gptq_model, "turtle_model", None) is not None:
        return []

    expected = _expected_quant_module_names(gptq_model)
    expert_expected = [name for name in expected if ".experts." in name]
    if not expert_expected:
        return []

    _ensure_defused_moe_layout(gptq_model)
    found = _discover_layer_modules(gptq_model)
    return [name for name in expert_expected if name not in found]


def configure_moe_quantize_config(gptq_model: GPTQModel, qcfg: QuantizeConfig) -> QuantizeConfig:
    """Apply MoE routing override and validate expert module layout when possible."""
    if not is_moe_gptq_model(gptq_model):
        return qcfg
    apply_moe_routing_override(qcfg)
    missing = preflight_moe_expert_modules(gptq_model)
    if missing:
        found = _discover_layer_modules(gptq_model)
        expert_found = sorted(key for key in found if "expert" in key.lower())
        mlp_found = sorted(key for key in found if key.startswith("mlp."))[:12]
        raise RuntimeError(
            "MoE expert modules from the model definition were not found after loading weights. "
            "GPTQModel expects defused per-expert linears such as "
            "`mlp.experts.0.gate_proj`. For MoE benchmarks use a supported Qwen MoE checkpoint, "
            "keep `offload_to_disk=False`, and ensure `defuser` is installed/up to date. "
            f"Missing sample: {missing[:6]}. "
            f"Found expert-related modules sample: {expert_found[:12]}. "
            f"Found mlp modules sample: {mlp_found}"
        )
    return qcfg
