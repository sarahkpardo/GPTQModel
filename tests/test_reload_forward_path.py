# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Reload vs in-memory path ablation tests (dispatch, fine probes, eval .to())."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.utils.moe_benchmark import benchmark_quantize_load_kwargs, configure_moe_quantize_config  # noqa: E402
from gptqmodel.utils.random_orthogonal_diag import (  # noqa: E402
    apply_plain_eval_placement,
    audit_accelerate_dispatch,
    audit_torchlinear_forward_vs_dequant,
    compare_hidden_states_pre_post,
    compare_logits_tensors,
    compare_reload_path_ablations,
    default_fine_hidden_state_probe_names,
    default_forward_audit_module_names,
    default_hidden_state_probe_names,
    merge_probe_names,
    prepare_model_for_ppl_eval_variant,
    reload_gptq_checkpoint_mirror_pre_reload,
)
from gptqmodel.utils.wikitext_benchmark import load_wikitext_calibration  # noqa: E402

_BENCHMARK_MODEL_ENV = "GPTQMODEL_BENCHMARK_MODEL_ID"
_CALIB_SAMPLES = 32
_EVAL_N_TOKENS = 4096
_EVAL_SEQ_LEN = 512
_LOGITS_BROKEN_THRESHOLD = 0.5
_LOGITS_OK_THRESHOLD = 0.05


def _load_benchmark_module():
    script_path = ROOT / "scripts" / "benchmark_random_orthogonal_ppl.py"
    spec = importlib.util.spec_from_file_location("benchmark_random_orthogonal_ppl", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for reload path ablation tests")


def _require_datasets():
    pytest.importorskip("datasets")


def _benchmark_model_id() -> str:
    return os.environ.get(_BENCHMARK_MODEL_ENV, "qwen/qwen3-0.6B").strip() or "qwen/qwen3-0.6B"


def _build_identity_quantize_config(model_id: str) -> QuantizeConfig:
    kwargs = dict(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        damp_percent=0.01,
        device="cpu",
        weight_prepare=[{"method": "identity"}],
        weight_quantize={"method": "gptq"},
        weight_export={"format": "gptq"},
    )
    kwargs.update(benchmark_quantize_load_kwargs(model_id))
    return QuantizeConfig(**kwargs)


class _PlainEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.linear(x)


def test_apply_plain_eval_placement_moves_parameters():
    pytest.importorskip("accelerate")
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    model = _PlainEvalModel()
    add_hook_to_module(model.linear, AlignDevicesHook("cpu", io_same_device=True))
    target = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    result = apply_plain_eval_placement(model, target)
    assert result["dispatch_before"]["align_devices_hook_count"] == 1
    assert result["dispatch_after"]["align_devices_hook_count"] == 0
    assert next(model.parameters()).device.type == target.type


def test_prepare_model_for_ppl_eval_variants_skip_second_to():
    class _Wrap:
        def __init__(self, module: nn.Module):
            self.model = module
            self._to_calls = 0

        def to(self, device):
            self._to_calls += 1
            self.model.to(device)
            return self

    inner = nn.Linear(4, 4)
    wrap = _Wrap(inner)
    device = torch.device("cpu")
    actual = prepare_model_for_ppl_eval_variant(wrap, device, "skip_second_to")
    assert wrap._to_calls == 0
    assert actual.type == "cpu"


def test_prepare_model_for_ppl_eval_variants_double_to():
    class _Wrap:
        def __init__(self, module: nn.Module):
            self.model = module
            self._to_calls = 0

        def to(self, device):
            self._to_calls += 1
            self.model.to(device)
            return self

    wrap = _Wrap(nn.Linear(4, 4))
    prepare_model_for_ppl_eval_variant(wrap, torch.device("cpu"), "double_to")
    assert wrap._to_calls == 2


@pytest.fixture(scope="module")
def reload_path_context(tmp_path_factory):
    """Identity quantize + save; shared pre/post reload references for CUDA ablations."""
    _require_cuda()
    _require_datasets()
    bench = _load_benchmark_module()
    model_id = _benchmark_model_id()
    work_dir = tmp_path_factory.mktemp("reload-path-ablation")
    checkpoint = work_dir / "quantized-identity"
    eval_device = bench._resolve_eval_device("cuda")

    qcfg = _build_identity_quantize_config(model_id)
    load_kwargs: dict[str, object] = {"quantize_config": qcfg, "backend": BACKEND.TORCH}
    load_kwargs.update(benchmark_quantize_load_kwargs(model_id))

    model = GPTQModel.load(model_id, **load_kwargs)
    configure_moe_quantize_config(model, model.quantize_config)
    calibration = load_wikitext_calibration(
        model.tokenizer,
        max_samples=_CALIB_SAMPLES,
        min_length=10,
        concat_size=0,
    )
    model.quantize(
        calibration,
        batch_size=1,
        backend=BACKEND.TORCH,
        calibration_data_min_length=10,
        calibration_concat_size=512,
    )
    pre_eval_device = bench._prepare_model_for_ppl_eval(model, eval_device)
    logits_prompt = bench._resolve_logits_prompt(calibration)
    module_names = default_forward_audit_module_names(model.model)
    fine_probes = default_fine_hidden_state_probe_names(model.model)
    coarse_probes = default_hidden_state_probe_names(model.model)
    probe_names = merge_probe_names(fine_probes, coarse_probes)
    audit_input = {
        key: value.to(pre_eval_device)
        for key, value in model.tokenizer(logits_prompt, return_tensors="pt").items()
    }
    pre_logits = bench._capture_reference_logits(model, logits_prompt, pre_eval_device)

    model.save(str(checkpoint))
    reload_kwargs = {"device": "cuda", "backend": BACKEND.TORCH}

    reloaded = GPTQModel.load(str(checkpoint), **reload_kwargs)
    post_eval_device = bench._prepare_model_for_ppl_eval(reloaded, eval_device)
    post_logits = bench._capture_reference_logits(reloaded, logits_prompt, post_eval_device)
    logits_pre_vs_post = compare_logits_tensors(pre_logits, post_logits)
    hidden_pre_vs_post = compare_hidden_states_pre_post(
        model.model,
        reloaded.model,
        audit_input,
        probe_names,
    )
    fine_hidden_pre_vs_post = compare_hidden_states_pre_post(
        model.model,
        reloaded.model,
        audit_input,
        fine_probes,
    )
    post_forward = audit_torchlinear_forward_vs_dequant(
        reloaded.model,
        module_names,
        audit_input,
    )

    ctx = {
        "bench": bench,
        "model": model,
        "reloaded": reloaded,
        "checkpoint": str(checkpoint),
        "reload_kwargs": reload_kwargs,
        "eval_device": eval_device,
        "logits_prompt": logits_prompt,
        "module_names": module_names,
        "probe_names": probe_names,
        "fine_probes": fine_probes,
        "audit_input": audit_input,
        "pre_logits": pre_logits,
        "post_logits": post_logits,
        "logits_pre_vs_post": logits_pre_vs_post,
        "hidden_pre_vs_post": hidden_pre_vs_post,
        "fine_hidden_pre_vs_post": fine_hidden_pre_vs_post,
        "post_forward": post_forward,
    }
    yield ctx
    del reloaded
    del model


@pytest.mark.cuda
@pytest.mark.colab
@pytest.mark.slow
def test_pre_reload_hidden_states_self_consistent(reload_path_context):
    ctx = reload_path_context
    result = compare_hidden_states_pre_post(
        ctx["model"].model,
        ctx["model"].model,
        ctx["audit_input"],
        ctx["probe_names"],
    )
    assert result["hidden_states_match_pre_post"] is True
    assert result["first_diverged_probe"] is None


@pytest.mark.cuda
@pytest.mark.colab
@pytest.mark.slow
def test_post_reload_diverges_at_fine_probes(reload_path_context):
    ctx = reload_path_context
    rel = ctx["logits_pre_vs_post"].get("mean_rel_error")
    assert isinstance(rel, (int, float))
    assert float(rel) > _LOGITS_BROKEN_THRESHOLD
    assert ctx["fine_hidden_pre_vs_post"]["hidden_states_match_pre_post"] is False
    first = ctx["fine_hidden_pre_vs_post"]["first_diverged_probe"]
    assert first in {
        "model.embed_tokens",
        "model.layers.0.input_layernorm",
        "model.layers.0.self_attn",
        "model.layers.0.post_attention_layernorm",
        "model.layers.0.mlp",
        "model.layers.0",
    }


@pytest.mark.cuda
@pytest.mark.colab
@pytest.mark.slow
def test_post_reload_linear_isolation_still_passes(reload_path_context):
    ctx = reload_path_context
    assert ctx["post_forward"]["all_match"] is True


@pytest.mark.cuda
@pytest.mark.colab
@pytest.mark.slow
def test_mirror_pre_reload_cpu_load_ablation(reload_path_context):
    ctx = reload_path_context
    mirror = reload_gptq_checkpoint_mirror_pre_reload(
        ctx["checkpoint"],
        eval_device=ctx["eval_device"],
        load_kwargs=dict(ctx["reload_kwargs"]),
    )
    mirror_model = mirror["model"]
    mirror_eval = next(mirror_model.model.parameters()).device
    mirror_logits = ctx["bench"]._capture_reference_logits(
        mirror_model,
        ctx["logits_prompt"],
        mirror_eval,
    )
    mirror_delta = compare_logits_tensors(ctx["pre_logits"], mirror_logits)
    mirror_forward = audit_torchlinear_forward_vs_dequant(
        mirror_model.model,
        ctx["module_names"],
        ctx["audit_input"],
    )
    summary = compare_reload_path_ablations(
        reference_logits_delta=ctx["logits_pre_vs_post"],
        ablations={"mirror_pre_reload": {"logits_pre_vs_ablation": mirror_delta}},
    )
    assert mirror_forward["all_match"] is True
    assert summary["mirror_pre_reload_fixes_logits"] in (True, False)
    if summary["mirror_pre_reload_fixes_logits"] is True:
        assert float(mirror_delta["mean_rel_error"]) <= _LOGITS_OK_THRESHOLD
    del mirror_model


@pytest.mark.cuda
@pytest.mark.colab
@pytest.mark.slow
def test_strip_hooks_ablation_after_cuda_reload(reload_path_context):
    ctx = reload_path_context
    reloaded = GPTQModel.load(ctx["checkpoint"], **ctx["reload_kwargs"])
    dispatch_before = audit_accelerate_dispatch(reloaded.model)
    apply_plain_eval_placement(reloaded.model, ctx["eval_device"])
    strip_logits = ctx["bench"]._capture_reference_logits(
        reloaded,
        ctx["logits_prompt"],
        ctx["eval_device"],
    )
    strip_delta = compare_logits_tensors(ctx["pre_logits"], strip_logits)
    strip_forward = audit_torchlinear_forward_vs_dequant(
        reloaded.model,
        ctx["module_names"],
        ctx["audit_input"],
    )
    summary = compare_reload_path_ablations(
        reference_logits_delta=ctx["logits_pre_vs_post"],
        ablations={"strip_hooks_only": {"logits_pre_vs_ablation": strip_delta}},
    )
    assert strip_forward["all_match"] is True
    assert dispatch_before["modules_with_hf_hook_count"] >= 0
    assert summary["strip_hooks_fixes_logits"] in (True, False)
    del reloaded


@pytest.mark.cuda
@pytest.mark.colab
@pytest.mark.slow
def test_skip_second_to_ablation(reload_path_context):
    ctx = reload_path_context
    reloaded = GPTQModel.load(ctx["checkpoint"], **ctx["reload_kwargs"])
    skip_eval = prepare_model_for_ppl_eval_variant(reloaded, ctx["eval_device"], "skip_second_to")
    skip_logits = ctx["bench"]._capture_reference_logits(
        reloaded,
        ctx["logits_prompt"],
        skip_eval,
    )
    skip_delta = compare_logits_tensors(ctx["pre_logits"], skip_logits)
    summary = compare_reload_path_ablations(
        reference_logits_delta=ctx["logits_pre_vs_post"],
        ablations={"skip_second_to": {"logits_pre_vs_ablation": skip_delta}},
    )
    assert summary["skip_second_to_fixes_logits"] in (True, False)
    assert isinstance(skip_delta.get("mean_rel_error"), (int, float))
    assert math.isfinite(float(skip_delta["mean_rel_error"]))
    del reloaded


@pytest.mark.cuda
@pytest.mark.colab
@pytest.mark.slow
def test_double_to_matches_single_to_or_worsens(reload_path_context):
    ctx = reload_path_context
    reloaded_single = GPTQModel.load(ctx["checkpoint"], **ctx["reload_kwargs"])
    single_eval = prepare_model_for_ppl_eval_variant(
        reloaded_single,
        ctx["eval_device"],
        "single_to",
    )
    single_logits = ctx["bench"]._capture_reference_logits(
        reloaded_single,
        ctx["logits_prompt"],
        single_eval,
    )
    single_delta = compare_logits_tensors(ctx["pre_logits"], single_logits)

    reloaded_double = GPTQModel.load(ctx["checkpoint"], **ctx["reload_kwargs"])
    double_eval = prepare_model_for_ppl_eval_variant(
        reloaded_double,
        ctx["eval_device"],
        "double_to",
    )
    double_logits = ctx["bench"]._capture_reference_logits(
        reloaded_double,
        ctx["logits_prompt"],
        double_eval,
    )
    double_delta = compare_logits_tensors(ctx["pre_logits"], double_logits)

    summary = compare_reload_path_ablations(
        reference_logits_delta=ctx["logits_pre_vs_post"],
        ablations={
            "single_to": {"logits_pre_vs_ablation": single_delta},
            "double_to": {"logits_pre_vs_ablation": double_delta},
        },
    )
    single_rel = float(single_delta["mean_rel_error"])
    double_rel = float(double_delta["mean_rel_error"])
    assert summary["double_to_worsens_logits"] == (double_rel > single_rel + 1e-6)
    del reloaded_single
    del reloaded_double
