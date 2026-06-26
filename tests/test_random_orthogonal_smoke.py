# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for random orthogonal transform + GPTQ/RTN PTQ pipeline."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.ptq.config import TransformPrepareConfig, WeightQuantizeTargetConfig
from gptqmodel.ptq.context import ModuleCalibContext, TransformState
from gptqmodel.ptq.pipeline import ModuleQuantizationPipeline
from gptqmodel.ptq.stats import StatisticsCollector
from gptqmodel.ptq.inference_data import InferenceTransformData
from gptqmodel.ptq.transforms.block_dense import apply_block_transform_to_activation
from gptqmodel.ptq.transforms.registry import build_transform_backend
from gptqmodel.quantization.config import ExpertsRoutingOverride, HessianConfig, MoEConfig

pytestmark = [pytest.mark.cpu, pytest.mark.slow]

_CALIBRATION_TEXTS = [
    "random orthogonal ptq calibration sample one with enough tokens for hessian capture",
    "random orthogonal ptq calibration sample two with repeated expert routing words",
    "random orthogonal ptq calibration sample three for stable positive definite hessian",
    "random orthogonal ptq calibration sample four covering all moe expert paths",
    "random orthogonal ptq calibration sample five activates every expert gate projection",
    "random orthogonal ptq calibration sample six provides additional activation rows",
] * 8


def _parse_loss(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).split(":")[-1].strip())



def _run_linear_pipeline(
    *,
    prepare_method: str,
    quantize_method: str,
    seed: int = 0,
) -> tuple[float, torch.Tensor]:
    torch.manual_seed(seed)
    group_size = 32
    linear = nn.Linear(64, 32, bias=False)
    qcfg = QuantizeConfig(bits=4, group_size=group_size)
    qcfg.hessian = HessianConfig(factorization="cholesky")
    collector = StatisticsCollector(columns=64, hessian=qcfg.hessian, row_buffer_max_rows=0)
    for _ in range(8):
        collector.add_batch(torch.randn(32, 64))
    ctx = collector.to_context(module_name="layer.0", rows=32)

    options = {"group_size": group_size, "opt_seed": 42}
    prepare = TransformPrepareConfig(method=prepare_method, options=options if prepare_method != "identity" else options)
    if prepare_method == "identity":
        prepare = TransformPrepareConfig(method="identity")

    pipeline = ModuleQuantizationPipeline(
        qcfg=qcfg,
        prepare_configs=[prepare],
        weight_quantize=WeightQuantizeTargetConfig(method=quantize_method),
    )
    result = pipeline.run_module(
        module=linear,
        collector=collector,
        ctx=ctx,
        device=torch.device("cpu"),
    )
    return _parse_loss(result.extra["loss"]), linear.weight.data.clone()


def _build_fixture(model_dir: Path) -> None:
    config = Qwen3MoeConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3MoeForCausalLM(config)
    model.save_pretrained(model_dir)

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )
    tokenizer.train_from_iterator(_CALIBRATION_TEXTS, trainer=trainer)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)


def _calibration(tokenizer: PreTrainedTokenizerFast) -> list[dict[str, object]]:
    dataset = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        dataset.append({"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]})
    return dataset


def test_random_orthogonal_bakes_nontrivial_transform():
    torch.manual_seed(0)
    weight = torch.randn(32, 64)
    backend = build_transform_backend(
        TransformPrepareConfig(method="random_orthogonal", options={"group_size": 32, "opt_seed": 7})
    )
    state = backend.fit(
        weight=weight,
        bias=None,
        ctx=ModuleCalibContext(module_name="layer.0", columns=64, rows=32),
        mode="standalone",
        device=torch.device("cpu"),
    )
    baked = backend.apply_to_weights(weight, state, device=torch.device("cpu"))
    assert not torch.allclose(baked, weight)
    inference = backend.get_inference_data(state)
    assert inference.T_X_matrices is not None
    assert inference.T_X_matrices.shape == (32, 32, 2)


def test_random_orthogonal_pipeline_configs_complete():
    configs = (
        ("identity", "gptq"),
        ("random_orthogonal", "rtn"),
        ("random_orthogonal", "gptq"),
    )
    losses = {}
    for prepare, quant in configs:
        loss, _ = _run_linear_pipeline(prepare_method=prepare, quantize_method=quant)
        losses[(prepare, quant)] = loss
        assert loss >= 0.0
    assert losses[("random_orthogonal", "gptq")] > 0.0


def test_tiny_qwen3_moe_random_orthogonal_gptq_smoke(tmp_path: Path):
    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    model_dir.mkdir(parents=True)
    _build_fixture(model_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    calibration = _calibration(tokenizer)

    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        damp_percent=0.05,
        damp_auto_increment=0.01,
        device="cpu",
        moe=MoEConfig(routing=ExpertsRoutingOverride()),
        weight_prepare=[
            {
                "method": "random_orthogonal",
                "group_size": 32,
                "opt_seed": 42,
            }
        ],
        weight_quantize={"method": "gptq"},
        weight_export={"format": "gptq"},
        hessian={"factorization": "cholesky", "row_buffer_max_rows": 256},
    )

    model = GPTQModel.load(str(model_dir), quantize_config=qcfg, backend=BACKEND.TORCH)
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)

    hooked = [
        (name, mod)
        for name, mod in model.named_modules()
        if hasattr(mod, "_ptq_inference_transform")
    ]
    assert hooked, "expected at least one module with a registered T_X hook"
    _, qmodule = hooked[0]
    assert isinstance(qmodule, TorchLinear)
    assert len(qmodule._forward_pre_hooks) > 0

    torch.manual_seed(123)
    in_features = qmodule.in_features
    x = torch.randn(4, in_features, dtype=torch.float32)
    payload = qmodule._ptq_inference_transform
    assert payload["transform_type"] == "dense"
    inference = InferenceTransformData.from_dict(payload)
    hook_state = TransformState(
        method="random_orthogonal",
        payload={
            "T_X_matrices": inference.T_X_matrices,
            "group_size": inference.block_size,
            "pad": 0,
            "inference_precision": "float16",
        },
    )
    hook = build_transform_backend(
        TransformPrepareConfig(method="random_orthogonal")
    ).activation_pre_hook(hook_state)
    x_ref = apply_block_transform_to_activation(
        x,
        inference.T_X_matrices,
        block_size=inference.block_size,
        pad=0,
        original_columns=x.shape[-1],
    )
    hook_args, _ = hook(qmodule, (x,), {}) or ((x,), {})
    x_hook = hook_args[0]
    assert torch.allclose(x_hook, x_ref, atol=1e-3, rtol=1e-3)

    model.save(quantized_dir)

    reloaded = GPTQModel.load(str(quantized_dir), backend=BACKEND.TORCH, device="cpu")
    modules = dict(reloaded.named_modules())
    for expert_index in range(4):
        for suffix in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.model.layers.0.mlp.experts.{expert_index}.{suffix}"
            assert isinstance(modules[name], TorchLinear), name

    hooked = [
        name
        for name, mod in modules.items()
        if hasattr(mod, "ptq_t_x_matrices")
        and isinstance(getattr(mod, "ptq_t_x_matrices"), torch.Tensor)
        and mod.ptq_t_x_matrices.numel() > 0
    ]
    assert hooked, "expected persistent T_X buffers after reload"
    for name in hooked:
        assert len(modules[name]._forward_pre_hooks) > 0, name
