# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Mathematical parity tests for random orthogonal transform + GPTQ PTQ."""

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
from gptqmodel.ptq.config import TransformPrepareConfig
from gptqmodel.ptq.context import ModuleCalibContext
from gptqmodel.ptq.inference_data import InferenceTransformData
from gptqmodel.ptq.inference_hooks import rehydrate_ptq_inference_hooks
from gptqmodel.ptq.solvers.hessian import compute_hessian_inverse
from gptqmodel.ptq.transforms.block_dense import apply_block_transform_to_activation
from gptqmodel.ptq.transforms.random_orthogonal import RandomOrthogonalTransform
from gptqmodel.ptq.transforms.registry import build_transform_backend
from gptqmodel.quantization.config import ExpertsRoutingOverride, HessianConfig, MoEConfig

pytestmark = [pytest.mark.cpu, pytest.mark.slow]

_CALIBRATION_TEXTS = [
    "random orthogonal parity calibration sample one with enough tokens for hessian capture",
    "random orthogonal parity calibration sample two with repeated expert routing words",
    "random orthogonal parity calibration sample three for stable positive definite hessian",
    "random orthogonal parity calibration sample four covering all moe expert paths",
    "random orthogonal parity calibration sample five activates every expert gate projection",
    "random orthogonal parity calibration sample six provides additional activation rows",
] * 8


def _make_backend(*, block_size: int = 32, seed: int = 42) -> RandomOrthogonalTransform:
    return RandomOrthogonalTransform(
        TransformPrepareConfig(
            method="random_orthogonal",
            bake_weights=True,
            options={"group_size": block_size, "opt_seed": seed},
        )
    )


def _fit_state(
    *,
    in_features: int = 64,
    out_features: int = 32,
    block_size: int = 32,
    seed: int = 42,
    weight_layout: str = "linear",
):
    backend = _make_backend(block_size=block_size, seed=seed)
    if weight_layout == "conv1d":
        weight = torch.randn(in_features, out_features)
    else:
        weight = torch.randn(out_features, in_features)
    ctx = ModuleCalibContext(module_name="layer.0", columns=in_features, rows=out_features)
    state = backend.fit(
        weight=weight,
        bias=None,
        ctx=ctx,
        mode="standalone",
        device=torch.device("cpu"),
    )
    return backend, state, weight


def test_layer_matmul_invariant_conv1d_and_linear():
    for layout in ("linear", "conv1d"):
        backend, state, weight = _fit_state(
            in_features=64,
            out_features=32,
            block_size=32,
            seed=11,
            weight_layout=layout,
        )
        x = torch.randn(16, 64)
        baked = backend.apply_to_weights(weight, state, device=torch.device("cpu"))
        t_x_float = torch.stack(
            [block.T.float() for block in state.payload["T_W_blocks"]],
            dim=2,
        )
        x_tx = apply_block_transform_to_activation(
            x,
            t_x_float,
            block_size=int(state.payload["group_size"]),
            pad=int(state.payload["pad"]),
        )
        if layout == "conv1d":
            baseline = x @ weight
            transformed = x_tx @ baked
        else:
            baseline = x @ weight.T
            transformed = x_tx @ baked.T
        assert torch.allclose(transformed, baseline, atol=1e-5, rtol=1e-4)


def test_transformed_hessian_pd_recoverable():
    backend, state, _ = _fit_state(in_features=64, block_size=32, seed=3)
    torch.manual_seed(0)
    h = torch.randn(64, 64)
    h = h @ h.T + 0.05 * torch.eye(64)
    h_prime = backend.transform_hessian(h.clone(), state)
    qcfg = QuantizeConfig(damp_percent=0.01, damp_auto_increment=0.01)
    hinv, damp = compute_hessian_inverse(
        h_prime,
        qcfg=qcfg,
        nsamples=512,
        module_name="parity.test",
    )
    assert hinv is not None
    assert damp < 1.0


def test_low_rank_hessian_fails_without_fallback():
    backend, state, _ = _fit_state(in_features=64, block_size=32, seed=5)
    torch.manual_seed(1)
    h = torch.randn(64, 64)
    h = 0.5 * (h + h.T)
    h = h - 3.0 * torch.eye(64) * h.abs().max()
    h_prime = backend.transform_hessian(h.clone(), state)
    qcfg = QuantizeConfig(damp_percent=0.01, damp_auto_increment=0.01)
    hinv, damp = compute_hessian_inverse(
        h_prime,
        qcfg=qcfg,
        nsamples=64,
        module_name="parity.indefinite",
    )
    assert hinv is None
    assert damp == 1.0


def _build_moe_fixture(model_dir: Path) -> PreTrainedTokenizerFast:
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
    return fast_tokenizer


def _calibration(tokenizer: PreTrainedTokenizerFast) -> list[dict[str, object]]:
    dataset = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        dataset.append({"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]})
    return dataset


def _random_orthogonal_moe_qcfg() -> QuantizeConfig:
    return QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        damp_percent=0.01,
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


def _reference_logits(model: Qwen3MoeForCausalLM, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(**batch).logits


def _assert_hooks_present(modules: dict[str, nn.Module]) -> list[str]:
    hooked = [
        name
        for name, mod in modules.items()
        if hasattr(mod, "ptq_t_x_matrices")
        and isinstance(getattr(mod, "ptq_t_x_matrices"), torch.Tensor)
        and mod.ptq_t_x_matrices.numel() > 0
    ]
    assert hooked, "expected persistent T_X buffers on at least one module"
    for name in hooked:
        mod = modules[name]
        assert len(mod._forward_pre_hooks) > 0, name
        assert hasattr(mod, "_ptq_inference_transform"), name
    return hooked


def test_hessian_full_congruence_preserves_pd():
    backend, state, _ = _fit_state(in_features=64, block_size=32, seed=7)
    torch.manual_seed(0)
    h = torch.randn(64, 64)
    h = h @ h.T + 0.05 * torch.eye(64)
    h_prime = backend.transform_hessian(h.clone(), state)
    evals = torch.linalg.eigvalsh(h_prime)
    assert evals.min().item() >= -1e-4


def test_load_ptq_inference_buffers_from_safetensors(tmp_path: Path):
    from safetensors.torch import save_file

    linear = nn.Linear(64, 32, bias=False)
    inference = InferenceTransformData(
        transform_type="dense",
        T_X_matrices=torch.randn(32, 32, 2),
        block_size=32,
        precision=torch.float16,
        extra={"pad": 0},
    )
    from gptqmodel.ptq.inference_hooks import (
        load_ptq_inference_buffers_from_checkpoint,
        persist_ptq_inference_buffers,
    )

    persist_ptq_inference_buffers(linear, inference, method="random_orthogonal", pad=0)
    prefix = "layer0"
    save_file(
        {
            f"{prefix}.ptq_t_x_matrices": linear.ptq_t_x_matrices,
            f"{prefix}.ptq_t_x_block_size": linear.ptq_t_x_block_size,
            f"{prefix}.ptq_t_x_pad": linear.ptq_t_x_pad,
            f"{prefix}.ptq_transform_method_bytes": linear.ptq_transform_method_bytes,
        },
        str(tmp_path / "model.safetensors"),
    )
    target = nn.Module()
    target.add_module("layer0", nn.Linear(64, 32, bias=False))
    loaded = load_ptq_inference_buffers_from_checkpoint(target, str(tmp_path))
    assert loaded == 1
    assert target.layer0.ptq_t_x_matrices.shape == (32, 32, 2)


@pytest.mark.colab
@pytest.mark.slow
def test_tiny_moe_random_orthogonal_pre_reload_hooks(tmp_path: Path):
    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    model_dir.mkdir(parents=True)
    tokenizer = _build_moe_fixture(model_dir)
    calibration = _calibration(tokenizer)

    qcfg = _random_orthogonal_moe_qcfg()
    model = GPTQModel.load(str(model_dir), quantize_config=qcfg, backend=BACKEND.TORCH)
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)

    modules = dict(model.named_modules())
    _assert_hooks_present(modules)
    model.save(quantized_dir)


@pytest.mark.colab
@pytest.mark.slow
def test_tiny_moe_random_orthogonal_post_reload(tmp_path: Path):
    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    model_dir.mkdir(parents=True)
    tokenizer = _build_moe_fixture(model_dir)
    calibration = _calibration(tokenizer)

    reference = Qwen3MoeForCausalLM.from_pretrained(model_dir)
    batch = calibration[0]
    ref_logits = _reference_logits(reference, batch)
    del reference

    qcfg = _random_orthogonal_moe_qcfg()
    model = GPTQModel.load(str(model_dir), quantize_config=qcfg, backend=BACKEND.TORCH)
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    model.save(quantized_dir)
    del model

    reloaded = GPTQModel.load(str(quantized_dir), backend=BACKEND.TORCH, device="cpu")
    modules = dict(reloaded.named_modules())
    _assert_hooks_present(modules)

    for expert_index in range(4):
        for suffix in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.model.layers.0.mlp.experts.{expert_index}.{suffix}"
            assert isinstance(modules[name], TorchLinear), name

    reloaded.eval()
    with torch.no_grad():
        post_reload_logits = reloaded.model(**batch).logits

    assert torch.isfinite(post_reload_logits).all()

    rel_err = (post_reload_logits - ref_logits).abs().mean() / ref_logits.abs().mean().clamp(min=1e-6)
    assert rel_err.item() < 0.5, f"post-reload logits diverged from fp32: rel_err={rel_err.item():.4f}"


def test_rehydrate_ptq_inference_hooks_from_buffers():
    inference = InferenceTransformData(
        transform_type="dense",
        T_X_matrices=torch.randn(32, 32, 2),
        block_size=32,
        precision=torch.float16,
        extra={"pad": 0},
    )
    linear = nn.Linear(64, 32, bias=False)
    from gptqmodel.ptq.inference_hooks import persist_ptq_inference_buffers, register_activation_pre_hook

    persist_ptq_inference_buffers(linear, inference, method="random_orthogonal", pad=0)
    register_activation_pre_hook(linear, None, inference=inference)
    assert len(linear._forward_pre_hooks) == 1

    linear._forward_pre_hooks.clear()
    del linear._ptq_inference_transform

    restored = rehydrate_ptq_inference_hooks(nn.ModuleList([linear]))
    assert restored == 1
    assert len(linear._forward_pre_hooks) == 1

    x = torch.randn(4, 64)
    hook = next(iter(linear._forward_pre_hooks.values()))
    hook_args, _ = hook(linear, (x,), {}) or ((x,), {})
    x_hook = hook_args[0]
    x_tx = apply_block_transform_to_activation(
        x,
        inference.T_X_matrices.to(dtype=torch.float16),
        block_size=32,
        pad=0,
    )
    assert torch.allclose(x_hook, x_tx, atol=5e-3, rtol=5e-3)
