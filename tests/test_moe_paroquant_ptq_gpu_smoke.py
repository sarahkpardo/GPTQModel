# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke: tiny Qwen3 MoE PTQ with ParoQuant transform + ParoLinear export."""

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig
from gptqmodel.utils.paroquant import prewarm_paroquant_rotation_extension

pytestmark = [pytest.mark.cuda, pytest.mark.slow]

_CALIBRATION_TEXTS = [
    "tiny moe paroquant ptq calibration sample one with enough tokens to survive minimum length filtering",
    "tiny moe paroquant ptq calibration sample two with repeated expert words for routing override",
] * 2


def _build_local_tokenizer(model_dir: Path) -> PreTrainedTokenizerFast:
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


def _build_fixture(model_dir: Path) -> Qwen3MoeConfig:
    config = Qwen3MoeConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
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
    _build_local_tokenizer(model_dir)
    return config


def _calibration(tokenizer: PreTrainedTokenizerFast) -> list[dict[str, object]]:
    dataset = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        dataset.append({"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]})
    return dataset


def _paroquant_moe_qcfg() -> QuantizeConfig:
    return QuantizeConfig(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        device="cuda",
        moe=MoEConfig(routing=ExpertsRoutingOverride()),
        hessian={"factorization": "cholesky", "row_buffer_max_rows": 256},
        weight_prepare=[
            {
                "method": "paroquant",
                "opt_rotation_epochs": 1,
                "opt_finetune_epochs": 0,
                "krot": 2,
                "opt_fused_rotation": True,
                "opt_train_samples": 64,
                "opt_validation_samples": 16,
                "opt_batch_size": 16,
                "group_size": 32,
            }
        ],
        weight_quantize={"method": "paroquant"},
        weight_export={"format": "paroquant"},
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ParoQuant GPU smoke")
def test_tiny_qwen3_moe_paroquant_ptq_gpu_smoke(tmp_path: Path):
    prewarm_paroquant_rotation_extension(
        fused_rotation=True,
        group_size=32,
        krot=2,
        device="cuda",
    )

    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    model_dir.mkdir(parents=True)

    config = _build_fixture(model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    calibration = _calibration(tokenizer)
    qcfg = _paroquant_moe_qcfg()

    model = GPTQModel.load(str(model_dir), quantize_config=qcfg, backend=BACKEND.TORCH)
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    model.save(quantized_dir)

    reloaded = GPTQModel.load(
        str(quantized_dir),
        backend=BACKEND.PAROQUANT_CUDA,
        device="cuda",
    )
    modules = dict(reloaded.named_modules())

    attention_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")
    for suffix in attention_suffixes:
        name = f"model.model.layers.0.self_attn.{suffix}"
        assert isinstance(modules[name], ParoLinear), name

    expert_linear = None
    for expert_index in range(config.num_experts):
        for suffix in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.model.layers.0.mlp.experts.{expert_index}.{suffix}"
            module = modules[name]
            assert isinstance(module, ParoLinear), name
            if expert_index == 0 and suffix == "gate_proj":
                expert_linear = module

    assert expert_linear is not None
    assert expert_linear.pairs.numel() > 0
    assert expert_linear.theta.numel() > 0
    assert expert_linear.channel_scales.numel() > 0

    sample_input = torch.randn(1, 4, expert_linear.in_features, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        sample_output = expert_linear(sample_input)
    assert sample_output.shape == (1, 4, expert_linear.out_features)
    assert torch.isfinite(sample_output).all()

    prompt = "tiny moe paroquant"
    token_ids = reloaded.generate(
        prompt,
        max_new_tokens=8,
        do_sample=False,
        num_beams=1,
    )
    assert token_ids is not None
    assert len(token_ids[0]) > 0
    assert torch.isfinite(token_ids[0].to(dtype=torch.float32)).all()

    assert reloaded.quantize_config.meta_get("moe")["routing"]["class"] == "ExpertsRoutingOverride"
