# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""MoE PTQ smoke: tiny Qwen3 MoE with SequentialPTQProcessor and identity prepare."""

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig

pytestmark = [pytest.mark.cpu, pytest.mark.slow]

_CALIBRATION_TEXTS = [
    "tiny moe ptq calibration sample one with enough tokens to survive minimum length filtering",
    "tiny moe ptq calibration sample two with repeated expert words for routing override",
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
    _build_local_tokenizer(model_dir)
    return config


def _calibration(tokenizer: PreTrainedTokenizerFast) -> list[dict[str, object]]:
    dataset = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        dataset.append({"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]})
    return dataset


def test_tiny_qwen3_moe_ptq_smoke(tmp_path: Path):
    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    model_dir.mkdir(parents=True)

    config = _build_fixture(model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    calibration = _calibration(tokenizer)

    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        moe=MoEConfig(routing=ExpertsRoutingOverride()),
        weight_prepare=[{"method": "identity"}],
        hessian={"factorization": "cholesky", "row_buffer_max_rows": 256},
    )

    model = GPTQModel.load(str(model_dir), quantize_config=qcfg, backend=BACKEND.TORCH)
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    model.save(quantized_dir)

    reloaded = GPTQModel.load(str(quantized_dir), backend=BACKEND.TORCH, device="cpu")
    modules = dict(reloaded.named_modules())
    for expert_index in range(config.num_experts):
        for suffix in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.model.layers.0.mlp.experts.{expert_index}.{suffix}"
            assert isinstance(modules[name], TorchLinear), name

    assert reloaded.quantize_config.meta_get("moe")["routing"]["class"] == "ExpertsRoutingOverride"
