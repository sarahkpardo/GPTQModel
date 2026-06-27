# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""WikiText-2 calibration and perplexity helpers for GPTQ benchmarking."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

WIKITEXT_DATASET = "salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"


def _load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for WikiText calibration and perplexity. "
            "Install with: pip install datasets"
        ) from exc
    return load_dataset


def _unwrap_model(model: Union[nn.Module, Any]) -> nn.Module:
    if isinstance(model, nn.Module):
        return model
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        return inner
    raise TypeError(f"Expected nn.Module or GPTQModel wrapper, got {type(model)!r}")


def _model_max_seq_len(module: nn.Module) -> int | None:
    config = getattr(module, "config", None)
    if config is None:
        return None
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _iter_wikitext_sequences(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    seq_len: int,
    max_seqs: int,
) -> list[torch.Tensor]:
    """Tokenize WikiText articles incrementally and pack fixed-length windows."""
    buf: list[int] = []
    seqs: list[torch.Tensor] = []
    for text in texts:
        if len(seqs) >= max_seqs:
            break
        stripped = text.strip()
        if not stripped:
            continue
        ids = tokenizer(
            stripped,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        ).input_ids[0].tolist()
        buf.extend(ids)
        while len(buf) >= seq_len and len(seqs) < max_seqs:
            seqs.append(torch.tensor(buf[:seq_len], dtype=torch.long))
            buf = buf[seq_len:]
    return seqs


def load_wikitext_calibration(
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_samples: int = 128,
    min_length: int = 10,
    concat_size: int = 0,
    split: str = "train",
) -> list[str] | list[dict[str, torch.Tensor]]:
    """Load WikiText-2 calibration samples from HuggingFace.

    When ``concat_size`` is 0, returns up to ``max_samples`` article strings whose
    tokenized length is at least ``min_length``.

    When ``concat_size`` > 0, concatenates token streams from articles and packs
    them into fixed-length chunks of ``concat_size`` tokens (returns pre-tokenized
    dicts with ``input_ids`` and ``attention_mask``).
    """
    load_dataset = _load_dataset()
    data = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split=split)

    if concat_size <= 0:
        samples: list[str] = []
        for text in data["text"]:
            if len(samples) >= max_samples:
                break
            stripped = text.strip()
            if not stripped:
                continue
            token_len = len(
                tokenizer(stripped, add_special_tokens=False, return_tensors="pt").input_ids[0]
            )
            if token_len < min_length:
                continue
            samples.append(stripped)
        return samples

    buf: list[int] = []
    packed: list[dict[str, torch.Tensor]] = []
    for text in data["text"]:
        if len(packed) >= max_samples:
            break
        stripped = text.strip()
        if not stripped:
            continue
        ids = tokenizer(
            stripped,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        ).input_ids[0].tolist()
        buf.extend(ids)
        while len(buf) >= concat_size and len(packed) < max_samples:
            chunk = torch.tensor(buf[:concat_size], dtype=torch.long)
            buf = buf[concat_size:]
            packed.append(
                {
                    "input_ids": chunk.unsqueeze(0),
                    "attention_mask": torch.ones(1, concat_size, dtype=torch.long),
                }
            )
    return packed


@torch.no_grad()
def compute_wikitext_perplexity(
    model: Union[nn.Module, Any],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    *,
    seq_len: int = 2048,
    n_tokens: int = 2048 * 32,
    split: str = "test",
) -> float:
    """Compute WikiText-2 split perplexity using sliding windows of ``seq_len`` tokens."""
    try:
        load_dataset = _load_dataset()
    except ImportError:
        return float("nan")

    data = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split=split)

    module = _unwrap_model(model)
    module.eval()

    effective_seq_len = seq_len
    max_model_len = _model_max_seq_len(module)
    if max_model_len is not None and effective_seq_len > max_model_len:
        effective_seq_len = max_model_len

    max_seqs = n_tokens // effective_seq_len
    if max_seqs <= 0:
        return float("nan")

    seqs = _iter_wikitext_sequences(
        tokenizer,
        data["text"],
        seq_len=effective_seq_len,
        max_seqs=max_seqs,
    )
    if not seqs:
        return float("nan")

    total_nll = 0.0
    for seq in seqs:
        chunk = seq.unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = module(chunk, labels=chunk)
        total_nll += out.loss.item()

    return math.exp(total_nll / len(seqs))


@torch.no_grad()
def compute_logits_relative_error(
    model: Union[nn.Module, Any],
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    reference_logits: torch.Tensor,
    device: torch.device,
) -> float:
    """Mean relative logits error vs a reference forward on the same prompt."""
    module = _unwrap_model(model)
    module.eval()
    batch = tokenizer(prompt, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits = module(**batch).logits
    ref = reference_logits.to(device=logits.device, dtype=logits.dtype)
    denom = ref.abs().mean().clamp(min=1e-6)
    return float((logits - ref).abs().mean().item() / denom.item())


def mean_quant_loss(quantize_result: dict[str, list[dict[str, str]]]) -> float | None:
    """Average numeric module ``loss`` values from ``GPTQModel.quantize()`` log output."""
    from .random_orthogonal_diag import summarize_quant_log

    return summarize_quant_log(quantize_result).mean_loss
