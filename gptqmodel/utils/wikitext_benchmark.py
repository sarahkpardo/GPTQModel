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
    text = "\n\n".join(data["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    tokens = tokens[:n_tokens]

    module = _unwrap_model(model)
    module.eval()
    total_nll = 0.0
    n_seqs = len(tokens) // seq_len
    if n_seqs == 0:
        return float("nan")

    for i in range(n_seqs):
        chunk = tokens[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = module(chunk, labels=chunk)
        total_nll += out.loss.item()

    return math.exp(total_nll / n_seqs)


def mean_quant_loss(quantize_result: dict[str, list[dict[str, str]]]) -> float | None:
    """Average numeric module ``loss`` values from ``GPTQModel.quantize()`` log output."""
    losses: list[float] = []
    for entries in quantize_result.values():
        for entry in entries:
            raw = entry.get("loss", "")
            if raw in {"", "unknown"}:
                continue
            try:
                losses.append(float(raw))
            except (TypeError, ValueError):
                continue
    if not losses:
        return None
    return sum(losses) / len(losses)
