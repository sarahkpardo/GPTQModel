# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shared linear/conv shape helpers for PTQ and legacy GPTQ."""

from __future__ import annotations

import numpy as np
import torch.nn as nn
import transformers

from ..looper.named_module import NamedModule


def get_number_of_rows_and_cols(layer: nn.Module) -> tuple[int, int]:
    if isinstance(layer, NamedModule):
        layer = layer.module

    if isinstance(layer, transformers.Conv1D):
        return layer.weight.shape[1], layer.weight.shape[0]
    return layer.weight.shape[0], np.prod(layer.weight.shape[1:])
