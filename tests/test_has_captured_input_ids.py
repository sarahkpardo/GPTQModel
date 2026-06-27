# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from gptqmodel.looper.quantizer_processor import QuantizerProcessor
except (ImportError, ModuleNotFoundError, RuntimeError):
    pytest.skip("full gptqmodel import unavailable", allow_module_level=True)


def test_split_has_captured_input_ids_missing_module_returns_false():
    processor = object.__new__(QuantizerProcessor)
    processor.capture_mode = "none"
    processor._split_modules = {}
    processor._collectors = {}
    assert processor.has_captured_input_ids("self_attn.q_proj") is False
