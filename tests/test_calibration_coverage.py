# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest


def _load_calibration_coverage():
    root = Path(__file__).resolve().parents[1]
    path = root / "gptqmodel" / "ptq" / "calibration_coverage.py"
    spec = importlib.util.spec_from_file_location("calibration_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_calibration_tokens_from_processor():
    mod = _load_calibration_coverage()

    class Processor:
        total_calibration_tokens = 256

    assert mod.expected_calibration_tokens(Processor()) == 256


@pytest.mark.parametrize("total", [None, 0, -3])
def test_expected_calibration_tokens_raises_when_unavailable(total):
    mod = _load_calibration_coverage()

    class Processor:
        total_calibration_tokens = total

    with pytest.raises(ValueError, match="Calibration token count is unavailable"):
        mod.expected_calibration_tokens(Processor())


def test_describe_calibration_coverage_reports_sparse_routing():
    mod = _load_calibration_coverage()
    message = mod.describe_calibration_coverage(
        module_name="layer.0.mlp.experts.3.down_proj",
        observed_rows=12,
        expected_tokens=256,
    )
    assert "observed_rows=12" in message
    assert "expected_calibration_tokens=256" in message
    assert "coverage=" in message
    assert "MoE or sparse routing" in message


def test_describe_calibration_coverage_matches_dense_modules():
    mod = _load_calibration_coverage()
    message = mod.describe_calibration_coverage(
        module_name="layer.0.attn.c_proj",
        observed_rows=128,
        expected_tokens=128,
    )
    assert "matches" in message


def test_format_coverage_stat():
    mod = _load_calibration_coverage()
    assert mod.format_coverage_stat(observed_rows=32, expected_tokens=128) == "25.0%"
