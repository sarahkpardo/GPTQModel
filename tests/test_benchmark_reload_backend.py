# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gptqmodel import BACKEND  # noqa: E402


def _load_benchmark_module():
    script_path = ROOT / "scripts" / "benchmark_random_orthogonal_ppl.py"
    spec = importlib.util.spec_from_file_location("benchmark_random_orthogonal_ppl", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ptq_backend_defaults_to_torch_on_cuda():
    bench = _load_benchmark_module()
    assert bench._resolve_ptq_backend("cuda") == BACKEND.TORCH
    assert bench._resolve_ptq_backend("cuda", legacy_auto_reload=True) is None
    assert bench._resolve_ptq_backend("cpu") == BACKEND.TORCH


@pytest.mark.parametrize(
    "device,legacy,expected",
    [
        ("cuda", False, BACKEND.TORCH),
        ("cuda", True, None),
        ("cpu", False, BACKEND.TORCH),
    ],
)
def test_ptq_backend_resolution(device, legacy, expected):
    bench = _load_benchmark_module()
    assert bench._resolve_ptq_backend(device, legacy_auto_reload=legacy) == expected
