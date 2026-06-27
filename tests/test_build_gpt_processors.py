# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from gptqmodel.looper.processor_args import build_gpt_quantizer_processors
except (ImportError, ModuleNotFoundError, RuntimeError):
    pytest.skip("full gptqmodel import unavailable", allow_module_level=True)


def test_build_gpt_quantizer_processors_returns_sequential_ptq():
    qcfg = type("Cfg", (), {"uses_ptq_transform_pipeline": lambda self: True})()
    args = {
        "tokenizer": None,
        "qcfg": qcfg,
        "calibration": None,
        "prepare_dataset_func": None,
        "calibration_concat_size": None,
        "calibration_sort": None,
        "calibration_concat_separator": None,
        "batch_size": 1,
    }
    processors = build_gpt_quantizer_processors(qcfg, args, preprocessors=[])
    assert len(processors) == 1
    assert processors[0].__class__.__name__ == "SequentialPTQProcessor"
