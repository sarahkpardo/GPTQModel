# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


def _load_processor_args():
    root = Path(__file__).resolve().parents[1]
    path = root / "gptqmodel" / "looper" / "processor_args.py"
    spec = importlib.util.spec_from_file_location("processor_args", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_processor_kwargs_strips_quantizer_flags():
    mod = _load_processor_args()
    args = {
        "tokenizer": None,
        "qcfg": object(),
        "calibration": [],
        "prepare_dataset_func": None,
        "calibration_concat_size": None,
        "calibration_sort": None,
        "calibration_concat_separator": None,
        "batch_size": 2,
        "calculate_w_wq_diff": True,
        "require_fwd": False,
    }
    calib = mod.calibration_processor_kwargs(args)
    assert "calculate_w_wq_diff" not in calib
    assert "require_fwd" not in calib
    assert calib["batch_size"] == 2


def test_quantizer_processor_kwargs_keeps_quantizer_flags():
    mod = _load_processor_args()
    args = {
        "tokenizer": None,
        "qcfg": object(),
        "calibration": [],
        "prepare_dataset_func": None,
        "calibration_concat_size": None,
        "calibration_sort": None,
        "calibration_concat_separator": None,
        "batch_size": 1,
        "calculate_w_wq_diff": True,
        "require_fwd": False,
    }
    quant = mod.quantizer_processor_kwargs(args)
    assert quant["calculate_w_wq_diff"] is True
    assert quant["require_fwd"] is False


def test_native_processor_kwargs_strips_quantizer_flags():
    mod = _load_processor_args()
    args = {
        "tokenizer": None,
        "qcfg": object(),
        "calibration": [],
        "prepare_dataset_func": None,
        "calibration_concat_size": None,
        "calibration_sort": None,
        "calibration_concat_separator": None,
        "batch_size": 1,
        "calculate_w_wq_diff": True,
    }
    native = mod.native_processor_kwargs(args)
    assert "calculate_w_wq_diff" not in native
