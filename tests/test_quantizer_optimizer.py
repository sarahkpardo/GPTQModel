# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

try:
    from gptqmodel.ptq.config import WeightQuantizeTargetConfig, resolve_weight_quantize_target
    from gptqmodel.ptq.optimizers.registry import build_weight_optimizer
except (ImportError, ModuleNotFoundError, RuntimeError):
    pytest.skip("full gptqmodel import unavailable", allow_module_level=True)


def test_resolve_weight_quantize_target_defaults_to_gptq():
    class FakeCfg:
        weight_quantize = None
        gptaq = None
        foem = None

    target = resolve_weight_quantize_target(FakeCfg())
    assert target.method == "gptq"


def test_resolve_weight_quantize_target_accepts_dict():
    class FakeCfg:
        weight_quantize = {"method": "rtn"}

    target = resolve_weight_quantize_target(FakeCfg())
    assert target.method == "rtn"


def test_build_weight_optimizer_supports_rtn():
    class FakeCfg:
        bits = 4
        sym = True
        group_size = 128
        format = None
        device = None
        smooth = None
        damp_percent = 0.01
        desc_act = False
        dynamic = None
        pack_dtype = None

    target = WeightQuantizeTargetConfig(method="rtn")
    optimizer = build_weight_optimizer(target, FakeCfg())
    assert optimizer.__class__.__name__ == "RtnWeightOptimizer"
