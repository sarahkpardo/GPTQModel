# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from gptqmodel.ptq.config import WeightQuantizeTargetConfig
    from gptqmodel.ptq.context import ModuleCalibContext
    from gptqmodel.ptq.optimizers.gptq import GptqWeightOptimizer
    from gptqmodel.ptq.optimizers.registry import (
        build_weight_optimizer,
        weight_optimizer_requires_calibration,
    )
    from gptqmodel.ptq.optimizers.rtn import RtnWeightOptimizer
    from gptqmodel.ptq.stats import StatisticsCollector
    from gptqmodel.quantization.config import HessianConfig, QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ
except (ImportError, ModuleNotFoundError, RuntimeError):
    pytest.skip("full gptqmodel import unavailable", allow_module_level=True)


class _FakeQCfg:
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
    hessian = HessianConfig(factorization="cholesky", row_buffer_max_rows=64)


def test_weight_optimizer_requires_calibration_flags():
    assert weight_optimizer_requires_calibration(WeightQuantizeTargetConfig(method="gptq"))
    assert weight_optimizer_requires_calibration(WeightQuantizeTargetConfig(method="gptaq"))
    assert weight_optimizer_requires_calibration(WeightQuantizeTargetConfig(method="foem"))
    assert not weight_optimizer_requires_calibration(WeightQuantizeTargetConfig(method="rtn"))


def test_optimizer_requires_calibration_class_attrs():
    assert GptqWeightOptimizer.requires_calibration is True
    assert RtnWeightOptimizer.requires_calibration is False


def test_gptq_optimizer_requires_calibration():
    linear = nn.Linear(8, 4, bias=False)
    linear.full_name = "layer.0.linear"
    optimizer = GptqWeightOptimizer(qcfg=_FakeQCfg())
    empty_ctx = ModuleCalibContext(module_name="layer.0.linear", columns=8, rows=4)

    with pytest.raises(ValueError, match="requires calibration statistics"):
        optimizer.optimize(
            module=linear,
            ctx=empty_ctx,
            transform=None,
            device=torch.device("cpu"),
        )


def test_rtn_optimizer_no_calibration():
    linear = nn.Linear(8, 4, bias=False)
    optimizer = build_weight_optimizer(WeightQuantizeTargetConfig(method="rtn"), _FakeQCfg())
    empty_ctx = ModuleCalibContext(module_name="layer.0.linear", columns=8, rows=4)

    result = optimizer.optimize(
        module=linear,
        ctx=empty_ctx,
        transform=None,
        device=torch.device("cpu"),
    )
    assert result.pack_weight is not None
    assert result.q_scales is not None


def test_gptq_optimizer_injected_hessian_matches_inline_gptq():
    torch.manual_seed(0)
    in_features, out_features = 16, 8
    linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    linear.full_name = "layer.0.proj"

    batches = [torch.randn(12, in_features), torch.randn(9, in_features)]
    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        damp_percent=0.01,
        hessian=HessianConfig(factorization="cholesky", row_buffer_max_rows=64),
    )

    inline = GPTQ(linear, qcfg=qcfg)
    inline.fallback = None
    inline.quantizer.configure(perchannel=True)
    for batch in batches:
        out = F.linear(batch, linear.weight)
        inline.add_batch(batch, out)
    inline_w, inline_scales, inline_zeros, inline_g_idx, *_ = inline.quantize(blocksize=128)
    inline.free()

    torch.manual_seed(0)
    linear2 = nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    linear2.full_name = "layer.0.proj"
    collector = StatisticsCollector(
        columns=in_features,
        hessian=qcfg.hessian,
        row_buffer_max_rows=qcfg.hessian.row_buffer_max_rows,
    )
    for batch in batches:
        collector.add_batch(batch)
    ctx = collector.to_context(module_name=linear2.full_name, rows=out_features)
    collector.free()

    optimizer = GptqWeightOptimizer(qcfg=qcfg)
    split = optimizer.optimize(
        module=linear2,
        ctx=ctx,
        transform=None,
        device=torch.device("cpu"),
        qcfg=qcfg,
    )
    backend = split.extra.get("gptq")
    if backend is not None:
        backend.free()

    assert torch.equal(split.pack_weight, inline_w)
    assert torch.equal(split.q_scales, inline_scales)
    assert torch.equal(split.q_zeros, inline_zeros)
    assert torch.equal(split.q_g_idx, inline_g_idx)
