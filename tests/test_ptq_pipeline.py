# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from gptqmodel.ptq.config import ExportTargetConfig, TransformPrepareConfig, normalize_transform_prepare
from gptqmodel.ptq.context import ModuleCalibContext
from gptqmodel.ptq.export.fpquant import FpQuantExport
from gptqmodel.ptq.export.registry import build_export_backend
from gptqmodel.ptq.stats import StatisticsCollector
from gptqmodel.ptq.transforms.identity import IdentityTransform
from gptqmodel.ptq.transforms.registry import build_transform_backend
from gptqmodel.quantization.config import GPTQConfig, HessianConfig
from gptqmodel.quantization.qr_gptq_linalg import merge_qr_factors, update_qr_factor


def test_statistics_collector_qr_matches_stacked_batches():
    hessian = HessianConfig(factorization="qr", row_buffer_max_rows=64)
    collector = StatisticsCollector(columns=6, hessian=hessian, row_buffer_max_rows=32)
    batches = [torch.randn(12, 6), torch.randn(9, 6), torch.randn(15, 6)]
    for batch in batches:
        collector.add_batch(batch)
    _, qr_r = collector.finalize()
    assert qr_r is not None

    incremental = None
    for batch in batches:
        incremental = update_qr_factor(incremental, batch)
    assert torch.allclose(qr_r.T @ qr_r, incremental.T @ incremental, atol=1e-4, rtol=1e-4)


def test_statistics_collector_row_buffer_is_bounded():
    hessian = HessianConfig(row_buffer_max_rows=16)
    collector = StatisticsCollector(columns=4, hessian=hessian, row_buffer_max_rows=16)
    for _ in range(10):
        collector.add_batch(torch.randn(8, 4))
    collector.finalize()
    assert collector.row_buffer is not None
    assert collector.row_buffer.shape[0] <= 16


def test_merge_qr_factors_via_collector_devices():
    hessian = HessianConfig(factorization="qr")
    left = StatisticsCollector(columns=5, hessian=hessian)
    right = StatisticsCollector(columns=5, hessian=hessian)
    left.add_batch(torch.randn(10, 5))
    right.add_batch(torch.randn(7, 5))
    _, r_left = left.finalize()
    _, r_right = right.finalize()
    merged = merge_qr_factors(r_left, r_right)
    joint = torch.cat([torch.randn(10, 5), torch.randn(7, 5)], dim=0)
    _, expected = torch.linalg.qr(joint.to(torch.float32), mode="reduced")
    assert torch.allclose(merged.T @ merged, expected.T @ expected, atol=1e-3, rtol=1e-3)


def test_normalize_transform_prepare_accepts_dict_and_list():
    one = normalize_transform_prepare({"method": "paroquant", "opt_rotation_epochs": 2})
    assert len(one) == 1
    assert one[0].method == "paroquant"
    many = normalize_transform_prepare(
        [{"method": "identity"}, {"method": "paroquant", "mode": "standalone"}]
    )
    assert len(many) == 2


def test_identity_transform_is_noop():
    backend = build_transform_backend(TransformPrepareConfig(method="identity"))
    weight = torch.randn(8, 4)
    ctx = ModuleCalibContext(module_name="layer.0.linear", columns=4, rows=8)
    state = backend.fit(
        weight=weight,
        bias=None,
        ctx=ctx,
        mode="standalone",
        device=torch.device("cpu"),
    )
    baked = backend.apply_to_weights(weight, state, device=torch.device("cpu"))
    assert torch.equal(baked, weight)


def test_gptq_config_ptq_pipeline_fields():
    cfg = GPTQConfig(
        bits=4,
        group_size=128,
        weight_prepare=[{"method": "paroquant", "opt_rotation_epochs": 1, "opt_finetune_epochs": 0}],
        weight_export={"format": "gptq", "impl": "marlin"},
    )
    assert cfg.uses_ptq_transform_pipeline()
    assert cfg.weight_prepare[0].method == "paroquant"
    assert cfg.weight_export["impl"] == "marlin"


def test_fpquant_export_without_dependency_sets_meta():
    module = nn.Linear(4, 8, bias=False)
    export = FpQuantExport(ExportTargetConfig(format="fpquant", impl="qutlass", options={"pseudoquantization": True}))
    export.pack_module(
        module_name="layer.0.linear",
        submodule=module,
        transform=None,
        weight_quant=None,
        model=module,
    )
    assert getattr(module, "_ptq_fpquant_meta", None) is not None


def test_build_export_backend_defaults_to_gptq():
    from gptqmodel.ptq.export.gptq import GptqExport

    backend = build_export_backend(ExportTargetConfig(format="gptq", impl="default"))
    assert isinstance(backend, GptqExport)


def test_module_context_gram_from_qr():
    r = update_qr_factor(None, torch.randn(32, 6))
    ctx = ModuleCalibContext(module_name="m", columns=6, rows=8, qr_R=r)
    gram = ctx.gram_from_qr()
    assert gram.shape == (6, 6)
