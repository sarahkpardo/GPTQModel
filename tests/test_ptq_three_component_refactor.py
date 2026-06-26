# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from gptqmodel.ptq.context import LayerStatistics, ModuleCalibContext
from gptqmodel.ptq.inference_data import InferenceTransformData
from gptqmodel.ptq.pipeline import ModuleQuantizationPipeline
from gptqmodel.ptq.stats import StatisticsCollector
from gptqmodel.ptq.transforms.identity import IdentityTransform
from gptqmodel.quantization.config import HessianConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def test_layer_statistics_alias():
    ctx = ModuleCalibContext(module_name="m", columns=4, rows=8)
    assert LayerStatistics is ModuleCalibContext
    assert ctx.block_M_W == []
    assert ctx.H_inv is None


def test_gptq_and_statistics_collector_share_hessian():
    torch.manual_seed(0)
    linear = nn.Linear(8, 4, bias=False)
    qcfg = QuantizeConfig()
    qcfg.hessian = HessianConfig(factorization="cholesky")

    batches = [torch.randn(16, 8), torch.randn(12, 8)]
    collector = StatisticsCollector(columns=8, hessian=qcfg.hessian, row_buffer_max_rows=0)
    for batch in batches:
        collector.add_batch(batch)
    collector.finalize()
    stats_h = collector.H

    gptq = GPTQ(linear, qcfg=qcfg)
    for batch in batches:
        gptq.add_batch(batch, torch.zeros(1))
    gptq.finalize_hessian()
    legacy_h = gptq.H

    assert stats_h is not None and legacy_h is not None
    assert torch.allclose(stats_h, legacy_h, atol=1e-5, rtol=1e-5)


def test_identity_transform_hessian_and_inference_data():
    backend = IdentityTransform()
    H = torch.eye(4)
    state = backend.fit(
        weight=torch.randn(6, 4),
        bias=None,
        ctx=ModuleCalibContext(module_name="m", columns=4, rows=6, H=H),
        mode="standalone",
        device=torch.device("cpu"),
    )
    assert torch.allclose(backend.transform_hessian(H, state), H)
    inference = backend.get_inference_data(state)
    assert isinstance(inference, InferenceTransformData)
    assert inference.is_identity()


def test_inference_transform_round_trip():
    payload = InferenceTransformData(
        transform_type="givens",
        T_X_angles=torch.tensor([0.1, 0.2]),
        precision=torch.float16,
        block_size=8,
    ).to_dict()
    restored = InferenceTransformData.from_dict(payload)
    assert restored.transform_type == "givens"
    assert restored.block_size == 8
    assert torch.allclose(restored.T_X_angles, payload["T_X_angles"])


def test_module_quantization_pipeline_identity_transform():
    torch.manual_seed(1)
    linear = nn.Linear(6, 4, bias=False)
    qcfg = QuantizeConfig(bits=4, group_size=4)
    collector = StatisticsCollector(columns=6, hessian=qcfg.hessian, row_buffer_max_rows=0)
    batch = torch.randn(32, 6)
    collector.add_batch(batch)
    ctx = collector.to_context(module_name="layer.0", rows=4)

    pipeline = ModuleQuantizationPipeline(qcfg=qcfg, prepare_configs=[])
    weight, state = pipeline.apply_transform(
        module=linear,
        weight=linear.weight.data,
        bias=None,
        ctx=ctx,
        device=torch.device("cpu"),
    )
    assert state.method == "identity"
    assert torch.allclose(weight, linear.weight.data)
