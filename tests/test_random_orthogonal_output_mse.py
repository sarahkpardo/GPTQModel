# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Output-space MSE regression tests for random_orthogonal vs identity GPTQ."""

from __future__ import annotations

import torch
import torch.nn as nn

from gptqmodel.ptq.stats import StatisticsCollector
from gptqmodel.quantization.config import HessianConfig, QuantizeConfig
from gptqmodel.utils.random_orthogonal_diag import (
    pre_quant_matmul_rel_error,
    run_layer_pipeline,
    summarize_quant_log,
)


def _capture_context(*, in_features: int, out_features: int, batches: int = 8):
    qcfg = QuantizeConfig(bits=4, group_size=32, damp_percent=0.01)
    qcfg.hessian = HessianConfig(factorization="cholesky")
    collector = StatisticsCollector(columns=in_features, hessian=qcfg.hessian, row_buffer_max_rows=512)
    for _ in range(batches):
        collector.add_batch(torch.randn(32, in_features))
    return collector.to_context(module_name="layer.test", rows=out_features), qcfg


def test_pre_quant_matmul_invariant_on_captured_activations():
    in_features = 64
    out_features = 32
    ctx, qcfg = _capture_context(in_features=in_features, out_features=out_features)
    linear = nn.Linear(in_features, out_features, bias=False)
    torch.manual_seed(0)
    linear.weight.data = torch.randn(out_features, in_features)

    result, _, transform_state = run_layer_pipeline(
        module=linear,
        ctx=ctx,
        qcfg=qcfg,
        weight_prepare="random_orthogonal",
        opt_seed=7,
    )
    assert transform_state is not None
    payload = transform_state.payload
    from gptqmodel.ptq.config import TransformPrepareConfig
    from gptqmodel.ptq.transforms.random_orthogonal import RandomOrthogonalTransform

    backend = RandomOrthogonalTransform(
        TransformPrepareConfig(method="random_orthogonal", options={"group_size": 32})
    )
    baked = backend.apply_to_weights(linear.weight.data, transform_state, device=torch.device("cpu"))
    x = ctx.row_buffer if ctx.row_buffer is not None else torch.randn(64, in_features)
    rel_err = pre_quant_matmul_rel_error(
        x=x,
        weight=linear.weight.data,
        baked=baked,
        t_w_blocks=payload["T_W_blocks"],
        block_size=int(payload["group_size"]),
        pad=int(payload.get("pad", 0)),
        weight_layout=str(payload.get("weight_layout", "linear")),
    )
    assert rel_err < 1e-3
    assert result.pre_quant_matmul_rel_err < 1e-3


def test_random_orthogonal_output_mse_not_much_worse_than_identity():
    in_features = 64
    out_features = 32
    ctx, qcfg = _capture_context(in_features=in_features, out_features=out_features)
    linear = nn.Linear(in_features, out_features, bias=False)
    torch.manual_seed(1)
    linear.weight.data = torch.randn(out_features, in_features)

    identity, _, _ = run_layer_pipeline(
        module=linear,
        ctx=ctx,
        qcfg=qcfg,
        weight_prepare="identity",
    )
    random, _, _ = run_layer_pipeline(
        module=linear,
        ctx=ctx,
        qcfg=qcfg,
        weight_prepare="random_orthogonal",
        opt_seed=42,
    )

    assert identity.post_quant_output_mse > 0.0
    ratio = random.post_quant_output_mse / identity.post_quant_output_mse
    assert ratio < 1.5, (
        f"random_orthogonal output MSE {random.post_quant_output_mse:.6f} vs "
        f"identity {identity.post_quant_output_mse:.6f} (ratio={ratio:.3f})"
    )


def test_summarize_quant_log_skips_statistics_rows():
    quant_log = {
        "0": [
            {"process": "statistics", "module": "fc1", "samples": "128"},
            {"process": "gptq", "module": "fc1", "loss": "0.001", "damp": "0.01000", "samples": "128"},
        ]
    }
    summary = summarize_quant_log(quant_log, base_damp=0.01)
    assert summary.module_count == 1
    assert summary.mean_loss == 0.001
    assert summary.max_damp == 0.01
    assert summary.modules_with_elevated_damp == 0
