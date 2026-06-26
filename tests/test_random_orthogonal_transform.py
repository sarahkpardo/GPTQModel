# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from gptqmodel.ptq.config import TransformPrepareConfig, WeightQuantizeTargetConfig
from gptqmodel.ptq.context import ModuleCalibContext
from gptqmodel.ptq.inference_data import InferenceTransformData
from gptqmodel.ptq.pipeline import ModuleQuantizationPipeline
from gptqmodel.ptq.stats import StatisticsCollector
from gptqmodel.ptq.transforms.block_dense import (
    apply_block_transform_to_activation,
    apply_block_transform_to_hessian,
    apply_block_transform_to_weight,
    generate_random_orthogonal_blocks,
    pad_columns,
)
from gptqmodel.ptq.transforms.random_orthogonal import RandomOrthogonalTransform
from gptqmodel.quantization.config import HessianConfig, QuantizeConfig


def _make_backend(*, block_size: int = 4, seed: int = 0) -> RandomOrthogonalTransform:
    return RandomOrthogonalTransform(
        TransformPrepareConfig(
            method="random_orthogonal",
            bake_weights=True,
            options={"group_size": block_size, "opt_seed": seed},
        )
    )


def _fit_state(*, in_features: int = 8, out_features: int = 4, block_size: int = 4, seed: int = 0):
    backend = _make_backend(block_size=block_size, seed=seed)
    weight = torch.randn(out_features, in_features)
    ctx = ModuleCalibContext(module_name="layer.0", columns=in_features, rows=out_features)
    state = backend.fit(
        weight=weight,
        bias=None,
        ctx=ctx,
        mode="standalone",
        device=torch.device("cpu"),
    )
    return backend, state, weight


def test_qr_blocks_are_orthogonal():
    t_w_blocks, _ = generate_random_orthogonal_blocks(4, 2, seed=123, device=torch.device("cpu"))
    for q in t_w_blocks:
        identity = torch.eye(4, dtype=q.dtype)
        assert torch.allclose(q @ q.T, identity, atol=1e-5, rtol=1e-5)
        assert torch.allclose(q.T @ q, identity, atol=1e-5, rtol=1e-5)


def test_bilinear_constraint():
    t_w_blocks, _ = generate_random_orthogonal_blocks(4, 2, seed=7, device=torch.device("cpu"))
    for q in t_w_blocks:
        t_x = q.T.float()
        assert torch.allclose(t_x @ q.float(), torch.eye(4), atol=1e-5, rtol=1e-5)


def test_weight_activation_preserve_matmul():
    backend, state, weight = _fit_state(in_features=8, out_features=4, block_size=4, seed=11)
    x = torch.randn(16, 8)
    baked = backend.apply_to_weights(weight, state, device=torch.device("cpu"))
    inference = backend.get_inference_data(state)
    x_tx = apply_block_transform_to_activation(
        x,
        torch.stack([block.T for block in state.payload["T_W_blocks"]], dim=2).float(),
        block_size=int(state.payload["group_size"]),
        pad=int(state.payload["pad"]),
    )
    baseline = x @ weight.T
    transformed = x_tx @ baked.T
    assert torch.allclose(transformed, baseline, atol=1e-5, rtol=1e-4)


def test_weight_activation_preserve_matmul_conv1d_layout():
    backend = _make_backend(block_size=4, seed=19)
    weight = torch.randn(8, 4)
    ctx = ModuleCalibContext(module_name="attn.c_attn", columns=8, rows=4)
    state = backend.fit(
        weight=weight,
        bias=None,
        ctx=ctx,
        mode="standalone",
        device=torch.device("cpu"),
    )
    assert state.payload["weight_layout"] == "conv1d"
    x = torch.randn(16, 8)
    baked = backend.apply_to_weights(weight, state, device=torch.device("cpu"))
    x_tx = apply_block_transform_to_activation(
        x,
        torch.stack([block.T for block in state.payload["T_W_blocks"]], dim=2).float(),
        block_size=int(state.payload["group_size"]),
        pad=int(state.payload["pad"]),
    )
    baseline = x @ weight
    transformed = x_tx @ baked
    assert torch.allclose(transformed, baseline, atol=1e-5, rtol=1e-4)


def test_hessian_congruence_preserves_spectrum():
    backend, state, _ = _fit_state(in_features=8, block_size=4, seed=3)
    torch.manual_seed(0)
    h = torch.randn(8, 8)
    h = h @ h.T
    h_prime = backend.transform_hessian(h.clone(), state)
    evals, _ = torch.linalg.eigh(h)
    evals_prime, _ = torch.linalg.eigh(h_prime)
    assert torch.allclose(torch.sort(evals).values, torch.sort(evals_prime).values, atol=1e-4, rtol=1e-4)


def test_inference_data_layout():
    backend, state, _ = _fit_state(in_features=8, block_size=4, seed=5)
    inference = backend.get_inference_data(state)
    assert isinstance(inference, InferenceTransformData)
    assert inference.transform_type == "dense"
    assert inference.T_X_matrices is not None
    assert inference.T_X_matrices.shape == (4, 4, 2)
    restored = InferenceTransformData.from_dict(inference.to_dict())
    assert restored.block_size == 4
    assert torch.allclose(restored.T_X_matrices, inference.T_X_matrices)


def test_activation_pre_hook():
    backend, state, weight = _fit_state(in_features=8, out_features=4, block_size=4, seed=17)
    baked = backend.apply_to_weights(weight, state, device=torch.device("cpu"))
    linear = nn.Linear(8, 4, bias=False)
    linear.weight.data = baked
    hook = backend.activation_pre_hook(state)
    linear.register_forward_pre_hook(hook, with_kwargs=True)
    x = torch.randn(5, 8)
    reference = apply_block_transform_to_activation(
        x,
        state.payload["T_X_matrices"],
        block_size=int(state.payload["group_size"]),
        pad=int(state.payload["pad"]),
    )
    out = linear(x)
    expected = reference @ baked.T
    assert torch.allclose(out, expected, atol=1e-4, rtol=1e-3)


def test_pipeline_random_orth_gptq():
    torch.manual_seed(2)
    linear = nn.Linear(8, 4, bias=False)
    qcfg = QuantizeConfig(bits=4, group_size=4)
    qcfg.hessian = HessianConfig(factorization="cholesky")
    collector = StatisticsCollector(columns=8, hessian=qcfg.hessian, row_buffer_max_rows=0)
    batch = torch.randn(64, 8)
    collector.add_batch(batch)
    ctx = collector.to_context(module_name="layer.0", rows=4)

    prepare = TransformPrepareConfig(
        method="random_orthogonal",
        options={"group_size": 4, "opt_seed": 42},
    )
    pipeline = ModuleQuantizationPipeline(
        qcfg=qcfg,
        prepare_configs=[prepare],
        weight_quantize=WeightQuantizeTargetConfig(method="gptq"),
    )
    result = pipeline.run_module(
        module=linear,
        collector=collector,
        ctx=ctx,
        device=torch.device("cpu"),
    )
    assert result.pack_weight is not None
    assert isinstance(result.extra.get("loss"), float)
    assert result.extra["loss"] >= 0.0
    assert ctx.transform is not None
    assert ctx.transform.method == "random_orthogonal"
    assert ctx.transform.inference is not None
    assert not ctx.transform.inference.is_identity()


def test_pipeline_random_orth_rtn():
    torch.manual_seed(3)
    linear = nn.Linear(8, 4, bias=False)
    qcfg = QuantizeConfig(bits=4, group_size=4)
    ctx = ModuleCalibContext(module_name="layer.0", columns=8, rows=4, nsamples=0)

    prepare = TransformPrepareConfig(
        method="random_orthogonal",
        options={"group_size": 4, "opt_seed": 99},
    )
    pipeline = ModuleQuantizationPipeline(
        qcfg=qcfg,
        prepare_configs=[prepare],
        weight_quantize=WeightQuantizeTargetConfig(method="rtn"),
    )
    result = pipeline.run_module(
        module=linear,
        collector=StatisticsCollector(columns=8, hessian=qcfg.hessian, row_buffer_max_rows=0),
        ctx=ctx,
        device=torch.device("cpu"),
    )
    assert result.pack_weight is not None
    loss = result.extra.get("loss")
    assert loss is not None
    if isinstance(loss, str):
        assert loss.startswith("rtn:")
    else:
        assert isinstance(loss, float)


def test_pad_columns_helper():
    padded, pad, blocks = pad_columns(10, 4)
    assert padded == 12
    assert pad == 2
    assert blocks == 3
