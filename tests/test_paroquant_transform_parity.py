# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.ptq.config import paro_prepare_from_config
from gptqmodel.ptq.transforms.paroquant import build_paroquant_optimize_kwargs, module_seed_from_options
from gptqmodel.ptq.transforms.paroquant_parity import (
    compare_paroquant_payloads,
    make_calibration_context,
    make_paroquant_prepare_config,
    make_synthetic_linear,
    payload_from_state,
    run_paroquant_reference,
    run_paroquant_transform,
)
from gptqmodel.quantization.config import ParoConfig


def test_transform_matches_direct_optimizer():
    linear, inputs = make_synthetic_linear(in_features=16, out_features=8, seed=0)
    cfg = make_paroquant_prepare_config(
        options={
            "opt_rotation_epochs": 1,
            "opt_finetune_epochs": 0,
            "krot": 2,
            "opt_train_samples": 128,
            "opt_validation_samples": 32,
            "opt_batch_size": 32,
        }
    )
    ctx = make_calibration_context(
        module_name="layer.0.linear",
        columns=linear.in_features,
        rows=linear.out_features,
        inputs=inputs,
    )
    reference = run_paroquant_reference(
        weight=linear.weight.data,
        bias=None,
        ctx=ctx,
        cfg=cfg,
    )
    state = run_paroquant_transform(
        weight=linear.weight.data,
        bias=None,
        ctx=ctx,
        cfg=cfg,
    )
    ok, diffs = compare_paroquant_payloads(reference, payload_from_state(state))
    assert ok, diffs


def test_transform_matches_legacy_module_seed():
    linear, inputs = make_synthetic_linear(in_features=12, out_features=6, seed=1)
    module_name = "model.layers.2.mlp.down_proj"
    cfg = make_paroquant_prepare_config(
        options={
            "opt_seed": 7,
            "opt_layer_index": 2,
            "opt_module_seed_key": module_name,
            "opt_scope": "module",
            "opt_rotation_epochs": 1,
            "opt_finetune_epochs": 0,
            "krot": 2,
        }
    )
    ctx = make_calibration_context(
        module_name=module_name,
        columns=linear.in_features,
        rows=linear.out_features,
        inputs=inputs,
    )
    expected_seed = module_seed_from_options(module_name=module_name, options=cfg.options)
    kwargs = build_paroquant_optimize_kwargs(
        weight=linear.weight.data,
        bias=None,
        ctx=ctx,
        cfg=cfg,
    )
    assert kwargs["seed"] == expected_seed

    reference = run_paroquant_reference(
        weight=linear.weight.data,
        bias=None,
        ctx=ctx,
        cfg=cfg,
    )
    state = run_paroquant_transform(
        weight=linear.weight.data,
        bias=None,
        ctx=ctx,
        cfg=cfg,
    )
    ok, diffs = compare_paroquant_payloads(reference, payload_from_state(state))
    assert ok, diffs


def test_paro_prepare_from_config_roundtrip():
    paro = ParoConfig(
        opt_rotation_epochs=3,
        opt_finetune_epochs=0,
        opt_train_samples=256,
        opt_validation_samples=32,
        opt_seed=11,
        krot=4,
    )
    cfg = paro_prepare_from_config(paro, mode="standalone")
    assert cfg.method == "paroquant"
    assert cfg.options["opt_rotation_epochs"] == 3
    assert cfg.options["opt_train_samples"] == 256
    assert cfg.options["opt_seed"] == 11
    assert cfg.options["opt_weight_decay"] == paro.opt_weight_decay
    assert cfg.options["opt_betas"] == paro.opt_betas


def test_row_buffer_vs_explicit_inputs():
    linear, inputs = make_synthetic_linear(in_features=12, out_features=5, seed=2)
    cfg = make_paroquant_prepare_config(
        options={
            "opt_rotation_epochs": 1,
            "opt_finetune_epochs": 0,
            "krot": 2,
            "opt_train_samples": 96,
            "opt_validation_samples": 24,
        }
    )
    ctx = make_calibration_context(
        module_name="layer.0.fc",
        columns=linear.in_features,
        rows=linear.out_features,
        inputs=inputs,
    )
    reference = run_paroquant_reference(
        weight=linear.weight.data,
        bias=None,
        ctx=ctx,
        cfg=cfg,
    )
    state = run_paroquant_transform(
        weight=linear.weight.data,
        bias=None,
        ctx=ctx,
        cfg=cfg,
    )
    ok, diffs = compare_paroquant_payloads(reference, payload_from_state(state))
    assert ok, diffs
    assert ctx.row_buffer is not None
    assert ctx.row_buffer.shape[0] == inputs.shape[0]
