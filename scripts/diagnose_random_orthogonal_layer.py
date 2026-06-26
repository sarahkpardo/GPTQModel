#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Diagnose random_orthogonal PTQ on a single module or full model.

Compares identity vs random_orthogonal on the same captured Hessian/activations,
reports coordinate-invariant output MSE, hook coverage, and optional activation drift.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require_runtime_deps() -> None:
    missing: list[str] = []
    for package in ("logbar", "pcre"):
        try:
            __import__(package)
        except ImportError:
            missing.append("pypcre" if package == "pcre" else package)
    if missing:
        raise SystemExit(
            "Missing runtime packages: "
            + ", ".join(missing)
            + ". Install GPTQModel first, e.g.\n"
            "  pip install -e .\n"
            "  python scripts/ensure_deps.py --skip-pip-check"
        )


_require_runtime_deps()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.ptq.stats import StatisticsCollector  # noqa: E402
from gptqmodel.quantization.gptq import get_number_of_rows_and_cols  # noqa: E402
from gptqmodel.utils.moe_benchmark import configure_moe_quantize_config, moe_quantize_load_kwargs  # noqa: E402
from gptqmodel.utils.random_orthogonal_diag import (  # noqa: E402
    LayerDiagResult,
    audit_ptq_hooks,
    measure_activation_drift,
    run_layer_pipeline,
    summarize_quant_log,
)
from gptqmodel.utils.wikitext_benchmark import load_wikitext_calibration  # noqa: E402


def _resolve_module(model: GPTQModel, module_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if module_name not in modules:
        candidates = [name for name in modules if name.endswith(module_name)]
        if len(candidates) == 1:
            module_name = candidates[0]
        elif candidates:
            raise SystemExit(
                f"Ambiguous module {module_name!r}; candidates: {candidates[:5]}"
            )
        else:
            raise SystemExit(f"Module {module_name!r} not found in model.")
    target = modules[module_name]
    if not isinstance(target, nn.Linear):
        raise SystemExit(f"Module {module_name!r} is {type(target)!r}, expected nn.Linear.")
    return target


def _run_calibration_forwards(model: GPTQModel, calibration, *, batch_size: int) -> None:
    device = next(model.model.parameters()).device
    model.model.eval()
    with torch.inference_mode():
        for index in range(0, len(calibration), batch_size):
            batch_samples = calibration[index : index + batch_size]
            for sample in batch_samples:
                if isinstance(sample, dict):
                    batch = {key: value.to(device) for key, value in sample.items()}
                else:
                    encoded = model.tokenizer(sample, return_tensors="pt")
                    batch = {key: value.to(device) for key, value in encoded.items()}
                model.model(**batch)


def _capture_module_context(
    model: GPTQModel,
    module: nn.Linear,
    module_name: str,
    calibration,
    *,
    batch_size: int,
    row_buffer_max_rows: int,
) -> tuple[object, StatisticsCollector]:
    _, columns = get_number_of_rows_and_cols(module)
    qcfg = model.quantize_config
    collector = StatisticsCollector(
        columns=columns,
        hessian=qcfg.hessian,
        row_buffer_max_rows=row_buffer_max_rows,
    )

    def _pre_hook(_mod, inp, _kwargs):
        if not inp:
            return
        tensor = inp[0] if isinstance(inp, tuple) else inp
        if isinstance(tensor, torch.Tensor):
            collector.add_batch(tensor.detach())

    handle = module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    try:
        _run_calibration_forwards(model, calibration, batch_size=batch_size)
    finally:
        handle.remove()

    rows, _ = get_number_of_rows_and_cols(module)
    ctx = collector.to_context(module_name=module_name, rows=rows)
    if ctx.nsamples <= 0 or ctx.H is None:
        raise RuntimeError(
            f"Failed to capture statistics for {module_name}: nsamples={ctx.nsamples}"
        )
    return ctx, collector


def _build_qcfg(
    *,
    bits: int,
    group_size: int,
    device: str,
    damp_percent: float,
    inference_precision: str,
    model_id: str,
    trust_remote_code: bool,
) -> QuantizeConfig:
    kwargs = dict(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        damp_percent=damp_percent,
        damp_auto_increment=0.01,
        device=device,
        hessian={"factorization": "cholesky", "row_buffer_max_rows": 512},
        weight_prepare=[{"method": "identity"}],
        weight_quantize={"method": "gptq"},
        weight_export={"format": "gptq"},
    )
    kwargs.update(moe_quantize_load_kwargs(model_id, trust_remote_code=trust_remote_code))
    return QuantizeConfig(**kwargs)


def _format_layer_result(result: LayerDiagResult) -> dict[str, object]:
    return {
        "weight_prepare": result.weight_prepare,
        "gptq_loss": result.gptq_loss,
        "damp": result.damp,
        "pre_quant_matmul_rel_err": result.pre_quant_matmul_rel_err,
        "post_quant_output_mse": result.post_quant_output_mse,
        "hessian_trace": result.hessian_trace,
        "hessian_trace_transformed": result.hessian_trace_transformed,
    }


def run_single_module_diag(
    *,
    model_id: str,
    module_name: str,
    calibration,
    bits: int,
    group_size: int,
    device: str,
    damp_percent: float,
    inference_precision: str,
    trust_remote_code: bool,
) -> dict[str, object]:
    load_kwargs = {
        "quantize_config": _build_qcfg(
            bits=bits,
            group_size=group_size,
            device=device,
            damp_percent=damp_percent,
            inference_precision=inference_precision,
            model_id=model_id,
            trust_remote_code=trust_remote_code,
        ),
        "backend": BACKEND.TORCH,
    }
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    model = GPTQModel.load(model_id, **load_kwargs)
    configure_moe_quantize_config(model, model.quantize_config)
    module = _resolve_module(model, module_name)
    full_name = next(name for name, mod in model.named_modules() if mod is module)

    ctx, collector = _capture_module_context(
        model,
        module,
        full_name,
        calibration,
        batch_size=1,
        row_buffer_max_rows=512,
    )

    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        damp_percent=damp_percent,
        damp_auto_increment=0.01,
        hessian={"factorization": "cholesky", "row_buffer_max_rows": 512},
    )

    identity_result, _, _ = run_layer_pipeline(
        module=module,
        ctx=ctx,
        qcfg=qcfg,
        weight_prepare="identity",
    )
    random_result, _, _ = run_layer_pipeline(
        module=module,
        ctx=ctx,
        qcfg=qcfg,
        weight_prepare="random_orthogonal",
        inference_precision=inference_precision,
    )
    collector.free()

    trace_delta = None
    if (
        identity_result.hessian_trace is not None
        and random_result.hessian_trace_transformed is not None
    ):
        trace_delta = abs(
            random_result.hessian_trace_transformed - identity_result.hessian_trace
        )

    return {
        "module": full_name,
        "nsamples": ctx.nsamples,
        "columns": ctx.columns,
        "identity": _format_layer_result(identity_result),
        "random_orthogonal": _format_layer_result(random_result),
        "hessian_trace_delta": trace_delta,
        "output_mse_ratio": (
            random_result.post_quant_output_mse / identity_result.post_quant_output_mse
            if identity_result.post_quant_output_mse > 0
            else None
        ),
    }


def run_full_model_audit(
    *,
    model_id: str,
    calibration,
    bits: int,
    group_size: int,
    device: str,
    damp_percent: float,
    inference_precision: str,
    trust_remote_code: bool,
    weight_prepare: str,
    probe_modules: Sequence[str],
    calib_concat_size: int,
) -> dict[str, object]:
    moe_kwargs = moe_quantize_load_kwargs(model_id, trust_remote_code=trust_remote_code)
    kwargs = dict(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        damp_percent=damp_percent,
        damp_auto_increment=0.01,
        device=device,
        hessian={"factorization": "cholesky", "row_buffer_max_rows": 512},
        weight_prepare=[{"method": weight_prepare}],
        weight_quantize={"method": "gptq"},
        weight_export={"format": "gptq"},
    )
    kwargs.update(moe_kwargs)
    if weight_prepare == "random_orthogonal":
        kwargs["weight_prepare"] = [
            {
                "method": "random_orthogonal",
                "group_size": group_size,
                "opt_seed": 42,
                "inference_precision": inference_precision,
            }
        ]

    load_kwargs = {"quantize_config": QuantizeConfig(**kwargs), "backend": BACKEND.TORCH}
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    fp16_load: dict[str, object] = {"backend": BACKEND.TORCH}
    if trust_remote_code:
        fp16_load["trust_remote_code"] = True
    fp16_reference = GPTQModel.load(model_id, **fp16_load)

    quant_model = GPTQModel.load(model_id, **load_kwargs)
    configure_moe_quantize_config(quant_model, quant_model.quantize_config)

    quantize_kwargs = {"batch_size": 1, "backend": BACKEND.TORCH, "calibration_data_min_length": 10}
    if calib_concat_size > 0:
        quantize_kwargs["calibration_concat_size"] = calib_concat_size
    quant_log = quant_model.quantize(calibration, **quantize_kwargs)
    log_summary = summarize_quant_log(quant_log, base_damp=damp_percent)

    work_dir = Path(tempfile.mkdtemp(prefix="rand-ortho-diag-"))
    checkpoint = work_dir / "quantized"
    quant_model.save(str(checkpoint))
    reloaded = GPTQModel.load(str(checkpoint), backend=BACKEND.TORCH, device=device)
    hook_audit = audit_ptq_hooks(reloaded.model)

    drift = {}
    if probe_modules and calibration:
        batch = calibration[0]
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
        else:
            input_ids = quant_model.tokenizer(batch, return_tensors="pt")["input_ids"]
        device_obj = next(reloaded.model.parameters()).device
        input_ids = input_ids.to(device_obj)
        drift = measure_activation_drift(
            fp16_reference.model,
            reloaded.model,
            input_ids,
            probe_module_names=probe_modules,
        )

    return {
        "weight_prepare": weight_prepare,
        "quant_log_summary": {
            "mean_loss": log_summary.mean_loss,
            "max_damp": log_summary.max_damp,
            "modules_with_elevated_damp": log_summary.modules_with_elevated_damp,
            "module_count": log_summary.module_count,
        },
        "hook_audit": {
            "modules_with_t_x_buffers": hook_audit.modules_with_t_x_buffers,
            "modules_with_active_hooks": hook_audit.modules_with_active_hooks,
            "padded_modules": hook_audit.modules_with_pad,
        },
        "activation_drift": drift,
        "checkpoint": str(checkpoint),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose random_orthogonal PTQ regressions.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--module-name", default=None, help="Single Linear module to compare.")
    parser.add_argument(
        "--mode",
        choices=("single_module", "full_audit", "both"),
        default="both",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calib-samples", type=int, default=32)
    parser.add_argument("--calib-concat-size", type=int, default=2048)
    parser.add_argument("--damp-percent", type=float, default=0.01)
    parser.add_argument(
        "--inference-precision",
        default="float16",
        choices=("float16", "float32", "bfloat16"),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--probe-modules",
        default="",
        help="Comma-separated module names for activation drift (full_audit).",
    )
    args = parser.parse_args()

    load_kwargs = {"backend": BACKEND.TORCH}
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    tokenizer_model = GPTQModel.load(args.model_id, **load_kwargs)
    calibration = load_wikitext_calibration(
        tokenizer_model.tokenizer,
        max_samples=args.calib_samples,
        min_length=10,
        concat_size=0,
    )
    del tokenizer_model

    module_name = args.module_name
    if module_name is None and args.mode in {"single_module", "both"}:
        module_name = "model.layers.0.self_attn.q_proj"

    probe_modules = [part.strip() for part in args.probe_modules.split(",") if part.strip()]
    if not probe_modules and args.mode in {"full_audit", "both"}:
        probe_modules = [
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
        ]

    report: dict[str, object] = {
        "model_id": args.model_id,
        "damp_percent": args.damp_percent,
        "inference_precision": args.inference_precision,
    }

    if args.mode in {"single_module", "both"}:
        assert module_name is not None
        print(f"Single-module diagnosis: {module_name}")
        single = run_single_module_diag(
            model_id=args.model_id,
            module_name=module_name,
            calibration=calibration,
            bits=args.bits,
            group_size=args.group_size,
            device=args.device,
            damp_percent=args.damp_percent,
            inference_precision=args.inference_precision,
            trust_remote_code=args.trust_remote_code,
        )
        report["single_module"] = single
        print(json.dumps(single, indent=2))

    for prepare in ("identity", "random_orthogonal"):
        if args.mode not in {"full_audit", "both"}:
            break
        print(f"\nFull-model audit ({prepare})")
        audit = run_full_model_audit(
            model_id=args.model_id,
            calibration=calibration,
            bits=args.bits,
            group_size=args.group_size,
            device=args.device,
            damp_percent=args.damp_percent,
            inference_precision=args.inference_precision,
            trust_remote_code=args.trust_remote_code,
            weight_prepare=prepare,
            probe_modules=probe_modules,
            calib_concat_size=args.calib_concat_size,
        )
        report[f"full_audit_{prepare}"] = audit
        print(json.dumps(audit, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nWrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
