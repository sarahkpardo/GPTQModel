#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Benchmark prepare × quantize PTQ combinations on WikiText-2 perplexity."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.utils.moe_benchmark import (  # noqa: E402
    benchmark_quantize_load_kwargs,
    configure_moe_quantize_config,
    is_moe_gptq_model,
)
from gptqmodel.utils.random_orthogonal_diag import (  # noqa: E402
    audit_ptq_hooks,
    summarize_quant_log,
)
from gptqmodel.utils.wikitext_benchmark import (  # noqa: E402
    compute_wikitext_perplexity,
    load_wikitext_calibration,
)

WeightPrepareMode = Literal["identity", "random_orthogonal", "paroquant"]
WeightQuantizeMode = Literal["gptq", "rtn", "paroquant"]
WeightExportMode = Literal["gptq", "paroquant"]


@dataclass(frozen=True)
class CombinationSpec:
    weight_prepare: WeightPrepareMode
    weight_quantize: WeightQuantizeMode
    weight_export: WeightExportMode
    label: str
    skip_reason: str | None = None


DEFAULT_COMBINATIONS: tuple[CombinationSpec, ...] = (
    CombinationSpec("identity", "gptq", "gptq", "identity + gptq"),
    CombinationSpec("random_orthogonal", "gptq", "gptq", "random_orthogonal + gptq"),
    CombinationSpec("paroquant", "gptq", "gptq", "paroquant + gptq"),
    CombinationSpec("identity", "rtn", "gptq", "identity + rtn"),
    CombinationSpec("paroquant", "paroquant", "paroquant", "paroquant + paroquant"),
    CombinationSpec(
        "random_orthogonal",
        "rtn",
        "gptq",
        "random_orthogonal + rtn",
    ),
    CombinationSpec(
        "wush",
        "rtn",
        "gptq",
        "wush + rtn",
        skip_reason="WUSH transform is not vendored yet (see WUSH/ and ptq/transforms/wush.py).",
    ),
)


def _build_quantize_config(
    *,
    spec: CombinationSpec,
    bits: int,
    group_size: int,
    device: str,
    damp_percent: float,
    inference_precision: str,
    paro_rotation_epochs: int,
    paro_finetune_epochs: int,
    extra_kwargs: dict[str, object] | None = None,
) -> QuantizeConfig:
    prepare_options: dict[str, object] = {"method": spec.weight_prepare}
    if spec.weight_prepare == "random_orthogonal":
        prepare_options.update(
            {
                "group_size": group_size,
                "opt_seed": 42,
                "inference_precision": inference_precision,
            }
        )
    elif spec.weight_prepare == "paroquant":
        prepare_options.update(
            {
                "krot": 8,
                "group_size": group_size,
                "opt_rotation_epochs": paro_rotation_epochs,
                "opt_finetune_epochs": paro_finetune_epochs,
                "opt_seed": 42,
            }
        )

    kwargs: dict[str, object] = dict(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        damp_percent=damp_percent,
        damp_auto_increment=0.01,
        device=device,
        hessian={"factorization": "cholesky", "row_buffer_max_rows": 512},
        weight_prepare=[prepare_options],
        weight_quantize={"method": spec.weight_quantize},
        weight_export={"format": spec.weight_export},
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return QuantizeConfig(**kwargs)


def _resolve_backend(device: str, *, weight_export: WeightExportMode):
    if weight_export == "paroquant" and device.startswith("cuda"):
        return BACKEND.PAROQUANT_CUDA
    if device == "cpu":
        return BACKEND.TORCH
    return None


def _count_ptq_hooks(model: GPTQModel) -> int:
    count = 0
    for module in model.model.modules():
        t_x = getattr(module, "ptq_t_x_matrices", None)
        if isinstance(t_x, torch.Tensor) and t_x.numel() > 0 and module._forward_pre_hooks:
            count += 1
    return count


def _run_combination(
    *,
    model_id: str,
    spec: CombinationSpec,
    calibration,
    bits: int,
    group_size: int,
    device: str,
    batch_size: int,
    calib_concat_size: int,
    eval_seq_len: int,
    eval_n_tokens: int,
    work_dir: Path,
    trust_remote_code: bool,
    damp_percent: float,
    inference_precision: str,
    paro_rotation_epochs: int,
    paro_finetune_epochs: int,
) -> dict[str, object]:
    if spec.skip_reason:
        return {
            "method": spec.label,
            "skipped": True,
            "skip_reason": spec.skip_reason,
        }

    moe_load_kwargs = benchmark_quantize_load_kwargs(model_id, trust_remote_code=trust_remote_code)
    qcfg = _build_quantize_config(
        spec=spec,
        bits=bits,
        group_size=group_size,
        device=device,
        damp_percent=damp_percent,
        inference_precision=inference_precision,
        paro_rotation_epochs=paro_rotation_epochs,
        paro_finetune_epochs=paro_finetune_epochs,
        extra_kwargs=moe_load_kwargs,
    )
    backend = _resolve_backend(device, weight_export=spec.weight_export)
    load_kwargs = {"quantize_config": qcfg}
    quantize_backend = BACKEND.TORCH if device.startswith("cuda") else backend
    if quantize_backend is not None:
        load_kwargs["backend"] = quantize_backend
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    print(f"\n--- Quantizing ({spec.label}) ---")
    if moe_load_kwargs:
        print("MoE: disabling offload_to_disk for eager expert module layout.")
    model = GPTQModel.load(model_id, **load_kwargs)
    configure_moe_quantize_config(model, model.quantize_config)
    if is_moe_gptq_model(model):
        print("MoE: using ExpertsRoutingOverride for calibration routing.")

    quantize_kwargs = {
        "batch_size": batch_size,
        "calibration_data_min_length": 10,
    }
    if calib_concat_size > 0:
        quantize_kwargs["calibration_concat_size"] = calib_concat_size
    if quantize_backend is not None:
        quantize_kwargs["backend"] = quantize_backend

    quant_result = model.quantize(calibration, **quantize_kwargs)
    log_summary = summarize_quant_log(quant_result, base_damp=damp_percent)

    slug = spec.label.replace(" + ", "_").replace(" ", "-")
    output_dir = work_dir / f"quantized-{slug}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    eval_device = next(model.model.parameters()).device
    del model

    reload_kwargs = {"device": device}
    reload_backend = _resolve_backend(device, weight_export=spec.weight_export)
    if reload_backend is not None:
        reload_kwargs["backend"] = reload_backend
    reloaded = GPTQModel.load(str(output_dir), **reload_kwargs)
    hook_count = _count_ptq_hooks(reloaded)
    hook_audit = audit_ptq_hooks(reloaded.model)

    if spec.weight_prepare == "random_orthogonal" and hook_count == 0:
        raise RuntimeError(
            f"{spec.label} checkpoint missing rehydrated T_X hooks (count={hook_count})."
        )

    ppl = compute_wikitext_perplexity(
        reloaded,
        reloaded.tokenizer,
        eval_device,
        seq_len=eval_seq_len,
        n_tokens=eval_n_tokens,
    )
    del reloaded

    return {
        "method": spec.label,
        "weight_prepare": spec.weight_prepare,
        "weight_quantize": spec.weight_quantize,
        "weight_export": spec.weight_export,
        "perplexity": ppl,
        "mean_quant_loss": log_summary.mean_loss,
        "max_damp": log_summary.max_damp,
        "modules_with_elevated_damp": log_summary.modules_with_elevated_damp,
        "quant_module_count": log_summary.module_count,
        "t_x_hooks": hook_count,
        "t_x_buffers": hook_audit.modules_with_t_x_buffers,
        "padded_modules": hook_audit.modules_with_pad[:10],
        "checkpoint": str(output_dir),
        "skipped": False,
    }


def _parse_combination(raw: str) -> CombinationSpec:
    parts = [part.strip() for part in raw.split("+")]
    if len(parts) != 2:
        raise ValueError(f"Expected prepare+quantize, got {raw!r}")
    prepare, quantize = parts
    export = "paroquant" if quantize == "paroquant" else "gptq"
    label = f"{prepare} + {quantize}"
    skip_reason = None
    if prepare == "wush":
        skip_reason = "WUSH transform is not vendored yet (see WUSH/ and ptq/transforms/wush.py)."
    return CombinationSpec(
        weight_prepare=prepare,  # type: ignore[arg-type]
        weight_quantize=quantize,  # type: ignore[arg-type]
        weight_export=export,  # type: ignore[arg-type]
        label=label,
        skip_reason=skip_reason,
    )


def _print_table(rows: list[dict[str, object]]) -> None:
    print(
        f"\n{'Method':<32} | {'PPL':>10} | {'Quant loss':>14} | "
        f"{'Max damp':>9} | {'T_X hooks':>9} | {'Status':>8}"
    )
    print("-" * 96)
    for row in rows:
        if row.get("skipped"):
            print(f"{row['method']:<32} | {'—':>10} | {'—':>14} | {'—':>9} | {'—':>9} | {'SKIP':>8}")
            continue
        ppl = row.get("perplexity")
        loss = row.get("mean_quant_loss")
        hooks = row.get("t_x_hooks", "—")
        max_damp = row.get("max_damp")
        ppl_str = f"{ppl:.4f}" if isinstance(ppl, float) and math.isfinite(ppl) else "nan"
        loss_str = f"{loss:.6f}" if isinstance(loss, float) else "—"
        hooks_str = str(hooks) if hooks != "—" else "—"
        max_damp_str = f"{max_damp:.5f}" if isinstance(max_damp, float) else "—"
        print(
            f"{row['method']:<32} | {ppl_str:>10} | {loss_str:>14} | "
            f"{max_damp_str:>9} | {hooks_str:>9} | {'OK':>8}"
        )
    print(
        "\nNote: quant loss is in each method's coordinate system and is not directly "
        "comparable across transform methods."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark PTQ prepare × quantize combinations on WikiText-2 PPL."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-concat-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-seq-len", type=int, default=2048)
    parser.add_argument("--eval-n-tokens", type=int, default=2048 * 32)
    parser.add_argument(
        "--combinations",
        default="",
        help="Comma-separated prepare+quantize pairs (e.g. identity+gptq,paroquant+gptq). "
        "Default: built-in matrix.",
    )
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--damp-percent", type=float, default=0.01)
    parser.add_argument(
        "--inference-precision",
        default="float32",
        choices=("float16", "float32", "bfloat16"),
    )
    parser.add_argument("--paro-rotation-epochs", type=int, default=1)
    parser.add_argument("--paro-finetune-epochs", type=int, default=0)
    args = parser.parse_args()

    if args.combinations.strip():
        specs = [_parse_combination(part.strip()) for part in args.combinations.split(",") if part.strip()]
    else:
        specs = list(DEFAULT_COMBINATIONS)

    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Combinations: {[spec.label for spec in specs]}")

    backend = _resolve_backend(args.device, weight_export="gptq")
    load_kwargs = {}
    if backend is not None:
        load_kwargs["backend"] = backend
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    baseline_model = GPTQModel.load(args.model_id, **load_kwargs)
    calibration = load_wikitext_calibration(
        baseline_model.tokenizer,
        max_samples=args.calib_samples,
        min_length=10,
        concat_size=0,
    )
    print(f"Calibration: {len(calibration)} WikiText train sample(s).")

    results: list[dict[str, object]] = []
    eval_device = next(baseline_model.model.parameters()).device

    if not args.skip_fp16:
        print("\n--- Evaluating FP16 baseline ---")
        fp16_ppl = compute_wikitext_perplexity(
            baseline_model,
            baseline_model.tokenizer,
            eval_device,
            seq_len=args.eval_seq_len,
            n_tokens=args.eval_n_tokens,
        )
        results.append({"method": "fp16 baseline", "perplexity": fp16_ppl, "skipped": False})
        print(f"FP16 baseline PPL: {fp16_ppl:.4f}")

    del baseline_model

    if args.work_dir is not None:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="ptq-combos-"))
        cleanup = True

    try:
        for spec in specs:
            if spec.skip_reason:
                row = {"method": spec.label, "skipped": True, "skip_reason": spec.skip_reason}
                results.append(row)
                print(f"{spec.label}: SKIP ({spec.skip_reason})")
                continue
            try:
                row = _run_combination(
                    model_id=args.model_id,
                    spec=spec,
                    calibration=calibration,
                    bits=args.bits,
                    group_size=args.group_size,
                    device=args.device,
                    batch_size=args.batch_size,
                    calib_concat_size=args.calib_concat_size,
                    eval_seq_len=args.eval_seq_len,
                    eval_n_tokens=args.eval_n_tokens,
                    work_dir=work_dir,
                    trust_remote_code=args.trust_remote_code,
                    damp_percent=args.damp_percent,
                    inference_precision=args.inference_precision,
                    paro_rotation_epochs=args.paro_rotation_epochs,
                    paro_finetune_epochs=args.paro_finetune_epochs,
                )
            except Exception as exc:
                row = {
                    "method": spec.label,
                    "skipped": True,
                    "skip_reason": str(exc),
                    "error": True,
                }
                print(f"{spec.label}: ERROR ({exc})", file=sys.stderr)
            results.append(row)
            if not row.get("skipped"):
                print(
                    f"{row['method']}: PPL={row['perplexity']:.4f} "
                    f"loss={row['mean_quant_loss']} hooks={row['t_x_hooks']}"
                )
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)

    _print_table(results)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": args.model_id,
            "device": args.device,
            "combinations": [spec.label for spec in specs],
            "results": results,
        }
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nWrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
