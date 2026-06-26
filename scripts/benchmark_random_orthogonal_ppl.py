#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Benchmark GPTQ with identity vs random_orthogonal on WikiText-2 perplexity."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
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
    mean_quant_loss,
)

WeightPrepareMode = Literal["identity", "random_orthogonal"]


def _build_quantize_config(
    *,
    weight_prepare: WeightPrepareMode,
    bits: int,
    group_size: int,
    device: str,
    damp_percent: float,
    inference_precision: str,
    legacy_random_orthogonal_damp: bool,
    extra_kwargs: dict[str, object] | None = None,
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
        weight_prepare=[{"method": weight_prepare}],
        weight_quantize={"method": "gptq"},
        weight_export={"format": "gptq"},
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    if weight_prepare == "random_orthogonal":
        kwargs["weight_prepare"] = [
            {
                "method": "random_orthogonal",
                "group_size": group_size,
                "opt_seed": 42,
                "inference_precision": inference_precision,
            }
        ]
        if legacy_random_orthogonal_damp:
            kwargs["damp_percent"] = 0.05
    return QuantizeConfig(**kwargs)


def _resolve_backend(device: str):
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


def _run_method(
    *,
    model_id: str,
    weight_prepare: WeightPrepareMode,
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
    legacy_random_orthogonal_damp: bool,
) -> dict[str, object]:
    moe_load_kwargs = benchmark_quantize_load_kwargs(model_id, trust_remote_code=trust_remote_code)
    qcfg = _build_quantize_config(
        weight_prepare=weight_prepare,
        bits=bits,
        group_size=group_size,
        device=device,
        damp_percent=damp_percent,
        inference_precision=inference_precision,
        legacy_random_orthogonal_damp=legacy_random_orthogonal_damp,
        extra_kwargs=moe_load_kwargs,
    )
    backend = _resolve_backend(device)
    load_kwargs = {"quantize_config": qcfg}
    quantize_backend = BACKEND.TORCH if device.startswith("cuda") else backend
    if quantize_backend is not None:
        load_kwargs["backend"] = quantize_backend
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    print(f"\n--- Quantizing ({weight_prepare} + gptq) ---")
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
    avg_loss = log_summary.mean_loss

    output_dir = work_dir / f"quantized-{weight_prepare}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    eval_device = next(model.model.parameters()).device
    del model

    reload_kwargs = {"device": device}
    if backend is not None:
        reload_kwargs["backend"] = backend
    reloaded = GPTQModel.load(str(output_dir), **reload_kwargs)
    hook_count = _count_ptq_hooks(reloaded)
    hook_audit = audit_ptq_hooks(reloaded.model)

    if weight_prepare == "random_orthogonal" and hook_count == 0:
        raise RuntimeError(
            f"random_orthogonal checkpoint missing rehydrated T_X hooks (count={hook_count})."
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
        "method": f"{weight_prepare} + gptq",
        "weight_prepare": weight_prepare,
        "perplexity": ppl,
        "mean_quant_loss": avg_loss,
        "max_damp": log_summary.max_damp,
        "modules_with_elevated_damp": log_summary.modules_with_elevated_damp,
        "quant_module_count": log_summary.module_count,
        "t_x_hooks": hook_count,
        "t_x_buffers": hook_audit.modules_with_t_x_buffers,
        "padded_modules": hook_audit.modules_with_pad[:10],
        "checkpoint": str(output_dir),
    }


def _print_table(rows: list[dict[str, object]]) -> None:
    print(
        f"\n{'Method':<28} | {'PPL':>10} | {'Quant loss':>14} | "
        f"{'Max damp':>9} | {'Elev damp':>9} | {'T_X hooks':>9}"
    )
    print("-" * 92)
    for row in rows:
        ppl = row.get("perplexity")
        loss = row.get("mean_quant_loss")
        hooks = row.get("t_x_hooks", "—")
        max_damp = row.get("max_damp")
        elevated = row.get("modules_with_elevated_damp", "—")
        ppl_str = f"{ppl:.4f}" if isinstance(ppl, float) and math.isfinite(ppl) else "nan"
        loss_str = f"{loss:.6f}" if isinstance(loss, float) else "—"
        hooks_str = str(hooks) if hooks != "—" else "—"
        max_damp_str = f"{max_damp:.5f}" if isinstance(max_damp, float) else "—"
        elevated_str = str(elevated) if elevated != "—" else "—"
        print(
            f"{row['method']:<28} | {ppl_str:>10} | {loss_str:>14} | "
            f"{max_damp_str:>9} | {elevated_str:>9} | {hooks_str:>9}"
        )
    print(
        "\nNote: mean quant loss is computed in each method's GPTQ coordinates and is "
        "not directly comparable across identity vs random_orthogonal."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare GPTQ identity vs random_orthogonal on WikiText-2 PPL."
    )
    parser.add_argument("--model-id", required=True, help="HuggingFace model id or local path.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for quantization and evaluation.",
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-concat-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-seq-len", type=int, default=2048)
    parser.add_argument("--eval-n-tokens", type=int, default=2048 * 32)
    parser.add_argument(
        "--methods",
        default="identity,random_orthogonal",
        help="Comma-separated weight_prepare methods to benchmark.",
    )
    parser.add_argument("--skip-fp16", action="store_true", help="Skip FP16 baseline PPL.")
    parser.add_argument("--output-json", type=Path, default=None, help="Write results JSON.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for quantized checkpoints (default: temp dir).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when resolving/loading the model.",
    )
    parser.add_argument(
        "--damp-percent",
        type=float,
        default=0.01,
        help="Hessian damp_percent for all methods (fair comparison default).",
    )
    parser.add_argument(
        "--inference-precision",
        default="float16",
        choices=("float16", "float32", "bfloat16"),
        help="T_X storage precision for random_orthogonal inference hooks.",
    )
    parser.add_argument(
        "--legacy-random-orthogonal-damp",
        action="store_true",
        help="Use historical damp_percent=0.05 for random_orthogonal only (unfair ablation).",
    )
    args = parser.parse_args()

    methods: list[WeightPrepareMode] = []
    for raw in args.methods.split(","):
        name = raw.strip()
        if name not in {"identity", "random_orthogonal"}:
            raise SystemExit(f"Unsupported method {name!r}; use identity or random_orthogonal.")
        methods.append(name)  # type: ignore[arg-type]

    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Methods: {methods}")
    print(f"damp_percent: {args.damp_percent} (legacy random_orthogonal damp: {args.legacy_random_orthogonal_damp})")
    print(f"random_orthogonal inference_precision: {args.inference_precision}")

    backend = _resolve_backend(args.device)
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
        results.append(
            {
                "method": "fp16 baseline",
                "weight_prepare": None,
                "perplexity": fp16_ppl,
                "mean_quant_loss": None,
                "t_x_hooks": "—",
            }
        )
        print(f"FP16 baseline PPL: {fp16_ppl:.4f}")

    del baseline_model

    if args.work_dir is not None:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="rand-ortho-ppl-"))
        cleanup = True

    try:
        for method in methods:
            row = _run_method(
                model_id=args.model_id,
                weight_prepare=method,
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
                legacy_random_orthogonal_damp=args.legacy_random_orthogonal_damp,
            )
            results.append(row)
            ppl = row["perplexity"]
            print(
                f"{row['method']}: PPL={ppl:.4f} "
                f"loss={row['mean_quant_loss']} max_damp={row.get('max_damp')} "
                f"elevated_damp={row.get('modules_with_elevated_damp')} hooks={row['t_x_hooks']}"
            )
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)

    _print_table(results)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model_id": args.model_id,
                    "device": args.device,
                    "bits": args.bits,
                    "group_size": args.group_size,
                    "calib_samples": args.calib_samples,
                    "calib_concat_size": args.calib_concat_size,
                    "eval_seq_len": args.eval_seq_len,
                    "eval_n_tokens": args.eval_n_tokens,
                    "damp_percent": args.damp_percent,
                    "inference_precision": args.inference_precision,
                    "legacy_random_orthogonal_damp": args.legacy_random_orthogonal_damp,
                    "results": results,
                },
                handle,
                indent=2,
            )
        print(f"\nWrote {args.output_json}")

    for row in results:
        if row.get("weight_prepare") == "random_orthogonal":
            ppl = row.get("perplexity")
            if not isinstance(ppl, float) or not math.isfinite(ppl):
                print("FAIL: random_orthogonal PPL is not finite.", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
