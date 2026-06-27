#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Benchmark GPTQ with identity vs random_orthogonal on WikiText-2 perplexity."""

from __future__ import annotations

import argparse
import json
import math
import os
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
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from gptqmodel.utils.moe_benchmark import (  # noqa: E402
    benchmark_quantize_load_kwargs,
    configure_moe_quantize_config,
    is_moe_gptq_model,
)
from gptqmodel.utils.random_orthogonal_diag import (  # noqa: E402
    audit_checkpoint_load_keys,
    audit_ptq_hooks,
    audit_quant_kernel_types,
    audit_tied_weight_aliasing,
    clear_torchlinear_inference_state,
    compare_inmem_reload_dequant_eager,
    compare_logits_tensors,
    resolve_model_param_dtype,
    sample_torchlinear_qweight_device,
    summarize_quant_log,
)
from gptqmodel.utils.wikitext_benchmark import (  # noqa: E402
    compute_logits_relative_error,
    compute_wikitext_perplexity,
    compute_wikitext_perplexity_detailed,
    load_wikitext_calibration,
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


def _resolve_ptq_backend(device: str, *, legacy_auto_reload: bool = False):
    """Return the kernel backend used for PTQ pack/reload in this benchmark."""
    if legacy_auto_reload and device.startswith("cuda"):
        return None
    if device.startswith("cuda") or device == "cpu":
        return BACKEND.TORCH
    return None


def _count_ptq_hooks(model: GPTQModel) -> int:
    count = 0
    for module in model.model.modules():
        t_x = getattr(module, "ptq_t_x_matrices", None)
        if isinstance(t_x, torch.Tensor) and t_x.numel() > 0 and module._forward_pre_hooks:
            count += 1
    return count


def _count_quant_linear_modules(model: GPTQModel) -> int:
    return sum(1 for module in model.model.modules() if isinstance(module, BaseQuantLinear))


def _resolve_eval_device(device: str) -> torch.device:
    """Return the benchmark evaluation device from the CLI ``--device`` flag."""
    return torch.device(device)


def _prepare_model_for_ppl_eval(model: GPTQModel, eval_device: torch.device) -> torch.device:
    """Move a freshly quantized model onto the eval device before PPL measurement."""
    model.to(eval_device)
    cleared = clear_torchlinear_inference_state(model.model)
    if cleared:
        print(f"PPL eval: cleared TorchLinear inference caches on {cleared} module(s).")
    actual = next(model.model.parameters()).device
    print(f"PPL eval device: {actual}")
    return actual


def _resolve_logits_prompt(calibration: list[str]) -> str:
    for sample in calibration:
        stripped = sample.strip()
        if stripped:
            return stripped[:512]
    return "The history of machine learning begins with early statistical methods."


@torch.no_grad()
def _capture_reference_logits(model: GPTQModel, prompt: str, device: torch.device) -> torch.Tensor:
    model.model.eval()
    batch = model.tokenizer(prompt, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        return model.model(**batch).logits.detach().cpu()


def _kernel_audit_dict(audit) -> dict[str, object]:
    return {
        "torch_linear": audit.torch_linear,
        "marlin_linear": audit.marlin_linear,
        "other_quant_linear": audit.other_quant_linear,
        "plain_linear": audit.plain_linear,
        "sample_torch_linear": audit.sample_torch_linear,
        "sample_marlin_linear": audit.sample_marlin_linear,
        "sample_plain_linear": audit.sample_plain_linear,
        "sample_other_quant_linear": audit.sample_other_quant_linear,
    }


def _ppl_eval_dict(result) -> dict[str, object]:
    return {
        "perplexity": result.perplexity,
        "n_windows": result.n_windows,
        "all_losses_finite": result.all_losses_finite,
        "non_finite_window_indices": result.non_finite_window_indices,
        "mean_loss": result.mean_loss,
    }


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
    eval_n_tokens_pre_reload: int | None,
    skip_pre_reload_ppl: bool,
    work_dir: Path,
    trust_remote_code: bool,
    damp_percent: float,
    inference_precision: str,
    legacy_random_orthogonal_damp: bool,
    legacy_auto_reload: bool,
    logits_prompt: str,
    reference_logits: torch.Tensor | None,
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
    ptq_backend = _resolve_ptq_backend(device, legacy_auto_reload=False)
    load_kwargs = {"quantize_config": qcfg}
    if ptq_backend is not None:
        load_kwargs["backend"] = ptq_backend
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
    if ptq_backend is not None:
        quantize_kwargs["backend"] = ptq_backend

    quant_result = model.quantize(calibration, **quantize_kwargs)
    log_summary = summarize_quant_log(quant_result, base_damp=damp_percent)
    avg_loss = log_summary.mean_loss

    eval_device = _resolve_eval_device(device)
    quant_linear_pre_reload = _count_quant_linear_modules(model)
    hook_audit_pre_reload = audit_ptq_hooks(model.model)
    kernel_audit_pre = audit_quant_kernel_types(model.model)
    eval_param_dtype_pre = resolve_model_param_dtype(model.model)

    ppl_pre_reload: float | None = None
    pre_reload_ppl_detail: dict[str, object] | None = None
    pre_reload_eval_device: torch.device | None = None
    pre_reload_logits: torch.Tensor | None = None
    tie_weights_pre_reload: dict[str, object] | None = None
    if not skip_pre_reload_ppl:
        pre_reload_eval_device = _prepare_model_for_ppl_eval(model, eval_device)
        tie_weights_pre_reload = audit_tied_weight_aliasing(model.model)
        pre_reload_logits = _capture_reference_logits(model, logits_prompt, pre_reload_eval_device)
        pre_reload_n_tokens = eval_n_tokens_pre_reload or eval_n_tokens
        print(
            f"Evaluating in-memory PPL ({weight_prepare}) "
            f"on {pre_reload_eval_device} ({pre_reload_n_tokens} tokens)..."
        )
        pre_detail = compute_wikitext_perplexity_detailed(
            model,
            model.tokenizer,
            pre_reload_eval_device,
            seq_len=eval_seq_len,
            n_tokens=pre_reload_n_tokens,
        )
        ppl_pre_reload = pre_detail.perplexity
        pre_reload_ppl_detail = _ppl_eval_dict(pre_detail)
        if not pre_detail.all_losses_finite:
            print(
                f"WARNING: in-memory PPL had non-finite losses in windows "
                f"{pre_detail.non_finite_window_indices}"
            )
    else:
        print(f"Skipping in-memory PPL ({weight_prepare}); --skip-pre-reload-ppl set.")

    output_dir = work_dir / f"quantized-{weight_prepare}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))

    reload_backend = _resolve_ptq_backend(device, legacy_auto_reload=legacy_auto_reload)
    reload_kwargs = {"device": device}
    if reload_backend is not None:
        reload_kwargs["backend"] = reload_backend
    reload_backend_label = reload_backend.value if reload_backend is not None else "auto"
    if legacy_auto_reload and device.startswith("cuda"):
        print(
            "WARNING: --legacy-auto-reload-backend set — CUDA reload uses backend=AUTO "
            "(typically Marlin). Expect dtype errors on bf16 models such as Qwen3."
        )
    print(f"Reloading checkpoint with backend={reload_backend_label}...")
    reloaded = GPTQModel.load(str(output_dir), **reload_kwargs)

    dequant_parity = compare_inmem_reload_dequant_eager(model.model, reloaded.model)
    if not dequant_parity["all_match"]:
        print(f"WARNING: in-memory vs reload dequant mismatch: {dequant_parity['comparisons'][:1]}")
    tie_weights_pre_reload = tie_weights_pre_reload or audit_tied_weight_aliasing(model.model)
    del model

    reload_load_keys = audit_checkpoint_load_keys(reloaded.model, output_dir)

    hook_count = _count_ptq_hooks(reloaded)
    hook_audit = audit_ptq_hooks(reloaded.model)
    quant_linear_post_reload = _count_quant_linear_modules(reloaded)
    kernel_audit_post = audit_quant_kernel_types(reloaded.model)
    eval_param_dtype_post = resolve_model_param_dtype(reloaded.model)

    if kernel_audit_post.marlin_linear > 0 and reload_backend_label == "torch":
        raise RuntimeError(
            f"Expected TorchLinear reload backend but found {kernel_audit_post.marlin_linear} "
            f"MarlinLinear module(s): {kernel_audit_post.sample_marlin_linear}"
        )

    if weight_prepare == "random_orthogonal" and hook_count == 0:
        raise RuntimeError(
            f"random_orthogonal checkpoint missing rehydrated T_X hooks (count={hook_count})."
        )

    print(f"Evaluating post-reload PPL ({weight_prepare})...")
    post_reload_eval_device = _prepare_model_for_ppl_eval(reloaded, eval_device)
    quant_buffer_device_post = sample_torchlinear_qweight_device(reloaded.model)
    tie_weights_post_reload = audit_tied_weight_aliasing(reloaded.model)
    post_reload_logits = _capture_reference_logits(reloaded, logits_prompt, post_reload_eval_device)
    post_detail = compute_wikitext_perplexity_detailed(
        reloaded,
        reloaded.tokenizer,
        post_reload_eval_device,
        seq_len=eval_seq_len,
        n_tokens=eval_n_tokens,
    )
    ppl_post_reload = post_detail.perplexity
    post_reload_ppl_detail = _ppl_eval_dict(post_detail)
    if not post_detail.all_losses_finite:
        print(
            f"WARNING: post-reload PPL had non-finite losses in windows "
            f"{post_detail.non_finite_window_indices}"
        )

    logits_rel_error = None
    if reference_logits is not None:
        logits_rel_error = compute_logits_relative_error(
            reloaded,
            reloaded.tokenizer,
            logits_prompt,
            reference_logits,
            post_reload_eval_device,
        )

    logits_pre_vs_post_reload = None
    if pre_reload_logits is not None:
        logits_pre_vs_post_reload = compare_logits_tensors(pre_reload_logits, post_reload_logits)

    del reloaded

    return {
        "method": f"{weight_prepare} + gptq",
        "weight_prepare": weight_prepare,
        "perplexity": ppl_post_reload,
        "ppl_pre_reload": ppl_pre_reload,
        "ppl_post_reload": ppl_post_reload,
        "ppl_pre_reload_detail": pre_reload_ppl_detail,
        "ppl_post_reload_detail": post_reload_ppl_detail,
        "ppl_pre_reload_loss_finite": (
            pre_reload_ppl_detail["all_losses_finite"] if pre_reload_ppl_detail else None
        ),
        "ppl_post_reload_loss_finite": post_reload_ppl_detail["all_losses_finite"],
        "ppl_reload_delta": (
            (ppl_post_reload - ppl_pre_reload)
            if isinstance(ppl_pre_reload, float) and isinstance(ppl_post_reload, float)
            else None
        ),
        "pre_reload_eval_device": str(pre_reload_eval_device) if pre_reload_eval_device is not None else None,
        "post_reload_eval_device": str(post_reload_eval_device),
        "quant_buffer_device_post_reload": quant_buffer_device_post,
        "eval_param_dtype_pre_reload": eval_param_dtype_pre,
        "eval_param_dtype_post_reload": eval_param_dtype_post,
        "reload_backend": reload_backend_label,
        "kernel_types_pre_reload": _kernel_audit_dict(kernel_audit_pre),
        "kernel_types_post_reload": _kernel_audit_dict(kernel_audit_post),
        "dequant_inmem_vs_reload": dequant_parity,
        "mean_quant_loss": avg_loss,
        "max_damp": log_summary.max_damp,
        "modules_with_elevated_damp": log_summary.modules_with_elevated_damp,
        "quant_module_count": log_summary.module_count,
        "quant_linear_pre_reload": quant_linear_pre_reload,
        "quant_linear_post_reload": quant_linear_post_reload,
        "t_x_hooks": hook_count,
        "t_x_hooks_pre_reload": hook_audit_pre_reload.modules_with_active_hooks,
        "t_x_buffers": hook_audit.modules_with_t_x_buffers,
        "t_x_buffers_pre_reload": hook_audit_pre_reload.modules_with_t_x_buffers,
        "padded_modules": hook_audit.modules_with_pad[:10],
        "logits_rel_error_vs_fp16": logits_rel_error,
        "logits_pre_vs_post_reload": logits_pre_vs_post_reload,
        "tie_weights_pre_reload": tie_weights_pre_reload,
        "tie_weights_post_reload": tie_weights_post_reload,
        "reload_load_keys": reload_load_keys,
        "checkpoint": str(output_dir),
    }


def _print_table(rows: list[dict[str, object]]) -> None:
    print(
        f"\n{'Method':<28} | {'PPL post':>10} | {'PPL pre':>10} | {'Quant loss':>14} | "
        f"{'Logits Δ':>10} | {'T_X hooks':>9}"
    )
    print("-" * 98)
    for row in rows:
        ppl = row.get("ppl_post_reload", row.get("perplexity"))
        ppl_pre = row.get("ppl_pre_reload", "—")
        loss = row.get("mean_quant_loss")
        hooks = row.get("t_x_hooks", "—")
        logits_delta = row.get("logits_pre_vs_post_reload")
        if isinstance(logits_delta, dict):
            logits_err = logits_delta.get("mean_rel_error")
        else:
            logits_err = row.get("logits_rel_error_vs_fp16")
        ppl_str = f"{ppl:.4f}" if isinstance(ppl, float) and math.isfinite(ppl) else "nan"
        ppl_pre_str = (
            f"{ppl_pre:.4f}" if isinstance(ppl_pre, float) and math.isfinite(ppl_pre) else "—"
        )
        loss_str = f"{loss:.6f}" if isinstance(loss, float) else "—"
        hooks_str = str(hooks) if hooks != "—" else "—"
        logits_str = f"{logits_err:.4f}" if isinstance(logits_err, float) else "—"
        print(
            f"{row['method']:<28} | {ppl_str:>10} | {ppl_pre_str:>10} | {loss_str:>14} | "
            f"{logits_str:>10} | {hooks_str:>9}"
        )
    print(
        "\nNote: mean quant loss is computed in each method's GPTQ coordinates and is "
        "not directly comparable across identity vs random_orthogonal."
    )
    print(
        "Note: PPL pre = in-memory after quantize; PPL post = after save/reload. "
        "A large delta implicates reload/kernel wiring. "
        "In-memory eval moves the model to --device (quantize leaves weights on CPU)."
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
        "--eval-n-tokens-pre-reload",
        type=int,
        default=None,
        help="Token budget for in-memory PPL only (default: same as --eval-n-tokens). "
        "Use a smaller value for a faster reload-isolation check.",
    )
    parser.add_argument(
        "--skip-pre-reload-ppl",
        action="store_true",
        help="Skip the in-memory PPL pass (post-reload PPL only).",
    )
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
    parser.add_argument(
        "--legacy-auto-reload-backend",
        action="store_true",
        help="Reproduce old CUDA reload behavior (backend=AUTO instead of TORCH).",
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
    print(f"reload backend: {'auto (legacy)' if args.legacy_auto_reload_backend else 'torch'}")
    if args.legacy_auto_reload_backend and args.device.startswith("cuda"):
        print(
            "WARNING: --legacy-auto-reload-backend — CUDA reload will use AUTO (Marlin). "
            "Use default reload for TORCH parity tests."
        )
    compile_disabled = os.environ.get("GPTQ_TORCH_DISABLE_COMPILE", "0") not in {"0", "false", "False"}
    print(f"GPTQ_TORCH_DISABLE_COMPILE: {compile_disabled}")

    ptq_backend = _resolve_ptq_backend(
        args.device,
        legacy_auto_reload=args.legacy_auto_reload_backend,
    )
    load_kwargs = {}
    if ptq_backend is not None:
        load_kwargs["backend"] = ptq_backend
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
    eval_device = _resolve_eval_device(args.device)
    logits_prompt = _resolve_logits_prompt(calibration)
    reference_logits = None
    if not args.skip_fp16:
        reference_logits = _capture_reference_logits(baseline_model, logits_prompt, eval_device)

    if not args.skip_fp16:
        print("\n--- Evaluating FP16 baseline ---")
        if next(baseline_model.model.parameters()).device != eval_device:
            baseline_model.to(eval_device)
        print(f"PPL eval device: {next(baseline_model.model.parameters()).device}")
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
                "ppl_post_reload": fp16_ppl,
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
                eval_n_tokens_pre_reload=args.eval_n_tokens_pre_reload,
                skip_pre_reload_ppl=args.skip_pre_reload_ppl,
                work_dir=work_dir,
                trust_remote_code=args.trust_remote_code,
                damp_percent=args.damp_percent,
                inference_precision=args.inference_precision,
                legacy_random_orthogonal_damp=args.legacy_random_orthogonal_damp,
                legacy_auto_reload=args.legacy_auto_reload_backend,
                logits_prompt=logits_prompt,
                reference_logits=reference_logits,
            )
            results.append(row)
            print(
                f"{row['method']}: pre={row.get('ppl_pre_reload', '—')} post={row['ppl_post_reload']:.4f} "
                f"delta={row.get('ppl_reload_delta', '—')} loss={row['mean_quant_loss']} "
                f"logits_pre_post={row.get('logits_pre_vs_post_reload', {}).get('mean_rel_error')} "
                f"tie_post={row.get('tie_weights_post_reload', {}).get('aliased_as_expected')} "
                f"hooks={row['t_x_hooks']} reload_backend={row['reload_backend']} "
                f"pre_dev={row.get('pre_reload_eval_device')} post_dev={row.get('post_reload_eval_device')}"
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
                    "legacy_auto_reload_backend": args.legacy_auto_reload_backend,
                    "gptq_torch_disable_compile": compile_disabled,
                    "skip_pre_reload_ppl": args.skip_pre_reload_ppl,
                    "eval_n_tokens_pre_reload": args.eval_n_tokens_pre_reload,
                    "logits_prompt": logits_prompt,
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
