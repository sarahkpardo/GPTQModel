#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Audit reload forward path: dequant parity, forward-vs-dequant, hidden-state drift."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

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
    audit_tied_weight_aliasing,
    audit_torchlinear_forward_vs_dequant,
    capture_torchlinear_env_snapshot,
    compare_hidden_states_pre_post,
    compare_inmem_reload_dequant_layers,
    compare_logits_tensors,
    default_forward_audit_module_names,
    default_hidden_state_probe_names,
    summarize_reload_forward_audit,
)
from gptqmodel.utils.wikitext_benchmark import (  # noqa: E402
    compute_wikitext_perplexity_detailed,
    load_wikitext_calibration,
)


def _load_benchmark_module():
    script_path = ROOT / "scripts" / "benchmark_random_orthogonal_ppl.py"
    spec = importlib.util.spec_from_file_location("benchmark_random_orthogonal_ppl", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _resolve_flat_device_map(device: str) -> dict[str, str]:
    if device.startswith("cuda") and ":" not in device:
        return {"": "cuda:0"}
    return {"": device}


@torch.no_grad()
def _audit_batch(model: GPTQModel, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    batch = model.tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}


def _run_triton_ablation(
    reloaded: GPTQModel,
    module_names: list[str],
    batch: dict[str, torch.Tensor],
    probe_names: list[str],
    inmem_model: GPTQModel,
) -> dict[str, object]:
    previous = os.environ.get("GPTQ_TORCH_TRITON_DEQUANT")
    os.environ["GPTQ_TORCH_TRITON_DEQUANT"] = "0"
    try:
        for module in reloaded.model.modules():
            if hasattr(module, "clear_weight_cache"):
                module.clear_weight_cache()
            if hasattr(module, "_stream_reset_cache"):
                module._stream_reset_cache()
        forward_post = audit_torchlinear_forward_vs_dequant(
            reloaded.model,
            module_names,
            batch,
        )
        hidden = compare_hidden_states_pre_post(
            inmem_model.model,
            reloaded.model,
            batch,
            probe_names,
        )
        return {
            "env": capture_torchlinear_env_snapshot(),
            "post_reload_forward_vs_dequant": forward_post,
            "hidden_state_pre_vs_post": hidden,
        }
    finally:
        if previous is None:
            os.environ.pop("GPTQ_TORCH_TRITON_DEQUANT", None)
        else:
            os.environ["GPTQ_TORCH_TRITON_DEQUANT"] = previous


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit GPTQ reload forward path regressions.")
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
    parser.add_argument("--eval-n-tokens", type=int, default=4096)
    parser.add_argument("--damp-percent", type=float, default=0.01)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--dequant-all-modules",
        action="store_true",
        help="Tier-2 dequant parity across all TorchLinear modules.",
    )
    parser.add_argument(
        "--flat-device-map-ablation",
        action="store_true",
        help="Reload again with flat device_map and compare logits/PPL.",
    )
    parser.add_argument(
        "--triton-ablation",
        action="store_true",
        help="Re-run post-reload forward/hidden audits with GPTQ_TORCH_TRITON_DEQUANT=0.",
    )
    parser.add_argument("--skip-ppl", action="store_true")
    args = parser.parse_args()

    bench = _load_benchmark_module()
    eval_device = bench._resolve_eval_device(args.device)
    moe_load_kwargs = benchmark_quantize_load_kwargs(args.model_id, trust_remote_code=args.trust_remote_code)
    qcfg = bench._build_quantize_config(
        weight_prepare="identity",
        bits=args.bits,
        group_size=args.group_size,
        device=args.device,
        damp_percent=args.damp_percent,
        inference_precision="float16",
        legacy_random_orthogonal_damp=False,
        extra_kwargs=moe_load_kwargs,
    )
    ptq_backend = bench._resolve_ptq_backend(args.device, legacy_auto_reload=False)
    load_kwargs: dict[str, object] = {"quantize_config": qcfg}
    if ptq_backend is not None:
        load_kwargs["backend"] = ptq_backend
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    if args.work_dir is not None:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="reload-forward-audit-"))
        cleanup = True

    output_dir = work_dir / "quantized-identity"
    env_snapshot = capture_torchlinear_env_snapshot()

    try:
        print(f"Loading {args.model_id}...")
        model = GPTQModel.load(args.model_id, **load_kwargs)
        configure_moe_quantize_config(model, model.quantize_config)
        calibration = load_wikitext_calibration(
            model.tokenizer,
            max_samples=args.calib_samples,
            min_length=10,
            concat_size=0,
        )
        logits_prompt = bench._resolve_logits_prompt(calibration)
        print("Quantizing (identity + gptq)...")
        quantize_kwargs = {
            "batch_size": args.batch_size,
            "calibration_data_min_length": 10,
        }
        if args.calib_concat_size > 0:
            quantize_kwargs["calibration_concat_size"] = args.calib_concat_size
        if ptq_backend is not None:
            quantize_kwargs["backend"] = ptq_backend
        model.quantize(calibration, **quantize_kwargs)

        pre_eval_device = bench._prepare_model_for_ppl_eval(model, eval_device)
        module_names = default_forward_audit_module_names(model.model)
        probe_names = default_hidden_state_probe_names(model.model)
        audit_input = _audit_batch(model, logits_prompt, pre_eval_device)
        pre_forward = audit_torchlinear_forward_vs_dequant(
            model.model,
            module_names,
            audit_input,
        )
        pre_logits = bench._capture_reference_logits(model, logits_prompt, pre_eval_device)
        pre_ppl = None
        if not args.skip_ppl:
            pre_detail = compute_wikitext_perplexity_detailed(
                model,
                model.tokenizer,
                pre_eval_device,
                seq_len=args.eval_seq_len,
                n_tokens=args.eval_n_tokens,
            )
            pre_ppl = pre_detail.perplexity

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoint to {output_dir}...")
        model.save(str(output_dir))

        reload_kwargs: dict[str, object] = {"device": args.device}
        if ptq_backend is not None:
            reload_kwargs["backend"] = ptq_backend
        if args.trust_remote_code:
            reload_kwargs["trust_remote_code"] = True

        print("Reloading (layerwise device_map)...")
        reloaded = GPTQModel.load(str(output_dir), **reload_kwargs)
        dequant_per_layer = compare_inmem_reload_dequant_layers(model.model, reloaded.model)
        dequant_all_modules = None
        if args.dequant_all_modules:
            print("Running tier-2 dequant parity (all TorchLinear modules)...")
            dequant_all_modules = compare_inmem_reload_dequant_layers(
                model.model,
                reloaded.model,
                all_modules=True,
            )

        post_eval_device = bench._prepare_model_for_ppl_eval(reloaded, eval_device)
        post_forward = audit_torchlinear_forward_vs_dequant(
            reloaded.model,
            module_names,
            audit_input,
        )
        hidden_state_pre_vs_post = compare_hidden_states_pre_post(
            model.model,
            reloaded.model,
            audit_input,
            probe_names,
        )
        post_logits = bench._capture_reference_logits(reloaded, logits_prompt, post_eval_device)
        logits_pre_vs_post = compare_logits_tensors(pre_logits, post_logits)
        post_ppl = None
        if not args.skip_ppl:
            post_detail = compute_wikitext_perplexity_detailed(
                reloaded,
                reloaded.tokenizer,
                post_eval_device,
                seq_len=args.eval_seq_len,
                n_tokens=args.eval_n_tokens,
            )
            post_ppl = post_detail.perplexity

        gap_delta = {}
        for name in module_names:
            pre_row = pre_forward.get("per_module", {}).get(name, {})
            post_row = post_forward.get("per_module", {}).get(name, {})
            pre_rel = pre_row.get("mean_rel_error") if isinstance(pre_row, dict) else None
            post_rel = post_row.get("mean_rel_error") if isinstance(post_row, dict) else None
            if isinstance(pre_rel, (int, float)) and isinstance(post_rel, (int, float)):
                gap_delta[name] = float(post_rel) - float(pre_rel)
            else:
                gap_delta[name] = None

        forward_vs_dequant = {
            "module_names": module_names,
            "pre_inmem": pre_forward,
            "post_reload_layerwise": post_forward,
            "gap_delta": gap_delta,
            "post_reload_flat_map": None,
        }

        device_map_ablation = {
            "layerwise_hf_device_map": audit_tied_weight_aliasing(reloaded.model).get(
                "hf_device_map_present"
            ),
            "flat_map_logits_pre_vs_post": None,
            "flat_map_ppl_post": None,
            "flat_map_fixes_logits_or_ppl": None,
        }

        triton_ablation_result = None
        if args.flat_device_map_ablation:
            print("Flat device_map ablation reload...")
            flat_kwargs = dict(reload_kwargs)
            flat_kwargs["device_map"] = _resolve_flat_device_map(args.device)
            reloaded_flat = GPTQModel.load(str(output_dir), **flat_kwargs)
            flat_eval_device = bench._prepare_model_for_ppl_eval(reloaded_flat, eval_device)
            flat_logits = bench._capture_reference_logits(reloaded_flat, logits_prompt, flat_eval_device)
            flat_logits_delta = compare_logits_tensors(pre_logits, flat_logits)
            flat_ppl = None
            if not args.skip_ppl:
                flat_detail = compute_wikitext_perplexity_detailed(
                    reloaded_flat,
                    reloaded_flat.tokenizer,
                    flat_eval_device,
                    seq_len=args.eval_seq_len,
                    n_tokens=args.eval_n_tokens,
                )
                flat_ppl = flat_detail.perplexity
            forward_vs_dequant["post_reload_flat_map"] = audit_torchlinear_forward_vs_dequant(
                reloaded_flat.model,
                module_names,
                audit_input,
            )
            device_map_ablation["flat_map_logits_pre_vs_post"] = flat_logits_delta
            device_map_ablation["flat_map_ppl_post"] = flat_ppl
            layerwise_bad = isinstance(post_ppl, float) and post_ppl > 1000
            flat_ok = isinstance(flat_ppl, float) and math.isfinite(flat_ppl) and flat_ppl < 1000
            logits_improved = (
                isinstance(flat_logits_delta.get("mean_rel_error"), (int, float))
                and isinstance(logits_pre_vs_post.get("mean_rel_error"), (int, float))
                and float(flat_logits_delta["mean_rel_error"])
                < float(logits_pre_vs_post["mean_rel_error"])
            )
            device_map_ablation["flat_map_fixes_logits_or_ppl"] = bool(
                (layerwise_bad and flat_ok) or logits_improved
            )
            del reloaded_flat

        if args.triton_ablation:
            print("Triton-off ablation on layerwise reload...")
            triton_ablation_result = _run_triton_ablation(
                reloaded,
                module_names,
                audit_input,
                probe_names,
                model,
            )

        verdict = summarize_reload_forward_audit(
            dequant_per_layer=dequant_per_layer,
            dequant_all_modules=dequant_all_modules,
            forward_vs_dequant=forward_vs_dequant,
            hidden_state_pre_vs_post=hidden_state_pre_vs_post,
            device_map_ablation=device_map_ablation if args.flat_device_map_ablation else None,
        )

        payload = {
            "model_id": args.model_id,
            "device": args.device,
            "checkpoint": str(output_dir),
            "logits_prompt": logits_prompt,
            "env": env_snapshot,
            "forward_audit_module_names": module_names,
            "hidden_state_probe_names": probe_names,
            "ppl_pre_reload": pre_ppl,
            "ppl_post_reload_layerwise": post_ppl,
            "logits_pre_vs_post_reload": logits_pre_vs_post,
            "dequant_per_layer": dequant_per_layer,
            "dequant_all_modules": dequant_all_modules,
            "forward_vs_dequant": forward_vs_dequant,
            "hidden_state_pre_vs_post": hidden_state_pre_vs_post,
            "device_map_ablation": device_map_ablation if args.flat_device_map_ablation else None,
            "triton_ablation": triton_ablation_result,
            "tie_weights_pre_reload": audit_tied_weight_aliasing(model.model),
            "tie_weights_post_reload": audit_tied_weight_aliasing(reloaded.model),
            "verdict": verdict,
        }

        print("\n--- Reload forward audit verdict ---")
        for key, value in verdict.items():
            print(f"{key}: {value}")

        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with args.output_json.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            print(f"\nWrote {args.output_json}")

        del reloaded
        del model
        return 0
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
