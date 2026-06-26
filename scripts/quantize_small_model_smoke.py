#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test: quantize a small HF model, save, reload, and generate.

Uses Cholesky Hessian by default. The PTQ pipeline runs SequentialPTQProcessor
(capture → transform → quantize per module, Chen et al. ordering).

Usage:
    python scripts/quantize_small_model_smoke.py
    python scripts/quantize_small_model_smoke.py --model-id gpt2 --pipeline ptq --device cpu
    python scripts/quantize_small_model_smoke.py --model-id gpt2 --compare --device cpu
    python scripts/quantize_small_model_smoke.py --model-id gpt2 --pipeline ptq --weight-prepare random_orthogonal --weight-quantize gptq --device cpu --check-parity
    python scripts/quantize_small_model_smoke.py --model-fixture tiny-qwen3-moe --device cpu --pipeline ptq --weight-prepare random_orthogonal --weight-quantize gptq --check-parity
"""

from __future__ import annotations

import argparse
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

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402

PipelineMode = Literal["legacy", "ptq", "ptq-paroquant-moe"]
ModelFixture = Literal["hf", "tiny-qwen3-moe"]
WeightPrepareMode = Literal["identity", "paroquant", "random_orthogonal"]
WeightExportMode = Literal["gptq", "paroquant"]
WeightQuantizeMode = Literal["gptq", "rtn"]

_DEFAULT_CALIBRATION = [
    "GPTQModel quantizes language models with calibration data.",
    "Small models are useful for fast regression testing of quantization pipelines.",
    "Cholesky Hessian factorization avoids the QR path during PTQ refactor testing.",
    "Identity transforms should preserve legacy GPTQ behavior on full models.",
] * 2

_RANDOM_ORTHOGONAL_CALIBRATION = [
    "Random orthogonal transforms rotate weight blocks to reduce quantization error.",
    "Calibration data must provide enough activation samples for positive definite Hessians.",
    "Each transformer layer captures statistics during a forward pass over calibration batches.",
    "Mixture of experts models require routing overrides so every expert sees calibration data.",
    "Block diagonal orthogonal matrices preserve the bilinear inner product at inference time.",
    "GPTQ uses the transformed Hessian after offline weight rotation during quantization.",
] * 8


def _resolve_calibration(weight_prepare: WeightPrepareMode | None) -> list[str]:
    if weight_prepare == "random_orthogonal":
        return list(_RANDOM_ORTHOGONAL_CALIBRATION)
    return list(_DEFAULT_CALIBRATION)


def _build_quantize_config(
    pipeline: PipelineMode,
    *,
    bits: int,
    group_size: int,
    device: str,
    moe: bool = False,
    weight_prepare: WeightPrepareMode | None = None,
    weight_export: WeightExportMode | None = None,
    weight_quantize: WeightQuantizeMode | None = None,
) -> QuantizeConfig:
    kwargs = dict(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        damp_percent=0.01,
        damp_auto_increment=0.01,
        device=device,
        hessian={"factorization": "cholesky", "row_buffer_max_rows": 512},
    )
    if pipeline == "ptq-paroquant-moe":
        weight_prepare = "paroquant"
        weight_export = "paroquant"
    if pipeline == "ptq":
        kwargs["weight_prepare"] = [{"method": "identity"}]
    if weight_prepare == "random_orthogonal":
        kwargs["weight_prepare"] = [
            {
                "method": "random_orthogonal",
                "group_size": group_size,
                "opt_seed": 42,
            }
        ]
        kwargs["weight_export"] = {"format": "gptq"}
        kwargs["damp_percent"] = 0.05
        if pipeline == "legacy":
            kwargs.setdefault("weight_quantize", {"method": "gptq"})
    if weight_prepare == "paroquant":
        kwargs["weight_prepare"] = [
            {
                "method": "paroquant",
                "opt_rotation_epochs": 1,
                "opt_finetune_epochs": 0,
                "krot": 2,
                "opt_fused_rotation": True,
                "opt_train_samples": 64,
                "opt_validation_samples": 16,
                "opt_batch_size": 16,
                "group_size": group_size,
            }
        ]
        kwargs["weight_quantize"] = {"method": "paroquant"}
    if weight_export == "paroquant":
        kwargs["weight_export"] = {"format": "paroquant"}
    if weight_quantize in {"gptq", "rtn"}:
        kwargs["weight_quantize"] = {"method": weight_quantize}
    if moe:
        from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig

        kwargs["moe"] = MoEConfig(routing=ExpertsRoutingOverride())
    return QuantizeConfig(**kwargs)


def _build_tiny_qwen3_moe_fixture(
    model_dir: Path,
    *,
    moe_intermediate_size: int = 32,
    calibration_texts: list[str] | None = None,
) -> str:
    """Build and save a tiny Qwen3 MoE checkpoint; return the model directory path."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    texts = list(calibration_texts or _DEFAULT_CALIBRATION)
    config = Qwen3MoeConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=moe_intermediate_size,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3MoeForCausalLM(config)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)
    return str(model_dir)


def _resolve_model_source(
    *,
    model_id: str,
    model_fixture: ModelFixture,
    work_dir: Path,
    moe_intermediate_size: int = 32,
    calibration_texts: list[str] | None = None,
) -> str:
    if model_fixture == "tiny-qwen3-moe":
        return _build_tiny_qwen3_moe_fixture(
            work_dir / "tiny-qwen3-moe",
            moe_intermediate_size=moe_intermediate_size,
            calibration_texts=calibration_texts,
        )
    return model_id


def _count_ptq_hooks(model: GPTQModel) -> int:
    import torch

    count = 0
    for module in model.model.modules():
        t_x = getattr(module, "ptq_t_x_matrices", None)
        if isinstance(t_x, torch.Tensor) and t_x.numel() > 0 and module._forward_pre_hooks:
            count += 1
    return count


def _check_parity(
    *,
    model_source: str,
    reloaded: GPTQModel,
    prompt: str,
    weight_prepare: WeightPrepareMode | None,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    if weight_prepare != "random_orthogonal":
        return

    hook_count = _count_ptq_hooks(reloaded)
    if hook_count == 0:
        raise SystemExit(
            "FAIL: random_orthogonal parity check found no rehydrated T_X hooks after reload."
        )
    print(f"Parity: rehydrated T_X hooks on {hook_count} module(s).")

    reference = AutoModelForCausalLM.from_pretrained(model_source)
    reference.eval()
    reloaded.model.eval()
    tokenizer = reloaded.tokenizer
    batch = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        ref_logits = reference(**batch).logits
        quant_logits = reloaded.model(**batch).logits
    if not torch.isfinite(quant_logits).all():
        raise SystemExit("FAIL: random_orthogonal parity check produced non-finite logits.")
    rel_err = (quant_logits - ref_logits).abs().mean() / ref_logits.abs().mean().clamp(min=1e-6)
    print(f"Parity: mean relative logits error vs fp32 reference = {rel_err.item():.4f}")
    if rel_err.item() > 0.75:
        raise SystemExit(
            f"FAIL: random_orthogonal logits diverged from fp32 reference (rel_err={rel_err.item():.4f})."
        )
    print("PASS: random_orthogonal parity check succeeded.")


def _resolve_backend(device: str, *, weight_export: WeightExportMode | None = None):
    if device == "cpu":
        return BACKEND.TORCH
    if weight_export == "paroquant" and device.startswith("cuda"):
        return BACKEND.PAROQUANT_CUDA
    return None


def _run_pipeline_smoke(
    *,
    model_id: str,
    model_fixture: ModelFixture,
    work_dir: Path,
    pipeline: PipelineMode,
    output_dir: Path,
    calibration: list[str],
    bits: int,
    group_size: int,
    batch_size: int,
    device: str,
    prompt: str,
    max_new_tokens: int,
    weight_prepare: WeightPrepareMode | None = None,
    weight_export: WeightExportMode | None = None,
    weight_quantize: WeightQuantizeMode | None = None,
    check_parity: bool = False,
) -> list[int]:
    moe = model_fixture == "tiny-qwen3-moe"
    qcfg = _build_quantize_config(
        pipeline,
        bits=bits,
        group_size=group_size,
        device=device,
        moe=moe,
        weight_prepare=weight_prepare,
        weight_export=weight_export,
        weight_quantize=weight_quantize,
    )
    export_mode = weight_export
    if pipeline == "ptq-paroquant-moe":
        export_mode = "paroquant"
    elif export_mode is None and qcfg.weight_export is not None:
        export_mode = str(qcfg.weight_export.get("format", "gptq"))
    backend = _resolve_backend(device, weight_export=export_mode if export_mode in {"gptq", "paroquant"} else None)
    moe_intermediate_size = 64 if export_mode == "paroquant" else 32
    model_source = _resolve_model_source(
        model_id=model_id,
        model_fixture=model_fixture,
        work_dir=work_dir,
        moe_intermediate_size=moe_intermediate_size,
        calibration_texts=calibration,
    )

    print(f"Loading {model_source!r} (pipeline={pipeline}, fixture={model_fixture}, factorization=cholesky)...")
    load_kwargs = {"quantize_config": qcfg}
    quantize_backend = BACKEND.TORCH if device.startswith("cuda") else backend
    if quantize_backend is not None:
        load_kwargs["backend"] = quantize_backend
    model = GPTQModel.load(model_source, **load_kwargs)

    print("Quantizing...")
    quantize_kwargs = {
        "batch_size": batch_size,
        "calibration_data_min_length": 1,
    }
    if quantize_backend is not None:
        quantize_kwargs["backend"] = quantize_backend
    model.quantize(calibration, **quantize_kwargs)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_dir}...")
    model.save(str(output_dir))
    del model

    reload_kwargs = {"device": device}
    if backend is not None:
        reload_kwargs["backend"] = backend
    print("Reloading quantized checkpoint...")
    reloaded = GPTQModel.load(str(output_dir), **reload_kwargs)

    print(f"Generating from prompt: {prompt!r}")
    token_ids = reloaded.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    decoded = reloaded.tokenizer.decode(token_ids[0], skip_special_tokens=True)
    print(f"Generation: {decoded}")
    if check_parity:
        _check_parity(
            model_source=model_source,
            reloaded=reloaded,
            prompt=prompt,
            weight_prepare=weight_prepare,
        )
    del reloaded
    return token_ids[0].tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test GPTQ quantization on a small model.")
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument(
        "--model-fixture",
        choices=("hf", "tiny-qwen3-moe"),
        default="hf",
        help="hf = --model-id from HuggingFace; tiny-qwen3-moe = synthetic 1-layer MoE",
    )
    parser.add_argument(
        "--pipeline",
        choices=("legacy", "ptq", "ptq-paroquant-moe"),
        default="legacy",
        help="legacy = implicit identity prepare; ptq = explicit weight_prepare identity; "
        "ptq-paroquant-moe = tiny MoE ParoQuant PTQ preset",
    )
    parser.add_argument(
        "--weight-prepare",
        choices=("identity", "paroquant", "random_orthogonal"),
        default=None,
        help="Override PTQ weight_prepare (default: pipeline preset)",
    )
    parser.add_argument(
        "--weight-quantize",
        choices=("gptq", "rtn"),
        default=None,
        help="Override PTQ weight_quantize method (default: gptq)",
    )
    parser.add_argument(
        "--weight-export",
        choices=("gptq", "paroquant"),
        default=None,
        help="PTQ weight_export format (default: pipeline preset)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--check-parity",
        action="store_true",
        help="After reload, verify random_orthogonal T_X hooks and logits vs fp32 reference.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run legacy and ptq pipelines and compare generated token ids.",
    )
    args = parser.parse_args()
    if args.model_fixture == "tiny-qwen3-moe" and args.group_size == 128:
        args.group_size = 32
    if args.weight_prepare == "random_orthogonal" and args.pipeline == "legacy":
        print(
            "Note: random_orthogonal requires PTQ weight_prepare; using pipeline=ptq.",
            file=sys.stderr,
        )
        args.pipeline = "ptq"

    if args.compare and args.pipeline != "legacy":
        print("Note: --compare runs both legacy and ptq; ignoring --pipeline.", file=sys.stderr)

    calibration = _resolve_calibration(args.weight_prepare)

    with tempfile.TemporaryDirectory(prefix="gptqmodel-smoke-") as tmp_root:
        tmp_path = Path(tmp_root)
        if args.compare:
            legacy_tokens = _run_pipeline_smoke(
                model_id=args.model_id,
                model_fixture=args.model_fixture,
                work_dir=tmp_path,
                pipeline="legacy",
                output_dir=tmp_path / "legacy",
                calibration=calibration,
                bits=args.bits,
                group_size=args.group_size,
                batch_size=args.batch_size,
                device=args.device,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
            )
            ptq_tokens = _run_pipeline_smoke(
                model_id=args.model_id,
                model_fixture=args.model_fixture,
                work_dir=tmp_path,
                pipeline="ptq",
                output_dir=tmp_path / "ptq",
                calibration=calibration,
                bits=args.bits,
                group_size=args.group_size,
                batch_size=args.batch_size,
                device=args.device,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
            )
            if legacy_tokens == ptq_tokens:
                print("PASS: legacy and PTQ pipelines produced identical token ids.")
                return 0
            print("FAIL: legacy and PTQ pipelines produced different token ids.")
            print(f"  legacy: {legacy_tokens}")
            print(f"  ptq:    {ptq_tokens}")
            return 1

        output_dir = args.output_dir or (tmp_path / args.pipeline)
        _run_pipeline_smoke(
            model_id=args.model_id,
            model_fixture=args.model_fixture,
            work_dir=tmp_path,
            pipeline=args.pipeline,
            output_dir=output_dir,
            calibration=calibration,
            bits=args.bits,
            group_size=args.group_size,
            batch_size=args.batch_size,
            device=args.device,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            weight_prepare=args.weight_prepare,
            weight_export=args.weight_export,
            weight_quantize=args.weight_quantize,
            check_parity=args.check_parity,
        )
        if args.output_dir is not None:
            print(f"Checkpoint kept at {args.output_dir}")

    print(f"PASS: {args.pipeline} pipeline smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
