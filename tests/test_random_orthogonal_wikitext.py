# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""WikiText integration tests for random orthogonal GPTQ with real tokenizers."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.utils.moe_benchmark import configure_moe_quantize_config  # noqa: E402
from gptqmodel.utils.wikitext_benchmark import (  # noqa: E402
    compute_wikitext_perplexity,
    load_wikitext_calibration,
)

pytestmark = [pytest.mark.colab, pytest.mark.slow]

_BENCHMARK_MODEL_ENV = "GPTQMODEL_BENCHMARK_MODEL_ID"
_CALIB_SAMPLES = 32
_EVAL_N_TOKENS = 4096
_EVAL_SEQ_LEN = 512


def _require_benchmark_model_id() -> str:
    model_id = os.environ.get(_BENCHMARK_MODEL_ENV, "").strip()
    if not model_id:
        pytest.skip(
            f"Set {_BENCHMARK_MODEL_ENV} to a small causal LM id (e.g. gpt2) for WikiText tests."
        )
    return model_id


def _require_datasets():
    pytest.importorskip("datasets")


def _build_quantize_config(*, weight_prepare: str, group_size: int = 128) -> QuantizeConfig:
    kwargs = dict(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        damp_percent=0.01,
        damp_auto_increment=0.01,
        device="cpu",
        hessian={"factorization": "cholesky", "row_buffer_max_rows": 512},
        weight_prepare=[{"method": weight_prepare}],
        weight_quantize={"method": "gptq"},
        weight_export={"format": "gptq"},
    )
    if weight_prepare == "random_orthogonal":
        kwargs["weight_prepare"] = [
            {
                "method": "random_orthogonal",
                "group_size": group_size,
                "opt_seed": 42,
            }
        ]
        kwargs["damp_percent"] = 0.05
    return QuantizeConfig(**kwargs)


def _count_ptq_hooks(model: GPTQModel) -> int:
    count = 0
    for module in model.model.modules():
        t_x = getattr(module, "ptq_t_x_matrices", None)
        if isinstance(t_x, torch.Tensor) and t_x.numel() > 0 and module._forward_pre_hooks:
            count += 1
    return count


@pytest.mark.parametrize("weight_prepare", ["identity", "random_orthogonal"])
def test_wikitext_quantize_reload_perplexity(tmp_path: Path, weight_prepare: str):
    _require_datasets()
    model_id = _require_benchmark_model_id()

    qcfg = _build_quantize_config(weight_prepare=weight_prepare)
    model = GPTQModel.load(model_id, quantize_config=qcfg, backend=BACKEND.TORCH)
    configure_moe_quantize_config(model, model.quantize_config)
    calibration = load_wikitext_calibration(
        model.tokenizer,
        max_samples=_CALIB_SAMPLES,
        min_length=10,
    )
    assert calibration, "expected non-empty WikiText calibration"

    model.quantize(
        calibration,
        batch_size=1,
        backend=BACKEND.TORCH,
        calibration_data_min_length=10,
        calibration_concat_size=512,
    )

    quantized_dir = tmp_path / f"quantized-{weight_prepare}"
    model.save(str(quantized_dir))
    del model

    reloaded = GPTQModel.load(str(quantized_dir), backend=BACKEND.TORCH, device="cpu")
    if weight_prepare == "random_orthogonal":
        assert _count_ptq_hooks(reloaded) > 0

    eval_device = next(reloaded.model.parameters()).device
    ppl = compute_wikitext_perplexity(
        reloaded,
        reloaded.tokenizer,
        eval_device,
        seq_len=_EVAL_SEQ_LEN,
        n_tokens=_EVAL_N_TOKENS,
    )
    assert math.isfinite(ppl)
    assert ppl > 1.0

    test_snippet = "The history of machine learning begins with early statistical methods."
    batch = reloaded.tokenizer(test_snippet, return_tensors="pt").to(eval_device)
    reloaded.model.eval()
    with torch.no_grad():
        logits = reloaded.model(**batch).logits
    assert torch.isfinite(logits).all()


def test_benchmark_script_runs(tmp_path: Path):
    _require_datasets()
    model_id = _require_benchmark_model_id()

    env = os.environ.copy()
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_random_orthogonal_ppl.py"),
        "--model-id",
        model_id,
        "--device",
        "cpu",
        "--calib-samples",
        "16",
        "--calib-concat-size",
        "512",
        "--eval-seq-len",
        str(_EVAL_SEQ_LEN),
        "--eval-n-tokens",
        str(_EVAL_N_TOKENS),
        "--skip-fp16",
        "--work-dir",
        str(tmp_path / "checkpoints"),
        "--output-json",
        str(tmp_path / "results.json"),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(
            "benchmark_random_orthogonal_ppl.py failed:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    assert (tmp_path / "results.json").is_file()
