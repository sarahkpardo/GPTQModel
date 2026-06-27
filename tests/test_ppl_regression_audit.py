# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PPL regression audit helpers."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gptqmodel.nn_modules.qlinear.torch import TorchLinear  # noqa: E402
from gptqmodel.quantization import FORMAT, METHOD, QuantizeConfig  # noqa: E402
from gptqmodel.utils.random_orthogonal_diag import (  # noqa: E402
    audit_quant_kernel_types,
    compare_inmem_reload_dequant,
    measure_identity_torchlinear_mse,
)
from gptqmodel.utils.model import (  # noqa: E402
    gptqmodel_post_init,
    maybe_convert_gptq_v1_to_v2_runtime,
)
from gptqmodel.utils.wikitext_benchmark import compute_wikitext_perplexity_detailed  # noqa: E402


def _load_benchmark_module():
    script_path = ROOT / "scripts" / "benchmark_random_orthogonal_ppl.py"
    spec = importlib.util.spec_from_file_location("benchmark_random_orthogonal_ppl", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_quant_kernel_types_counts_torch_linear():
    class _Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = TorchLinear(
                bits=4,
                group_size=128,
                sym=True,
                desc_act=False,
                in_features=32,
                out_features=16,
                bias=False,
            )
            self.plain = nn.Linear(8, 4)

    audit = audit_quant_kernel_types(_Wrap())
    assert audit.torch_linear == 1
    assert audit.plain_linear == 1
    assert audit.marlin_linear == 0


def test_compare_inmem_reload_dequant_matches_same_module():
    module_a = TorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=32,
        bias=False,
    )
    module_b = TorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=32,
        bias=False,
    )
    fake_weight = torch.randn(64, 32)

    def _fake_dequant(num_itr=1):
        del num_itr
        return fake_weight

    module_a.dequantize_weight = _fake_dequant  # type: ignore[method-assign]
    module_b.dequantize_weight = _fake_dequant  # type: ignore[method-assign]
    module_a.qweight = torch.empty(0)
    module_b.qweight = torch.empty(0)

    result = compare_inmem_reload_dequant(
        nn.Sequential(module_a),
        nn.Sequential(module_b),
        max_modules=1,
    )
    assert result["modules_compared"] == 1
    assert result["all_match"] is True


def test_measure_identity_torchlinear_mse_runs_with_mocked_quant():
    linear = nn.Linear(32, 16, bias=False)
    quant = TorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=16,
        bias=False,
    )

    def _fake_forward(x):
        return torch.nn.functional.linear(x, linear.weight, linear.bias)

    quant.forward = _fake_forward
    stats = measure_identity_torchlinear_mse(
        linear,
        quant,
        device=torch.device("cpu"),
        batch_size=2,
        seq_len=8,
    )
    assert stats["mean_rel_error"] == pytest.approx(0.0, abs=1e-6)


def test_compute_wikitext_perplexity_detailed_reports_non_finite(monkeypatch):
    class _FakeLoss:
        def __init__(self, value: float):
            self.loss = torch.tensor(value)

    class _FakeModel(nn.Module):
        def __init__(self, losses: list[float]):
            super().__init__()
            self._losses = list(losses)
            self._index = 0
            self.config = type("Cfg", (), {"max_position_embeddings": 2048})()

        def forward(self, chunk, labels=None):
            value = self._losses[min(self._index, len(self._losses) - 1)]
            self._index += 1
            return _FakeLoss(value)

    class _FakeTokenizer:
        pass

    monkeypatch.setattr(
        "gptqmodel.utils.wikitext_benchmark._load_dataset",
        lambda: (lambda *args, **kwargs: {"text": ["hello world"] * 4}),
    )
    monkeypatch.setattr(
        "gptqmodel.utils.wikitext_benchmark._iter_wikitext_sequences",
        lambda *args, **kwargs: [torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)],
    )

    model = _FakeModel([1.0, float("nan")])
    result = compute_wikitext_perplexity_detailed(
        model,
        _FakeTokenizer(),
        torch.device("cpu"),
        seq_len=8,
        n_tokens=16,
    )
    assert result.n_windows == 2
    assert result.all_losses_finite is False
    assert result.non_finite_window_indices == [1]
    assert math.isnan(result.perplexity)


def test_benchmark_kernel_audit_helpers():
    bench = _load_benchmark_module()
    audit = audit_quant_kernel_types(nn.Sequential(nn.Linear(4, 4)))
    payload = bench._kernel_audit_dict(audit)
    assert "torch_linear" in payload
    assert payload["plain_linear"] == 1


def test_torch_linear_respects_disable_compile_env(monkeypatch):
    monkeypatch.setenv("GPTQ_TORCH_DISABLE_COMPILE", "1")

    compile_calls: list[str] = []

    def _fake_compile(fn, **kwargs):
        compile_calls.append(getattr(fn, "__name__", "fn"))
        return fn

    monkeypatch.setattr("gptqmodel.nn_modules.qlinear.torch.torch_compile", _fake_compile)

    module = TorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=16,
        bias=False,
    )
    module.post_init()
    assert compile_calls == []


def test_coerce_quantized_gptq_marlin_dtype(monkeypatch):
    from gptqmodel.models.loader import _coerce_quantized_gptq_marlin_dtype
    from gptqmodel.quantization.config import METHOD, QuantizeConfig
    from gptqmodel.utils.backend import BACKEND

    qcfg = QuantizeConfig(bits=4, group_size=128)
    qcfg.quant_method = METHOD.GPTQ

    monkeypatch.setattr(
        "gptqmodel.models.loader.marlin_runtime_available",
        lambda dtype: dtype == torch.float16,
    )
    coerced = _coerce_quantized_gptq_marlin_dtype(
        backend=BACKEND.GPTQ_MARLIN,
        qcfg=qcfg,
        dtype=torch.bfloat16,
    )
    assert coerced == torch.float16


def test_gptqmodel_post_init_converts_v1_qzeros_to_v2_runtime():
    module = TorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=32,
        bias=False,
    )
    module.register_buffer("qweight", torch.zeros(1, 32, dtype=torch.int32))
    module.register_buffer("qzeros", torch.zeros(1, 32, dtype=torch.int32))
    module.register_buffer("scales", torch.ones(1, 32, dtype=torch.float16))
    module.register_buffer("g_idx", torch.zeros(64, dtype=torch.int32))
    module.qzero_format(format=1)
    v1_qzeros = module.qzeros.data.clone()

    wrapper = nn.Module()
    wrapper.layer = module
    qcfg = QuantizeConfig(bits=4, group_size=128, format=FORMAT.GPTQ, quant_method=METHOD.GPTQ)
    gptqmodel_post_init(wrapper, use_act_order=False, quantize_config=qcfg)

    assert module.qzero_format() == 2
    assert not torch.equal(module.qzeros.data, v1_qzeros)


def test_maybe_convert_gptq_v1_to_v2_runtime_skips_already_v2():
    module = TorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=32,
        bias=False,
    )
    module.register_buffer("qweight", torch.zeros(1, 32, dtype=torch.int32))
    module.register_buffer("qzeros", torch.zeros(1, 32, dtype=torch.int32))
    module.register_buffer("scales", torch.ones(1, 32, dtype=torch.float16))
    module.register_buffer("g_idx", torch.zeros(64, dtype=torch.int32))
    module.qzero_format(format=2)
    v2_qzeros = module.qzeros.data.clone()

    wrapper = nn.Module()
    wrapper.layer = module
    qcfg = QuantizeConfig(bits=4, group_size=128, format=FORMAT.GPTQ, quant_method=METHOD.GPTQ)
    maybe_convert_gptq_v1_to_v2_runtime(wrapper, qcfg)

    assert module.qzero_format() == 2
    assert torch.equal(module.qzeros.data, v2_qzeros)
