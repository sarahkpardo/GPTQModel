#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Compare intermediate PTQ artifacts (Hessian H / qr_R) between capture paths.

Usage:
    python scripts/compare_pipeline_intermediates.py
    python scripts/compare_pipeline_intermediates.py --module-name fc2
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib

_ptq_context = importlib.import_module("gptqmodel.ptq.context")
_ptq_gptq_opt = importlib.import_module("gptqmodel.ptq.optimizers.gptq")
_ptq_stats = importlib.import_module("gptqmodel.ptq.stats")
_qcfg = importlib.import_module("gptqmodel.quantization.config")
_gptq = importlib.import_module("gptqmodel.quantization.gptq")

ModuleCalibContext = _ptq_context.ModuleCalibContext
GptqWeightOptimizer = _ptq_gptq_opt.GptqWeightOptimizer
StatisticsCollector = _ptq_stats.StatisticsCollector
HessianConfig = _qcfg.HessianConfig
QuantizeConfig = _qcfg.QuantizeConfig
GPTQ = _gptq.GPTQ


class _TwoLinearChain(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width, bias=False, dtype=torch.float32)
        self.fc2 = nn.Linear(width, width, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def _tensor_fingerprint(t: Optional[torch.Tensor]) -> str:
    if t is None:
        return "none"
    flat = t.detach().to(torch.float32).cpu().contiguous().view(-1)
    digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]
    return f"shape={tuple(t.shape)} norm={flat.norm().item():.6f} sha256={digest}"


def _capture_inline(module: nn.Linear, batches: list[torch.Tensor], qcfg: QuantizeConfig) -> ModuleCalibContext:
    gptq = GPTQ(module, qcfg=qcfg)
    gptq.fallback = None
    gptq.quantizer.configure(perchannel=True)
    for batch in batches:
        out = F.linear(batch, module.weight)
        gptq.add_batch(batch, out)
    gptq.finalize_hessian()
    return ModuleCalibContext(
        module_name=getattr(module, "full_name", "module"),
        columns=module.in_features,
        rows=module.out_features,
        nsamples=gptq.nsamples,
        H=gptq.H.clone(),
        qr_R=gptq._qr_R.clone() if gptq._qr_R is not None else None,
    )


def _capture_collector(module: nn.Linear, batches: list[torch.Tensor], qcfg: QuantizeConfig) -> ModuleCalibContext:
    collector = StatisticsCollector(
        columns=module.in_features,
        hessian=qcfg.hessian,
        row_buffer_max_rows=qcfg.hessian.row_buffer_max_rows,
    )
    for batch in batches:
        collector.add_batch(batch)
    ctx = collector.to_context(
        module_name=getattr(module, "full_name", "module"),
        rows=module.out_features,
    )
    collector.free()
    return ctx


def _sequential_fc2_batches(model: _TwoLinearChain, batches: list[torch.Tensor], qcfg: QuantizeConfig):
    """Quantize fc1 inline, then build fc2 inputs on the partially quantized network."""
    inline_fc1 = GPTQ(model.fc1, qcfg=qcfg)
    inline_fc1.fallback = None
    inline_fc1.quantizer.configure(perchannel=True)
    for batch in batches:
        out = F.linear(batch, model.fc1.weight)
        inline_fc1.add_batch(batch, out)
    wq, *_ = inline_fc1.quantize(blocksize=128)
    inline_fc1.free()
    model.fc1.weight.data = wq

    fc2_batches = []
    with torch.inference_mode():
        for batch in batches:
            fc2_batches.append(F.relu(F.linear(batch, model.fc1.weight)))
    return fc2_batches


def _compare_context(label_a: str, ctx_a: ModuleCalibContext, label_b: str, ctx_b: ModuleCalibContext) -> bool:
    h_match = ctx_a.H is not None and ctx_b.H is not None and torch.allclose(ctx_a.H, ctx_b.H, atol=1e-4, rtol=1e-4)
    qr_match = True
    if ctx_a.qr_R is not None or ctx_b.qr_R is not None:
        qr_match = (
            ctx_a.qr_R is not None
            and ctx_b.qr_R is not None
            and torch.allclose(ctx_a.qr_R, ctx_b.qr_R, atol=1e-4, rtol=1e-4)
        )
    print(f"[{label_a}] H: {_tensor_fingerprint(ctx_a.H)}")
    print(f"[{label_b}] H: {_tensor_fingerprint(ctx_b.H)}")
    print(f"H match: {h_match}")
    if ctx_a.qr_R is not None or ctx_b.qr_R is not None:
        print(f"[{label_a}] qr_R: {_tensor_fingerprint(ctx_a.qr_R)}")
        print(f"[{label_b}] qr_R: {_tensor_fingerprint(ctx_b.qr_R)}")
        print(f"qr_R match: {qr_match}")
    return h_match and qr_match


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PTQ intermediate Hessian captures.")
    parser.add_argument("--module-name", default="fc2", choices=("fc1", "fc2"))
    parser.add_argument("--factorization", default="cholesky", choices=("cholesky", "qr"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        damp_percent=0.01,
        hessian=HessianConfig(factorization=args.factorization, row_buffer_max_rows=128),
    )

    model = _TwoLinearChain()
    model.fc1.full_name = "chain.fc1"
    model.fc2.full_name = "chain.fc2"
    batches = [torch.randn(12, 16), torch.randn(9, 16)]

    if args.module_name == "fc1":
        target = model.fc1
        capture_batches = batches
    else:
        torch.manual_seed(args.seed)
        model_b = _TwoLinearChain()
        model_b.fc1.full_name = "chain.fc1"
        model_b.fc2.full_name = "chain.fc2"
        capture_batches = _sequential_fc2_batches(model_b, batches, qcfg)
        target = model_b.fc2

    inline_ctx = _capture_inline(target, capture_batches, qcfg)
    collector_ctx = _capture_collector(target, capture_batches, qcfg)

    ok = _compare_context("inline", inline_ctx, "collector", collector_ctx)
    if not ok:
        print("FAIL: Hessian intermediates diverged.")
        return 1

    optimizer = GptqWeightOptimizer(qcfg=qcfg)
    split = optimizer.optimize(
        module=target,
        ctx=collector_ctx,
        transform=None,
        device=torch.device("cpu"),
        qcfg=qcfg,
    )
    inline_gptq = GPTQ(target, qcfg=qcfg)
    inline_gptq.H = inline_ctx.H
    inline_gptq.nsamples = inline_ctx.nsamples
    inline_gptq._hessian_dirty = False
    if inline_ctx.qr_R is not None:
        inline_gptq._qr_R = inline_ctx.qr_R
    inline_gptq.quantizer.configure(perchannel=True)
    inline_w, *_ = inline_gptq.quantize(blocksize=128)
    inline_gptq.free()
    backend = split.extra.get("gptq")
    if backend is not None:
        backend.free()

    quant_match = torch.allclose(split.pack_weight, inline_w, atol=1e-4, rtol=1e-4)
    print(f"Quant tensor match: {quant_match}")
    if not quant_match:
        print("FAIL: quant outputs diverged for matched Hessian.")
        return 1

    print(f"PASS: {args.module_name} intermediates and quant outputs align.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
