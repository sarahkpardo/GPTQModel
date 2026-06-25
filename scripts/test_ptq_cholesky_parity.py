#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Compare legacy GPTQ vs PTQ split (statistics + transform + GPTQ) using Cholesky Hessian.

This script avoids the QR factorization path so you can validate the transform/optimizer
refactor independently of QR-GPTQ numerics.

Usage:
    python scripts/test_ptq_cholesky_parity.py
    python scripts/test_ptq_cholesky_parity.py --seed 0 --batches 8 --tokens 64
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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

from gptqmodel.looper.named_module import NamedModule  # noqa: E402
from gptqmodel.ptq.config import TransformPrepareConfig  # noqa: E402
from gptqmodel.ptq.stats import StatisticsCollector  # noqa: E402
from gptqmodel.ptq.transforms.registry import build_transform_backend  # noqa: E402
from gptqmodel.quantization.config import HessianConfig, QuantizeConfig  # noqa: E402
from gptqmodel.quantization.gptq import GPTQ  # noqa: E402


@dataclass
class QuantResult:
    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    g_idx: torch.Tensor
    avg_loss: float


def _cholesky_qcfg(*, with_ptq_identity: bool) -> QuantizeConfig:
    hessian = HessianConfig(
        factorization="cholesky",
        row_buffer_max_rows=2048,
    )
    cfg = QuantizeConfig(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        damp_percent=0.01,
        damp_auto_increment=0.01,
        hessian=hessian,
    )
    if with_ptq_identity:
        cfg.weight_prepare = [{"method": "identity"}]
    return cfg


def _make_module(
    *,
    in_features: int,
    out_features: int,
    seed: int,
    device: torch.device,
) -> NamedModule:
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=False).to(device=device, dtype=torch.float32)
    return NamedModule(
        module=linear,
        name="proj",
        full_name="layer.0.proj",
        layer_index=0,
    )


def _make_calib(
    *,
    batches: int,
    tokens: int,
    in_features: int,
    seed: int,
    device: torch.device,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    return [
        torch.randn(tokens, in_features, generator=generator, device=device, dtype=torch.float32)
        for _ in range(batches)
    ]


def _run_legacy_gptq(
    named: NamedModule,
    calib: list[torch.Tensor],
    qcfg: QuantizeConfig,
    *,
    blocksize: int,
) -> QuantResult:
    gptq = GPTQ(named, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    for batch in calib:
        out = F.linear(batch, named.module.weight)
        gptq.add_batch(batch, out)
    qweight, scales, zeros, g_idx, _duration, avg_loss, _damp, _nsamples = gptq.quantize(
        blocksize=blocksize,
    )
    gptq.free()
    return QuantResult(
        qweight=qweight.detach().cpu(),
        scales=scales.detach().cpu(),
        zeros=zeros.detach().cpu(),
        g_idx=g_idx.detach().cpu(),
        avg_loss=float(avg_loss),
    )


def _run_ptq_split_gptq(
    named: NamedModule,
    calib: list[torch.Tensor],
    qcfg: QuantizeConfig,
    *,
    blocksize: int,
) -> QuantResult:
    rows, columns = named.module.weight.shape
    collector = StatisticsCollector(
        columns=columns,
        hessian=qcfg.hessian,
        row_buffer_max_rows=qcfg.hessian.row_buffer_max_rows,
    )
    for batch in calib:
        collector.add_batch(batch)
    ctx = collector.to_context(module_name=named.full_name, rows=rows)

    transform_cfg = TransformPrepareConfig(method="identity")
    transform = build_transform_backend(transform_cfg)
    transform_state = transform.fit(
        weight=named.module.weight.data,
        bias=None,
        ctx=ctx,
        mode="standalone",
        device=named.module.weight.device,
    )
    named.module.weight.data = transform.apply_to_weights(
        named.module.weight.data,
        transform_state,
        device=named.module.weight.device,
    )
    ctx.transform = transform_state
    collector.free()

    gptq = GPTQ(named, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    for batch in calib:
        out = F.linear(batch, named.module.weight)
        gptq.add_batch(batch, out)
    qweight, scales, zeros, g_idx, _duration, avg_loss, _damp, _nsamples = gptq.quantize(
        blocksize=blocksize,
    )
    gptq.free()
    return QuantResult(
        qweight=qweight.detach().cpu(),
        scales=scales.detach().cpu(),
        zeros=zeros.detach().cpu(),
        g_idx=g_idx.detach().cpu(),
        avg_loss=float(avg_loss),
    )


def _report_diff(label: str, left: torch.Tensor, right: torch.Tensor) -> None:
    if left.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        mismatches = int((left != right).sum().item())
        print(f"{label}: mismatches={mismatches}, equal={torch.equal(left, right)}")
        return

    diff = (left - right).abs()
    print(
        f"{label}: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}, "
        f"allclose={torch.allclose(left, right, atol=0, rtol=0)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare legacy GPTQ vs PTQ split using Cholesky Hessian factorization.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in-features", type=int, default=128)
    parser.add_argument("--out-features", type=int, default=64)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    calib = _make_calib(
        batches=args.batches,
        tokens=args.tokens,
        in_features=args.in_features,
        seed=args.seed,
        device=device,
    )

    legacy_module = _make_module(
        in_features=args.in_features,
        out_features=args.out_features,
        seed=args.seed,
        device=device,
    )
    ptq_module = _make_module(
        in_features=args.in_features,
        out_features=args.out_features,
        seed=args.seed,
        device=device,
    )

    legacy_cfg = _cholesky_qcfg(with_ptq_identity=False)
    ptq_cfg = _cholesky_qcfg(with_ptq_identity=True)
    assert legacy_cfg.hessian.factorization == "cholesky"
    assert ptq_cfg.hessian.factorization == "cholesky"
    assert ptq_cfg.uses_ptq_transform_pipeline()

    print("Running legacy GPTQ (Cholesky)...")
    legacy = _run_legacy_gptq(
        legacy_module,
        calib,
        legacy_cfg,
        blocksize=args.blocksize,
    )
    print("Running PTQ split (statistics + identity transform + GPTQ, Cholesky)...")
    ptq = _run_ptq_split_gptq(
        ptq_module,
        calib,
        ptq_cfg,
        blocksize=args.blocksize,
    )

    print()
    print("=== Parity report ===")
    _report_diff("qweight", legacy.qweight, ptq.qweight)
    _report_diff("scales", legacy.scales, ptq.scales)
    _report_diff("zeros", legacy.zeros, ptq.zeros)
    _report_diff("g_idx", legacy.g_idx.to(torch.int64), ptq.g_idx.to(torch.int64))
    print(f"avg_loss: legacy={legacy.avg_loss:.8f}, ptq={ptq.avg_loss:.8f}")

    eval_inp = _make_calib(
        batches=1,
        tokens=8,
        in_features=args.in_features,
        seed=args.seed + 99,
        device=device,
    )[0]
    legacy_out = F.linear(eval_inp, legacy.qweight.to(device))
    ptq_out = F.linear(eval_inp, ptq.qweight.to(device))
    _report_diff("forward(eval)", legacy_out, ptq_out)

    ok = (
        torch.equal(legacy.qweight, ptq.qweight)
        and torch.equal(legacy.scales, ptq.scales)
        and torch.equal(legacy.zeros, ptq.zeros)
        and torch.equal(legacy.g_idx, ptq.g_idx)
    )
    print()
    if ok:
        print("PASS: PTQ split matches legacy GPTQ for identity transform + Cholesky Hessian.")
        return 0

    print("FAIL: PTQ split diverged from legacy GPTQ.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
