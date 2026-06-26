#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Compare ParoQuantTransform.fit payload against a direct optimizer reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib

import torch

_parity = importlib.import_module("gptqmodel.ptq.transforms.paroquant_parity")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--out-features", type=int, default=8)
    parser.add_argument("--train-rows", type=int, default=160)
    parser.add_argument("--rotation-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--krot", type=int, default=2)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--module-name", type=str, default="layer.0.linear")
    parser.add_argument(
        "--reference",
        choices=("direct",),
        default="direct",
        help="Reference path (direct optimize_paroquant_linear only for now).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    linear, _ = _parity.make_synthetic_linear(
        in_features=args.in_features,
        out_features=args.out_features,
        seed=args.seed,
    )
    torch.manual_seed(args.seed)
    inputs = torch.randn(args.train_rows, args.in_features, dtype=torch.float32)

    cfg = _parity.make_paroquant_prepare_config(
        options={
            "opt_rotation_epochs": args.rotation_epochs,
            "opt_finetune_epochs": args.finetune_epochs,
            "krot": args.krot,
            "opt_seed": args.seed,
            "opt_layer_index": args.layer_index,
            "opt_module_seed_key": args.module_name,
            "opt_scope": "module",
            "group_size": args.group_size,
            "opt_train_samples": min(128, args.train_rows),
            "opt_validation_samples": min(32, max(1, args.train_rows // 4)),
            "opt_batch_size": 32,
        }
    )
    ctx = _parity.make_calibration_context(
        module_name=args.module_name,
        columns=linear.in_features,
        rows=linear.out_features,
        inputs=inputs,
    )

    reference = _parity.run_paroquant_reference(
        weight=linear.weight.data,
        bias=None,
        ctx=ctx,
        cfg=cfg,
    )
    state = _parity.run_paroquant_transform(
        weight=linear.weight.data,
        bias=None,
        ctx=ctx,
        cfg=cfg,
    )
    subject = _parity.payload_from_state(state)

    print(f"reference={args.reference} module={args.module_name}")
    ok, diffs = _parity.compare_paroquant_payloads(reference, subject)
    for key in _parity.PAROQUANT_PAYLOAD_KEYS:
        ref_value = reference.get(key)
        sub_value = subject.get(key)
        if isinstance(ref_value, torch.Tensor):
            print(f"{key}: ref={_parity.tensor_fingerprint(ref_value)}")
            print(f"{key}: sub={_parity.tensor_fingerprint(sub_value)}")
            max_diff = (ref_value.to(torch.float32) - sub_value.to(torch.float32)).abs().max().item()
            print(f"{key}: max_abs_diff={max_diff:.6e}")
        else:
            print(f"{key}: ref={ref_value} sub={sub_value}")

    if not ok:
        print("MISMATCH:")
        for line in diffs:
            print(f"  - {line}")
        return 1

    print("PASS: ParoQuantTransform payload matches direct optimizer reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
