# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Calibration token expectations vs per-module activation row capture."""

from __future__ import annotations


def expected_calibration_tokens(processor) -> int:
    """Return total calibration tokens derived from the processor calibration dataset."""
    total = getattr(processor, "total_calibration_tokens", None)
    if total is None or int(total) <= 0:
        raise ValueError(
            "Calibration token count is unavailable; ensure the processor received a non-empty calibration dataset."
        )
    return int(total)


def calibration_coverage_ratio(*, observed_rows: int, expected_tokens: int) -> float:
    if expected_tokens <= 0:
        return 0.0
    return float(observed_rows) / float(expected_tokens)


def describe_calibration_coverage(
    *,
    module_name: str,
    observed_rows: int,
    expected_tokens: int,
) -> str:
    """Explain when observed activation rows differ from expected calibration tokens."""
    if expected_tokens <= 0:
        return (
            f"Module `{module_name}` captured {observed_rows} activation rows but "
            f"expected_calibration_tokens is unset."
        )

    ratio = calibration_coverage_ratio(observed_rows=observed_rows, expected_tokens=expected_tokens)
    ratio_pct = f"{ratio:.1%}"

    if observed_rows == expected_tokens:
        return (
            f"Module `{module_name}` observed_rows={observed_rows} matches "
            f"expected_calibration_tokens={expected_tokens}."
        )

    reasons = []
    if observed_rows < expected_tokens:
        reasons.extend(
            [
                "MoE or sparse routing may deliver fewer tokens to this module than the global calibration total",
                "the module may not be invoked on every calibration forward/subset",
                "attention-mask filtering counts only kept token positions",
            ]
        )
    else:
        reasons.extend(
            [
                "conv/unfold reshaping or repeated forwards can increase row counts above token totals",
                "batching may duplicate rows relative to a flat token count",
            ]
        )

    reason_text = "; ".join(reasons)
    return (
        f"Module `{module_name}` observed_rows={observed_rows} differs from "
        f"expected_calibration_tokens={expected_tokens} (coverage={ratio_pct}). "
        f"Dense full-path modules usually match; differences are normal for sparse/MoE paths. "
        f"Likely causes: {reason_text}."
    )


def format_coverage_stat(*, observed_rows: int, expected_tokens: int) -> str:
    if expected_tokens <= 0:
        return "n/a"
    return f"{calibration_coverage_ratio(observed_rows=observed_rows, expected_tokens=expected_tokens):.1%}"
