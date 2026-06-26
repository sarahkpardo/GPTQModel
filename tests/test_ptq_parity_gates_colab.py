# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Colab parity gate checklist for the three-component PTQ refactor."""

from __future__ import annotations

import pytest

COLAB_PARITY_COMMANDS = [
    "python scripts/compare_pipeline_intermediates.py",
    "python scripts/compare_paroquant_transform_intermediates.py --rotation-epochs 1 --finetune-epochs 0",
    "python scripts/quantize_small_model_smoke.py --compare",
    "pytest tests/test_ptq_pipeline.py tests/test_moe_ptq_smoke.py tests/test_ptq_three_component_refactor.py tests/test_paroquant_transform_parity.py",
]


@pytest.mark.colab
@pytest.mark.parametrize("command", COLAB_PARITY_COMMANDS)
def test_colab_parity_gate_command_documented(command: str):
    """Document the post-phase validation commands for Colab runs."""
    assert command
