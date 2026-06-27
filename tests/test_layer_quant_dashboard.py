# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gptqmodel.utils.logger import LayerQuantDashboard, layer_dashboard_enabled


def test_layer_dashboard_enabled_respects_env(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_LAYER_DASHBOARD", "0")
    assert layer_dashboard_enabled() is False


def test_layer_dashboard_render_writes_ansi_refresh(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("GPTQMODEL_LAYER_DASHBOARD", "1")

    lines: list[str] = []

    class _Stdout:
        def isatty(self):
            return True

        def write(self, text):
            lines.append(text)

        def flush(self):
            return None

    monkeypatch.setattr("gptqmodel.utils.logger.sys.stdout", _Stdout())

    dashboard = LayerQuantDashboard()
    rows = [
        {
            "method": "gptq",
            "module": "q_proj",
            "loss": "0.01",
            "nsamples": "128",
            "damp_percent": "0.01000",
            "time": "0.500",
        }
    ]
    dashboard.render(layer_index=0, layer_label="layers.0", rows=rows)
    dashboard.render(layer_index=1, layer_label="layers.1", rows=rows)
    dashboard.close()

    assert any("\033[" in chunk for chunk in lines)
    assert any("Layer 0" in chunk for chunk in lines)
    assert any("Layer 1" in chunk for chunk in lines)
