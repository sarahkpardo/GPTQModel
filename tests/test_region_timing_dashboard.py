# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gptqmodel.utils.logger import QuantizationRegionTimer, RegionTimingDashboard


def test_region_timing_dashboard_render_writes_ansi_refresh(monkeypatch):
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

    dashboard = RegionTimingDashboard()
    rows = [["Process quant", "3", "0.100", "0.120", "0.360", "60.0%", "q_proj"]]
    dashboard.render(title="Region timing — Layer 0 (layers.0)", rows=rows)
    dashboard.render(title="Region timing — Layer 1 (layers.1)", rows=rows)
    dashboard.close()

    assert any("\033[" in chunk for chunk in lines)
    assert any("Region timing" in chunk for chunk in lines)
    assert any("Process quant" in chunk for chunk in lines)


def test_region_timer_writes_json_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_LAYER_DASHBOARD", "0")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_region_timing_dashboard.py::test")
    monkeypatch.chdir(tmp_path)

    timer = QuantizationRegionTimer()
    timer.record("process_quant", 0.5, source="model.layers.0.q_proj")
    timer.record("pre_quant_forward", 0.3, source="model.layers.0:subset1/1")
    timer.flush(layer_index=0, layer_label="layers.0")

    assert timer._log_file is not None
    assert timer._log_file.exists()
    snapshots = [json.loads(block) for block in timer._log_file.read_text(encoding="utf-8").split("\n\n") if block.strip()]
    assert len(snapshots) == 1
    assert snapshots[0]["layer_index"] == 0
    assert "process_quant" in snapshots[0]["regions"]

    summary_path = timer.write_full_log_summary()
    assert summary_path is not None
    summary_text = Path(summary_path).read_text(encoding="utf-8")
    assert "Process quant" in summary_text
    assert "Region timing summary" in summary_text
