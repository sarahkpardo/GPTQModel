# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packaging.requirements import Requirement

from build_support.deps import (  # noqa: E402
    collect_outdated,
    is_satisfied,
    parse_requirements,
    requirement_applies,
)


def test_parse_requirements_reads_markers(tmp_path: Path):
    req_file = tmp_path / "requirements.txt"
    req_file.write_text(
        "\n".join(
            [
                "numpy==2.2.6; python_version < '3.14'",
                "numpy>=2.3.0; python_version >= '3.14'",
                "pypcre>=0.3.2",
            ]
        ),
        encoding="utf-8",
    )

    requirements = parse_requirements(req_file)
    assert len(requirements) == 3
    assert requirements[0].name == "numpy"
    assert requirements[2].name == "pypcre"


def test_requirement_applies_respects_python_version_marker():
    req = Requirement("numpy==2.2.6; python_version < '3.14'")
    assert requirement_applies(req) is (sys.version_info < (3, 14))


def test_is_satisfied_when_version_meets_lower_bound(monkeypatch):
    req = Requirement("transformers>=5.4.0")

    monkeypatch.setattr(
        "build_support.deps.get_installed_version",
        lambda name: "5.4.0" if name == "transformers" else None,
    )

    assert is_satisfied(req) is True


def test_is_satisfied_when_version_below_lower_bound(monkeypatch):
    req = Requirement("pypcre>=0.3.2")

    monkeypatch.setattr(
        "build_support.deps.get_installed_version",
        lambda name: "0.3.1" if name == "pypcre" else None,
    )

    assert is_satisfied(req) is False


def test_is_satisfied_when_package_missing(monkeypatch):
    req = Requirement("pypcre>=0.3.2")
    monkeypatch.setattr("build_support.deps.get_installed_version", lambda name: None)
    assert is_satisfied(req) is False


def test_collect_outdated_flags_missing_pypcre(monkeypatch):
    requirements = [Requirement("pypcre>=0.3.2"), Requirement("packaging>=24.2")]

    def fake_version(name: str):
        return {"packaging": "24.2"}[name]

    monkeypatch.setattr("build_support.deps.get_installed_version", fake_version)
    outdated = collect_outdated(requirements)

    assert outdated == ["pypcre>=0.3.2"]


def test_collect_outdated_skips_satisfied_packages(monkeypatch):
    requirements = [Requirement("pypcre>=0.3.2"), Requirement("packaging>=24.2")]

    def fake_version(name: str):
        return {"pypcre": "0.3.2", "packaging": "24.2"}[name]

    monkeypatch.setattr("build_support.deps.get_installed_version", fake_version)
    assert collect_outdated(requirements) == []
