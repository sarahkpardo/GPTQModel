# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_FILE = ROOT / "requirements.txt"
SKIP_ENV = "GPTQMODEL_SKIP_DEP_UPGRADE"

# Keep aligned with pyproject.toml [build-system].requires plus pip/wheel minimums.
MIN_BUILD_PACKAGES = (
    "pip>=24.2",
    "setuptools>=77.0.1,<83",
    "wheel>=0.43.0",
    "packaging>=24.2",
)


def parse_requirements(path: Path = REQUIREMENTS_FILE) -> list[Requirement]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dependency file: {path}")

    requirements: list[Requirement] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(Requirement(line))
    return requirements


def requirement_applies(req: Requirement) -> bool:
    return req.marker is None or req.marker.evaluate()


def get_installed_version(name: str) -> str | None:
    try:
        return installed_version(canonicalize_name(name))
    except PackageNotFoundError:
        return None


def is_satisfied(req: Requirement) -> bool:
    if not requirement_applies(req):
        return True

    version = get_installed_version(req.name)
    if version is None:
        return False

    if not req.specifier:
        return True

    return version in SpecifierSet(str(req.specifier))


def collect_outdated(requirements: list[Requirement]) -> list[str]:
    outdated: list[str] = []
    for req in requirements:
        if not requirement_applies(req):
            continue
        if not is_satisfied(req):
            outdated.append(str(req))
    return outdated


def _upgrade_packages(specs: list[str]) -> None:
    if not specs:
        return
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            *specs,
        ]
    )


def pip_check() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "check"])


def ensure_build_tools() -> list[str]:
    if os.environ.get(SKIP_ENV) == "1":
        return []

    outdated = collect_outdated([Requirement(spec) for spec in MIN_BUILD_PACKAGES])
    if outdated:
        _upgrade_packages(outdated)
    return outdated


def ensure_runtime_requirements(
    requirements_file: Path = REQUIREMENTS_FILE,
    *,
    run_pip_check: bool = True,
) -> list[str]:
    if os.environ.get(SKIP_ENV) == "1":
        return []

    outdated = collect_outdated(parse_requirements(requirements_file))
    if outdated:
        _upgrade_packages(outdated)

    if run_pip_check:
        pip_check()

    return outdated
