#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_support.deps import (  # noqa: E402
    MIN_BUILD_PACKAGES,
    REQUIREMENTS_FILE,
    Requirement,
    collect_outdated,
    ensure_build_tools,
    ensure_runtime_requirements,
    parse_requirements,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upgrade GPTQModel dependencies that are below required lower bounds.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Also upgrade build tools (pip, setuptools, wheel, packaging).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print outdated packages without installing.",
    )
    parser.add_argument(
        "--requirements-file",
        type=Path,
        default=REQUIREMENTS_FILE,
        help="Requirements file to validate (default: requirements.txt).",
    )
    parser.add_argument(
        "--skip-pip-check",
        action="store_true",
        help="Skip pip check after upgrading runtime dependencies.",
    )
    args = parser.parse_args()

    runtime_reqs = parse_requirements(args.requirements_file)
    outdated_runtime = collect_outdated(runtime_reqs)
    outdated_build: list[str] = []
    if args.build:
        outdated_build = collect_outdated([Requirement(spec) for spec in MIN_BUILD_PACKAGES])

    if args.dry_run:
        if outdated_build:
            print("Outdated build tools:")
            for spec in outdated_build:
                print(f"  - {spec}")
        if outdated_runtime:
            print("Outdated runtime dependencies:")
            for spec in outdated_runtime:
                print(f"  - {spec}")
        if not outdated_build and not outdated_runtime:
            print("All dependencies satisfy required lower bounds.")
        return 0

    if args.build:
        ensure_build_tools()
    ensure_runtime_requirements(
        args.requirements_file,
        run_pip_check=not args.skip_pip_check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
