# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from setuptools import build_meta as _orig


def _ensure_build_dependencies(*, runtime: bool) -> None:
    from build_support.deps import ensure_build_tools, ensure_runtime_requirements

    ensure_build_tools()
    if runtime:
        ensure_runtime_requirements()


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _ensure_build_dependencies(runtime=True)
    return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _ensure_build_dependencies(runtime=True)
    return _orig.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _ensure_build_dependencies(runtime=True)
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _ensure_build_dependencies(runtime=True)
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _ensure_build_dependencies(runtime=False)
    return _orig.build_sdist(sdist_directory, config_settings)


def get_requires_for_build_wheel(config_settings=None):
    return _orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _orig.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _orig.get_requires_for_build_editable(config_settings)
