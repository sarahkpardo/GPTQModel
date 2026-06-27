# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from .quantizer_processor import QuantizerProcessor, clone_gptq_config_for_module

# Deprecated alias — production GPTQ uses SequentialPTQProcessor.
GPTQProcessor = QuantizerProcessor

__all__ = ["GPTQProcessor", "QuantizerProcessor", "clone_gptq_config_for_module"]
