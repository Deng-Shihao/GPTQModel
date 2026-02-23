# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# AWQ-only minimal definitions surface.
from .llama import LlamaQModel
from .qwen3 import Qwen3QModel

__all__ = [
    "LlamaQModel",
    "Qwen3QModel",
]
