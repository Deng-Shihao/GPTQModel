# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from enum import Enum


class BACKEND(str, Enum):
    AUTO = "auto"  # choose the optimal local kernel based on quant_config compatibility
    AUTO_TRAINABLE = "auto_trainable" # choose the optimal trainable local kernel for post-quant training

    # awq backends
    GEMM = "gemm"
    GEMM_TRITON = "gemm_triton"
    GEMV = "gemv"
    GEMV_FAST = "gemv_fast"
    TORCH_AWQ = "torch_awq"
