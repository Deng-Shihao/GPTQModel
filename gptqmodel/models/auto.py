# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os

from ..utils.logger import setup_logger


log = setup_logger()

ASCII_LOGO = r"""
_____/\\\\\\\\\\\\__/\\\\\\\\\\\\\____/\\\\\\\\\\\\\\\______________________/\\\________/\\\\____________/\\\\_______________________/\\\__________________/\\\\\\____
 ___/\\\//////////__\/\\\/////////\\\_\///////\\\/////____________________/\\\\/\\\\____\/\\\\\\________/\\\\\\______________________\/\\\_________________\////\\\____
  __/\\\_____________\/\\\_______\/\\\_______\/\\\_______________________/\\\//\////\\\__\/\\\//\\\____/\\\//\\\______________________\/\\\____________________\/\\\____
   _\/\\\____/\\\\\\\_\/\\\\\\\\\\\\\/________\/\\\________/\\\\\\\\\\\__/\\\______\//\\\_\/\\\\///\\\/\\\/_\/\\\_____/\\\\\___________\/\\\______/\\\\\\\\_____\/\\\____
    _\/\\\___\/////\\\_\/\\\/////////__________\/\\\_______\///////////__\//\\\______/\\\__\/\\\__\///\\\/___\/\\\___/\\\///\\\____/\\\\\\\\\____/\\\/////\\\____\/\\\____
     _\/\\\_______\/\\\_\/\\\___________________\/\\\______________________\///\\\\/\\\\/___\/\\\____\///_____\/\\\__/\\\__\//\\\__/\\\////\\\___/\\\\\\\\\\\_____\/\\\____
      _\/\\\_______\/\\\_\/\\\___________________\/\\\________________________\////\\\//_____\/\\\_____________\/\\\_\//\\\__/\\\__\/\\\__\/\\\__\//\\///////______\/\\\____
       _\//\\\\\\\\\\\\/__\/\\\___________________\/\\\___________________________\///\\\\\\__\/\\\_____________\/\\\__\///\\\\\/___\//\\\\\\\/\\__\//\\\\\\\\\\__/\\\\\\\\\_
        __\////////////____\///____________________\///______________________________\//////___\///______________\///_____\/////______\///////\//____\//////////__\/////////__
"""

# if not os.environ.get("PYTHON_GIL", None):
#     os.environ["PYTHON_GIL"] = '0'
#     log.info("ENV: Auto disable GIL and use free-threading mode when applicable: Python 3.13t+. You must install the -t edition of Python.")

if not os.environ.get("PYTORCH_ALLOC_CONF", None):
    os.environ["PYTORCH_ALLOC_CONF"] = 'expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7'
    log.info("ENV: Auto setting PYTORCH_ALLOC_CONF='expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7' for memory saving.")

if not os.environ.get("CUDA_DEVICE_ORDER", None):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    log.info("ENV: Auto setting CUDA_DEVICE_ORDER=PCI_BUS_ID for correctness.")

# FIX ROCm env conflict with CUDA_VISIBLE_DEVICES if both exits
if 'CUDA_VISIBLE_DEVICES' in os.environ and 'ROCR_VISIBLE_DEVICES' in os.environ:
    del os.environ['ROCR_VISIBLE_DEVICES']

# if not os.environ.get("NCCL_SHM_DISABLE", None):
#     os.environ["NCCL_SHM_DISABLE"] = '1'
#     log.info("ENV: Auto setting NCCL_SHM_DISABLE=1 for multi-gpu memory safety.")

import sys  # noqa: E402


# TODO: waiting for pytorch implementgation of aten ops for MPS
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import os.path  # noqa: E402
import random  # noqa: E402
from os.path import isdir, join  # noqa: E402
from typing import Dict, Optional, Union  # noqa: E402

import numpy  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import list_repo_files  # noqa: E402
from transformers import AutoConfig  # noqa: E402

from ..quantization import METHOD, QUANT_CONFIG_FILENAME  # noqa: E402
from ..utils import BACKEND  # noqa: E402
from .base import BaseQModel, QuantizeConfig  # noqa: E402
from .definitions.llama import LlamaQModel  # noqa: E402
from .definitions.qwen3 import Qwen3QModel  # noqa: E402


# make quants and inference more determinisitc
torch.manual_seed(787)
random.seed(787)
numpy.random.seed(787)

MODEL_MAP = {
    "llama": LlamaQModel,
    "qwen3": Qwen3QModel,
}

SUPPORTED_MODELS = list(MODEL_MAP.keys())


def _is_supported_quantization_config(config: AutoConfig) -> bool:
    quantization_config = getattr(config, "quantization_config", None)
    if not isinstance(quantization_config, dict):
        return False

    quant_format = quantization_config.get("quant_format")
    if isinstance(quant_format, str) and quant_format.lower() == METHOD.AWQ:
        return True

    quant_method = quantization_config.get("quant_method")
    if isinstance(quant_method, str) and quant_method.lower() == METHOD.AWQ:
        return True

    return False


def check_and_get_model_definition(model_dir, trust_remote_code=False):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    model_type = config.model_type.lower()

    # if model_type is not supported, use BaseQModel, will use auto_detect_module_tree to generate module tree
    if model_type not in SUPPORTED_MODELS:
        return BaseQModel

    return MODEL_MAP[model_type]

class GPTQModel:
    def __init__(self):
        raise EnvironmentError(
            "GPTQModel is not designed to be instantiated\n"
            "use `GPTQModel.from_pretrained` to load pretrained model and prepare for quantization via `.quantize()`.\n"
            "use `GPTQModel.from_quantized` to inference with post-quantized model."
        )

    @classmethod
    def load(
            cls,
            model_id_or_path: Optional[str],
            quantize_config: Optional[QuantizeConfig | Dict] = None,
            device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
            device: Optional[Union[str, torch.device]] = None,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            trust_remote_code: bool = False,
            **kwargs,
    ):
        if isinstance(model_id_or_path, str):
            model_id_or_path = model_id_or_path.strip()

        # normalize config to cfg instance
        if isinstance(quantize_config, Dict):
            quantize_config = QuantizeConfig(**quantize_config)

        if isinstance(backend, str):
            backend = BACKEND(backend)

        is_gptqmodel_quantized = False
        model_cfg = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
        if _is_supported_quantization_config(model_cfg):
            # only if the model is quantized or compatible with gptqmodel should we set is_quantized to true
            is_gptqmodel_quantized = True
        else:
            # TODO FIX ME...not decoded to check if quant method is compatible or quantized by gptqmodel
            for name in [QUANT_CONFIG_FILENAME, "quant_config.json"]:
                if isdir(model_id_or_path):  # Local
                    if os.path.exists(join(model_id_or_path, name)):
                        is_gptqmodel_quantized = True
                        break

                else:  # Remote
                    files = list_repo_files(repo_id=model_id_or_path)
                    for f in files:
                        if f == name:
                            is_gptqmodel_quantized = True
                            break

        if is_gptqmodel_quantized:
            m = cls.from_quantized(
                model_id_or_path=model_id_or_path,
                device_map=device_map,
                device=device,
                backend=backend,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        else:
            m = cls.from_pretrained(
                model_id_or_path=model_id_or_path,
                quantize_config=quantize_config,
                device_map=device_map,
                device=device,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        # debug model structure
        # if debug:
        #     print_module_tree(m.model)

        return m


    @classmethod
    def from_pretrained(
            cls,
            model_id_or_path: str,
            quantize_config: QuantizeConfig,
            trust_remote_code: bool = False,
            **model_init_kwargs,
    ) -> BaseQModel:
        config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
        if _is_supported_quantization_config(config):
            log.warn("Model is already quantized, will use `from_quantized` to load quantized model.\n"
                           "If you want to quantize the model, please pass un_quantized model path or id, and use "
                           "`from_pretrained` with `quantize_config`.")
            return cls.from_quantized(model_id_or_path, trust_remote_code=trust_remote_code)

        if quantize_config and quantize_config.dynamic:
            log.warn(
                "GPTQModel's per-module `dynamic` quantization feature is fully supported in latest vLLM and SGLang but not yet available in hf transformers.")

        model_definition = check_and_get_model_definition(model_id_or_path, trust_remote_code)

        return model_definition.from_pretrained(
            pretrained_model_id_or_path=model_id_or_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
            cls,
            model_id_or_path: Optional[str],
            device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
            device: Optional[Union[str, int]] = None,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            trust_remote_code: bool = False,
            **kwargs,
    ) -> BaseQModel:
        model_definition = check_and_get_model_definition(model_id_or_path, trust_remote_code)

        if isinstance(backend, str):
            backend = BACKEND(backend)

        return model_definition.from_quantized(
            model_id_or_path=model_id_or_path,
            device_map=device_map,
            device=device,
            backend=backend,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @classmethod
    def eval(cls, *args, **kwargs):
        raise NotImplementedError("AWQ-only quantization build does not include evaluation helpers.")

    @staticmethod
    def export(*args, **kwargs):
        raise NotImplementedError("AWQ-only quantization build does not include model export helpers.")

    @staticmethod
    def push_to_hub(*args, **kwargs):
        raise NotImplementedError("AWQ-only quantization build does not include Hub upload helpers.")
