
import os


# isort: off
from .utils.nogil_patcher import patch_safetensors_save_file, patch_triton_autotuner  # noqa: E402
# isort: on

patch_safetensors_save_file()

# TODO: waiting for official fix from triton
#  monkeypatching triton threading issues is fragile
try:
    patch_triton_autotuner()
except Exception:
    pass

from .utils.env import env_flag
from .utils.logger import setup_logger
from .utils.modelscope import ensure_modelscope_available


DEBUG_ON = env_flag("DEBUG")

from .utils.linalg_warmup import run_torch_linalg_warmup
from .utils.threadx import DeviceThreadPool


DEVICE_THREAD_POOL = DeviceThreadPool(
    inference_mode=True,
    warmups={
        "cuda": run_torch_linalg_warmup,
        "xpu": run_torch_linalg_warmup,
        "mps": run_torch_linalg_warmup,
        "cpu": run_torch_linalg_warmup,
    },
    workers={
        "cuda:per": 4,
        "xpu:per": 1,
        "mps": 8,
        "cpu": min(12, max(1, (os.cpu_count() or 1) + 1 // 2)), # count + 1, fixed pool size > 1 check when count=3
        "model_loader:cpu": 2,
    },
    empty_cache_every_n=512,
)

from .models import AweQuant, get_best_device
from .models.auto import ASCII_LOGO
from .quantization import BaseQuantizeConfig, QuantizeConfig
from .utils import BACKEND
from .version import __version__


def exllama_set_max_input_length(*_args, **_kwargs):
    raise NotImplementedError("AWQ-only build does not support ExLlama backends.")


setup_logger().info("\n%s", ASCII_LOGO)


if ensure_modelscope_available():
    try:
        from modelscope.utils.hf_util.patcher import patch_hub
        patch_hub()
    except Exception as exc:
        raise ModuleNotFoundError(
            "env `AWEQUANT_USE_MODELSCOPE` used but modelscope pkg is not found: please install with `pip install modelscope`."
        ) from exc
