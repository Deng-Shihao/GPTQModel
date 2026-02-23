
# AWQ-only minimal definitions surface.
from .llama import LlamaQModel
from .qwen3 import Qwen3QModel

__all__ = [
    "LlamaQModel",
    "Qwen3QModel",
]
