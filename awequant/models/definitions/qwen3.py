
from .llama import LlamaQModel


class Qwen3QModel(LlamaQModel):
    """
    Qwen3 inherits the Llama-style layout but inserts Q/K RMS norm layers
    ahead of the attention projections. We mark those helper modules as
    non-quantized so the layer walker captures the complete structure.
    """

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:!",
                "k_norm:!",
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]
