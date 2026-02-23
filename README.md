# GPTQModel (AWQ-Only)

This fork is intentionally simplified to support **AWQ quantization only**.

## Scope

- Supported quantization method: `METHOD.AWQ`
- Supported checkpoint formats: `FORMAT.GEMM`, `FORMAT.GEMV`, `FORMAT.GEMV_FAST`, `FORMAT.LLM_AWQ`
- Unsupported in this fork: GPTQ, QQQ, GPTAQ, EoRA, vLLM/SGLang/MLX integration extras

## Install

```bash
uv pip install -v . --no-build-isolation
```

## AWQ Quantization Example

```python
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization import FORMAT, METHOD

model_id = "Qwen/Qwen3-4B"
quant_path = "./quantized_models/Qwen3-4B-awq-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(1024))["text"]

quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    quant_method=METHOD.AWQ,
    format=FORMAT.GEMM,
    sym=False,
)

model = GPTQModel.load(model_id, quantize_config=quant_config)
model.quantize(calibration_dataset, batch_size=2)
model.save(quant_path)
```

## Notes

- `QuantizeConfig` defaults are now AWQ-focused (`quant_method=METHOD.AWQ`, `format=FORMAT.GEMM`).
- Passing non-AWQ quantization methods raises a configuration error.
- Build defaults only compile AWQ extensions; non-AWQ kernels are opt-in via environment variables.
