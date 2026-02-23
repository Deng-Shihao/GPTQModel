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

# Increase `batch_size` based on your GPU VRAM for faster quantization.
model.quantize(calibration_dataset, batch_size=2)
model.save(quant_path)
