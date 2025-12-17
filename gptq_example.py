# PYTHON_GIL=0 python gptq_example.py

from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "Qwen/Qwen3-0.6B"
quant_path = "./quantized_models/Qwen3-0.6B-gptq-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=24)

model.save(quant_path)