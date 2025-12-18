# PYTHON_GIL=0 python lora_example.py
#
# EoRA (Error-oriented Low-Rank Adaptation) 示例
# 本示例展示如何使用 GPTQModel 的 EoRA 功能来提升量化模型的精度
#
# 使用方法:
#   1. 生成 EoRA adapter: python lora_example.py --mode generate
#   2. 加载并推理:        python lora_example.py --mode inference
#   3. 同时生成和推理:    python lora_example.py --mode both

import argparse
import os

from datasets import load_dataset

from gptqmodel import GPTQModel
from gptqmodel.adapter.adapter import Lora


# ============================================
# 1. 配置参数
# ============================================
MODEL_ID = "Qwen/Qwen3-0.6B"  # 原始全精度模型
QUANT_PATH = "./quantized_models/Qwen3-0.6B-gptq-4bit"  # 已量化模型路径
EORA_RANK = 32  # EoRA rank: 更高的rank提升精度但增加显存占用
                # 建议先测试 32 或 64，再尝试 128 或 256


# ============================================
# 2. 构建 Calibration 数据集
# ============================================
def construct_c4_dataset(nsamples: int = 1024):
    """
    使用 C4 数据集构建 calibration 数据
    C4 是通用的英文文本数据集，适合一般场景的 EoRA 训练
    """
    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train"
    ).select(range(nsamples))["text"]
    return calibration_dataset


def construct_mmlu_dataset():
    """
    使用 MMLU 数据集构建 calibration 数据
    MMLU 是多选题数据集，适合需要问答任务的 EoRA 训练
    """
    def format_question(question, choices, answer):
        return f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}"

    mmlu_dataset = load_dataset('cais/mmlu', 'all', split='validation')
    dataset = []
    for example in mmlu_dataset:
        question = example['question']
        choices = example['choices']
        answer = ['A', 'B', 'C', 'D'][example['answer']]
        dataset.append(format_question(question, choices, answer))

    # 混合一些 C4 数据避免过拟合
    c4_data = construct_c4_dataset(nsamples=1024)
    return dataset + list(c4_data)


# ============================================
# 3. 生成 EoRA Adapter
# ============================================
def generate_eora(model_id: str, quant_path: str, eora_save_path: str, rank: int = 32, dataset_type: str = "c4"):
    """
    为已量化的模型生成 EoRA adapter
    
    Args:
        model_id: 原始全精度模型的 ID 或路径
        quant_path: 已量化模型的路径
        eora_save_path: EoRA adapter 保存路径
        rank: EoRA 的 rank 值
        dataset_type: calibration 数据集类型 ("c4" 或 "mmlu")
    """
    print(f"=== 生成 EoRA Adapter ===")
    print(f"原始模型: {model_id}")
    print(f"量化模型: {quant_path}")
    print(f"EoRA 保存路径: {eora_save_path}")
    print(f"Rank: {rank}")
    
    # 创建 Lora 配置
    eora = Lora(
        path=eora_save_path,  # 生成时为保存路径
        rank=rank,
    )
    
    # 构建 calibration 数据集
    if dataset_type == "mmlu":
        calibration_dataset = construct_mmlu_dataset()
        print("使用 MMLU + C4 混合数据集")
    else:
        calibration_dataset = construct_c4_dataset()
        print("使用 C4 数据集")
    
    # 生成 EoRA adapter
    GPTQModel.adapter.generate(
        adapter=eora,
        model_id_or_path=model_id,
        quantized_model_id_or_path=quant_path,
        calibration_dataset=calibration_dataset,
        calibration_dataset_concat_size=0,  # 禁用拼接
    )
    
    print(f"EoRA adapter 已保存到: {eora_save_path}")


# ============================================
# 4. 加载 EoRA 并推理
# ============================================
def load_and_inference(quant_path: str, eora_path: str, rank: int = 32, prompt: str = "Capital of France is"):
    """
    加载量化模型和 EoRA adapter 进行推理
    
    Args:
        quant_path: 已量化模型的路径
        eora_path: EoRA adapter 的加载路径
        rank: EoRA 的 rank 值
        prompt: 推理用的提示词
    """
    print(f"=== 加载 EoRA 并推理 ===")
    print(f"量化模型: {quant_path}")
    print(f"EoRA 路径: {eora_path}")
    
    # 创建 Lora 配置
    eora = Lora(
        path=eora_path,  # 加载时为加载路径
        rank=rank,
    )
    
    # 加载带 EoRA 的量化模型
    model = GPTQModel.load(
        model_id_or_path=quant_path,
        adapter=eora
    )
    
    # 推理
    print(f"Prompt: {prompt}")
    tokens = model.generate(prompt)[0]
    result = model.tokenizer.decode(tokens)
    
    print(f"Result: {result}")
    return result


# ============================================
# 5. 主函数
# ============================================
def main():
    parser = argparse.ArgumentParser(description="EoRA (Error-oriented Low-Rank Adaptation) 示例")
    parser.add_argument(
        '--mode', type=str, choices=['generate', 'inference', 'both'],
        default='both',
        help='运行模式: generate(生成EoRA), inference(推理), both(两者都执行)'
    )
    parser.add_argument(
        '--model_id', type=str, default=MODEL_ID,
        help='原始全精度模型 ID 或路径'
    )
    parser.add_argument(
        '--quant_path', type=str, default=QUANT_PATH,
        help='已量化模型路径'
    )
    parser.add_argument(
        '--eora_path', type=str, default=None,
        help='EoRA adapter 路径 (默认: {quant_path}/eora_rank{rank})'
    )
    parser.add_argument(
        '--rank', type=int, default=EORA_RANK,
        help='EoRA rank 值 (默认: 32)'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['c4', 'mmlu'], default='c4',
        help='Calibration 数据集类型'
    )
    parser.add_argument(
        '--prompt', type=str, default="Capital of France is",
        help='推理用的提示词'
    )
    
    args = parser.parse_args()
    
    # 设置 EoRA 路径
    eora_path = args.eora_path or os.path.join(args.quant_path, f"eora_rank{args.rank}")
    
    if args.mode in ['generate', 'both']:
        generate_eora(
            model_id=args.model_id,
            quant_path=args.quant_path,
            eora_save_path=eora_path,
            rank=args.rank,
            dataset_type=args.dataset
        )
    
    if args.mode in ['inference', 'both']:
        load_and_inference(
            quant_path=args.quant_path,
            eora_path=eora_path,
            rank=args.rank,
            prompt=args.prompt
        )


if __name__ == '__main__':
    main()

# ============================================
# 更多详情请参考:
# - GPTQModel/examples/eora/README.md
# - GPTQModel/examples/eora/eora_generation.py (量化时同时生成 EoRA)
# - GPTQModel/examples/eora/post_quant_eora_generation.py (量化后生成 EoRA)
# - GPTQModel/examples/eora/eora_load_and_inference.py (加载和推理)
# ============================================