# -*- coding: utf-8 -*-
"""训练 P6 情感LoRA 外挂适配器 — Qwen2.5-1.5B 基座，监督回复（instruction 掩码）
用法: python p6_train_lora.py [--epochs 3] [--r 8] [--alpha 16] [--lr 2e-4] [--out 输出目录]
输出: f:\lora外挂\lora_adapters\p6_emotion\
"""
import os
import sys
import json
import argparse
from pathlib import Path

# ── 内存约束（低占用训练，避免挤占同机其他进程）──
os.environ.setdefault("HF_HOME", r"C:\P6临时盘\hf")
os.environ.setdefault("TORCH_HOME", r"C:\P6临时盘\torch")
os.environ.setdefault("TEMP", r"D:\P6临时盘\tmp")
os.environ.setdefault("TMP", r"D:\P6临时盘\tmp")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
os.environ.setdefault("OMP_NUM_THREADS", "4")

项目根 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(项目根))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

torch.set_num_threads(4)

数据集路径 = 项目根 / "data" / "p6_train.jsonl"
默认输出 = 项目根 / "lora_adapters" / "p6_emotion"


def 构建数据集(tokenizer, max_length):
    from datasets import Dataset
    原始 = []
    with open(数据集路径, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                原始.append(json.loads(line))

    def 格式化(d):
        messages = [
            {"role": "user", "content": d["instruction"]},
            {"role": "assistant", "content": d["response"]},
        ]
        全文 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # 只取提示部分长度（含 assistant 开始标记），用于掩码
        return {"text": 全文, "instruction": d["instruction"], "response": d["response"]}

    ds = Dataset.from_list([格式化(d) for d in 原始])

    def tokenize(example):
        # 提示部分（不含回复）用于掩码
        prompt_messages = [{"role": "user", "content": example["instruction"]}]
        提示文本 = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        提示ids = tokenizer(提示文本, add_special_tokens=False)["input_ids"]
        out = tokenizer(example["text"], truncation=True, max_length=max_length, padding=False)
        ids = out["input_ids"]
        labels = [-100] * len(ids)
        # 从提示长度之后开始监督（提示 tokens 掩码）
        监督起点 = min(len(提示ids), len(ids))
        labels[监督起点:] = ids[监督起点:]
        out["labels"] = labels
        return out

    return ds.map(tokenize, remove_columns=["text", "instruction", "response"])


def 主():
    global 默认输出
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="基座模型路径")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.out:
        默认输出 = Path(args.out)

    print(f"[加载基座] {args.model} fp16")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    配置 = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, 配置)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    数据集 = 构建数据集(tokenizer, max_length=256)
    print(f"训练数据: {len(数据集)} 条")
    print(f"样本示例 labels 监督起点测试：{数据集[0]['input_ids'][:20]}")

    训练参数 = TrainingArguments(
        output_dir=str(项目根 / "training_scripts" / "checkpoints_p6"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=2,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        fp16=True,
        dataloader_pin_memory=False,
    )
    训练器 = Trainer(
        model=model,
        args=训练参数,
        train_dataset=数据集,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    print("[开始训练]")
    检查点目录 = 项目根 / "training_scripts" / "checkpoints_p6"
    最新检查点 = None
    if 检查点目录.exists():
        for _子 in sorted(检查点目录.iterdir(), reverse=True):
            if _子.is_dir() and (_子 / "trainer_state.json").exists():
                最新检查点 = str(_子)
                break
    if 最新检查点:
        print(f"[断点续训] {最新检查点}")
        训练器.train(resume_from_checkpoint=最新检查点)
    else:
        训练器.train()
    默认输出.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(默认输出))
    tokenizer.save_pretrained(str(默认输出))
    print(f"P6 LoRA 已保存: {默认输出}")


if __name__ == "__main__":
    主()
