# -*- coding: utf-8 -*-
"""情感外挂 LoRA 简单测试 — 基线 / 外挂 / 回响 / 外挂+回响 四组对比
用法: python run_emotion_lora_test.py [--model 1.5B|3B|7B] [--adapter 适配器名]
输出: f:\lora外挂\evaluation\emotion_v1_test_{模型}.json
"""
import os
import sys
import json
import argparse
from pathlib import Path

项目根 = Path(__file__).resolve().parent.parent
工程根 = Path(r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程")
sys.path.insert(0, str(工程根 / "agent_echo"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from echo_common import 加载模型, 创建情感过滤器, 运行回响, 运行基线

# 模型路径表：与 coda.py 一致
模型表 = {
    "0.5B": r"i:\Desktop\语义回响\本地模型",
    "1.5B": r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct",
    "3B":   r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-3B-Instruct",
    "7B":   r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-7B-Instruct",
}
hidden_dim基准 = {"0.5B": 896, "1.5B": 1536, "3B": 2048, "7B": 3584}

# T2/T3 框架扫描最优 λ（3B=0.10、7B=0.06）；0.5B 用基准 0.5，1.5B 用 λ_norm 近似 0.29
最优λ表 = {"0.5B": 0.5, "1.5B": 0.29, "3B": 0.10, "7B": 0.06}

提示词集 = [
    ("快乐", "请告诉我一件让你开心的事情。"),
    ("悲伤", "最近有没有让你感到难过的事？"),
    ("愤怒", "什么事情会让你感到生气？"),
    ("中性", "今天天气怎么样？"),
    ("复杂混合", "你如何看待成功和失败？"),
]

温柔词 = ["抱抱", "别担心", "陪着你", "没关系", "慢慢来", "辛苦了", "加油", "心疼", "支持你", "照顾", "放松", "理解你", "一起", "相信你"]


def 主():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["0.5B", "1.5B", "3B", "7B"], default="1.5B")
    parser.add_argument("--adapter", default="emotion_v1", help="适配器目录名（默认 emotion_v1）")
    parser.add_argument("--quant", default="4bit")
    args = parser.parse_args()

    适配器路径 = 项目根 / "lora_adapters" / args.adapter
    模型路径 = 模型表[args.model]
    归一化基准 = hidden_dim基准[args.model]
    λ = 最优λ表[args.model]
    输出路径 = 项目根 / "evaluation" / f"emotion_{args.adapter}_test_{args.model}.json"
    输出路径.parent.mkdir(parents=True, exist_ok=True)

    过滤 = 创建情感过滤器()
    print(f"[加载基座] {模型路径} ({args.quant})")
    model, tokenizer = 加载模型(模型路径, 量化=args.quant)
    print(f"[挂载 {args.adapter}]")
    model = PeftModel.from_pretrained(model, str(适配器路径))
    已挂 = True

    def 测(回响开):
        结果集 = []
        for 维度, 提示 in 提示词集:
            if 回响开:
                r = 运行回响(model, tokenizer, 提示, λ, 0.05, 过滤, None, "衰减", 3,
                             256, 前缀="", 归一化基准=归一化基准, repetition_penalty=1.0)
            else:
                r = 运行基线(model, tokenizer, 提示, max_new_tokens=256)
            结果集.append({"维度": 维度, "熵": r["平均熵"], "重复率": r["重复率"],
                          "文本": r["文本"] if isinstance(r.get("文本"), str) else r.get("文本", "")[:100]})
        熵 = sum(x["熵"] for x in 结果集) / len(结果集)
        重复 = sum(x["重复率"] for x in 结果集) / len(结果集)
        温柔 = sum(1 for x in 结果集 for w in 温柔词 if w in x["文本"]) / len(结果集)
        return {"平均熵": round(熵, 4), "平均重复率": round(重复, 4),
                "温柔命中率": round(温柔, 4), "样例": 结果集}

    输出 = {}
    for 名字, 回响 in [("基线(无外挂无回响)", False), ("外挂emotion_v1(无回响)", False),
                     ("回响(无外挂)", True), ("外挂+回响", True)]:
        print(f"[测试] {名字} ...")
        # 基线/回响组需禁用外挂
        if "外挂" not in 名字:
            with model.disable_adapter():
                输出[名字] = 测(回响)
        else:
            输出[名字] = 测(回响)
        print(f"    {输出[名字]}")
        torch.cuda.empty_cache()

    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump(输出, f, ensure_ascii=False, indent=2)
    print(f"测试结果已保存: {输出路径}")


if __name__ == "__main__":
    主()
