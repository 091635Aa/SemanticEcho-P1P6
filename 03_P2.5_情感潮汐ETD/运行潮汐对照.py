# -*- coding: utf-8 -*-
"""
运行潮汐对照：同种子"裸 vs 潮汐"三组验证
==========================================
验证第三套架构（情感潮汐解码 ETD）相对裸模型的独立效果。

- 裸生成：Qwen2.5-1.5B-Instruct 标准 model.generate（无任何模块）
- 潮汐生成：同一模型挂潮汐解码器（感知 → 决策 → 表达）

同种子、同提示词、同采样参数，唯一变量是"是否挂潮汐引导"。
"""
import os
import sys
import json
import re
import time
import torch
from datetime import datetime

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from transformers import AutoModelForCausalLM, AutoTokenizer

from 潮汐感知器 import 潮汐感知器
from 潮汐决策器 import 潮汐决策器
from 潮汐解码器 import 潮汐解码器

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
输出目录 = os.path.join(本目录, "对照结果")
os.makedirs(输出目录, exist_ok=True)

生成参数 = dict(temperature=1.0, top_p=0.9, top_k=50, do_sample=True,
                repetition_penalty=1.05, max_new_tokens=128)

# 测试提示词（含情感语境，便于观察情感引导效果）
测试提示词 = [
    "我失恋了，心里好难受，感觉整个世界都塌了。",
    "今天在公司被领导当众批评，特别委屈。",
    "我升职了！同事们都说我实至名归！",
    "你好，请问今天天气怎么样？",
    "妈妈生病住院了，我好担心她。",
]


def 计算语义熵(logits: torch.Tensor) -> float:
    """单 token 位置语义熵（与架构一同口径：softmax 熵，nats）"""
    if logits.dim() == 2:
        logits = logits[0]
    logits = logits.clone().float()
    logits[logits == float('-inf')] = -1e4
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return -(probs * log_probs).sum().item()


def 计算文本重复率(文本: str, n: int = 2) -> float:
    """n-gram 重复率：1 - 唯一n-gram数/总n-gram数"""
    词 = [w for w in re.split(r"[\s，。！？、；：,.!?;:]+", 文本) if w]
    if len(词) < n:
        return 0.0
    grams = [tuple(词[i:i + n]) for i in range(len(词) - n + 1)]
    return 1.0 - len(set(grams)) / max(len(grams), 1)


def 情感命中率(文本: str, 感知器) -> float:
    """情感词密度：命中情感词数 / 分词词数（与架构一"命中率"口径一致）"""
    if not 文本.strip():
        return 0.0
    pos, neg, 命中词, _ = 感知器._扫描情感得分(文本)
    词数 = max(len(list(感知器._分词(文本))), 1)
    return (abs(pos) + abs(neg)) / 词数


def 裸生成(model, tokenizer, 提示, 种子):
    """标准 model.generate"""
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    消息 = [{"role": "user", "content": 提示}]
    应用提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(应用提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            temperature=1.0, top_p=0.9, top_k=50, do_sample=True,
            repetition_penalty=1.05, max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    文本 = tokenizer.decode(新token, skip_special_tokens=True).strip()
    # 熵：重算每步（近似——用生成器逐token前向）
    熵 = 0.0
    步 = 0
    ids = inputs.input_ids
    with torch.no_grad():
        for i in range(len(新token)):
            out2 = model(ids)
            lg = out2.logits[0, -1, :]
            熵 += 计算语义熵(lg)
            步 += 1
            ids = torch.cat([ids, 新token[i:i + 1].unsqueeze(0)], dim=-1)
    平均熵 = 熵 / max(步, 1)
    return 文本, 平均熵, 步


def 潮汐生成(model, tokenizer, 提示, 种子):
    """潮汐解码器生成"""
    感知器 = 潮汐感知器()
    决策器 = 潮汐决策器(感知器)
    解码器 = 潮汐解码器(model, tokenizer, 感知器, 决策器)

    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    消息 = [{"role": "user", "content": 提示}]
    应用提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(应用提示, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        out = 解码器.生成(
            inputs.input_ids, max_new_tokens=128,
            temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            用户文本=提示,
        )
    耗时 = time.time() - start

    新token = out[0, inputs.input_ids.shape[1]:]
    文本 = tokenizer.decode(新token, skip_special_tokens=True).strip()

    # 熵：逐 token 前向记录（不重新生成，用已生成序列）
    熵 = 0.0
    步 = 0
    ids = inputs.input_ids
    with torch.no_grad():
        for i in range(len(新token)):
            out2 = model(ids)
            lg = out2.logits[0, -1, :]
            熵 += 计算语义熵(lg)
            步 += 1
            ids = torch.cat([ids, 新token[i:i + 1].unsqueeze(0)], dim=-1)
    平均熵 = 熵 / max(步, 1)

    统计 = dict(解码器.统计)
    统计["耗时秒"] = round(耗时, 2)
    return 文本, 平均熵, 步, 统计


def 主程序():
    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[加载] Qwen2.5-1.5B-Instruct → {设备} ...")
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16 if 设备 == "cuda" else torch.float32,
        trust_remote_code=True).to(设备)
    模型.eval()
    print(f"[加载] 完成, {模型.num_parameters()/1e6:.0f}M 参数\n")

    种子 = 42
    感知器统计 = 潮汐感知器()

    结果列表 = []
    for idx, 提示 in enumerate(测试提示词):
        print(f"──── 测试 {idx + 1}/{len(测试提示词)} ────")
        print(f"提示: {提示}")

        # 裸
        裸文本, 裸熵, 裸步 = 裸生成(模型, 分词器, 提示, 种子)
        裸命中 = 情感命中率(裸文本, 感知器统计)
        裸重 = 计算文本重复率(裸文本)

        # 潮汐
        潮文本, 潮熵, 潮步, 潮统计 = 潮汐生成(模型, 分词器, 提示, 种子)
        潮命中 = 情感命中率(潮文本, 感知器统计)
        潮重 = 计算文本重复率(潮文本)

        print(f"[裸]   熵={裸熵:.3f} 重复={裸重:.3f} 情感命中={裸命中:.4f}")
        print(f"      {裸文本[:120]}")
        print(f"[潮汐] 熵={潮熵:.3f} 重复={潮重:.3f} 情感命中={潮命中:.4f} {潮统计}")
        print(f"      {潮文本[:120]}")

        结果列表.append({
            "提示": 提示,
            "种子": 种子,
            "裸": {"文本": 裸文本, "熵": 裸熵, "重复": 裸重, "情感命中": 裸命中, "步": 裸步},
            "潮汐": {"文本": 潮文本, "熵": 潮熵, "重复": 潮重, "情感命中": 潮命中, "步": 潮步,
                     "统计": 潮统计},
        })

    # 汇总
    print("\n" + "═" * 60)
    print("汇总（均值）")
    裸熵均 = sum(r["裸"]["熵"] for r in 结果列表) / len(结果列表)
    潮熵均 = sum(r["潮汐"]["熵"] for r in 结果列表) / len(结果列表)
    裸重均 = sum(r["裸"]["重复"] for r in 结果列表) / len(结果列表)
    潮重均 = sum(r["潮汐"]["重复"] for r in 结果列表) / len(结果列表)
    裸命均 = sum(r["裸"]["情感命中"] for r in 结果列表) / len(结果列表)
    潮命均 = sum(r["潮汐"]["情感命中"] for r in 结果列表) / len(结果列表)
    print(f"  语义熵      : 裸 {裸熵均:.3f} → 潮汐 {潮熵均:.3f} (Δ {潮熵均 - 裸熵均:+.3f})")
    print(f"  重复率      : 裸 {裸重均:.3f} → 潮汐 {潮重均:.3f} (Δ {潮重均 - 裸重均:+.3f})")
    print(f"  情感命中率  : 裸 {裸命均:.5f} → 潮汐 {潮命均:.5f} (Δ {潮命均 - 裸命均:+.5f})")

    汇总 = {
        "时间": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "模型": "Qwen2.5-1.5B-Instruct",
        "种子": 种子,
        "均值": {
            "裸熵": 裸熵均, "潮汐熵": 潮熵均, "熵Δ": 潮熵均 - 裸熵均,
            "裸重复": 裸重均, "潮汐重复": 潮重均, "重复Δ": 潮重均 - 裸重均,
            "裸情感命中": 裸命均, "潮汐情感命中": 潮命均, "情感命中Δ": 潮命均 - 裸命均,
        },
        "明细": 结果列表,
    }
    输出路径 = os.path.join(输出目录, f"对照_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump(汇总, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {输出路径}")


if __name__ == "__main__":
    主程序()
