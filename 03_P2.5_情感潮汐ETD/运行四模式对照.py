# -*- coding: utf-8 -*-
"""
四模式对照：裸 / 潮汐 / 回响 / 混合
=====================================
验证 回响×潮汐 混合方案的独立效果：
- 裸    ：标准生成
- 潮汐  ：极性定向概率引导
- 回响  ：hidden_state 向量注入
- 混合  ：回响(表示空间) + 潮汐(概率空间)

同种子、同提示词、同采样参数，唯一变量是挂载的方案。
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

回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)

from transformers import AutoModelForCausalLM, AutoTokenizer

from 潮汐感知器 import 潮汐感知器
from 潮汐决策器 import 潮汐决策器
from 潮汐解码器 import 潮汐解码器
from 混合注入器 import 混合注入器
from semantic_echo.采样处理器 import 回响注入器
from semantic_echo.回响池 import 语义回响池
from semantic_echo.情感过滤器 import 情感过滤器

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
输出目录 = os.path.join(本目录, "对照结果")
os.makedirs(输出目录, exist_ok=True)

测试提示词 = [
    "我失恋了，心里好难受，感觉整个世界都塌了。",
    "今天在公司被领导当众批评，特别委屈。",
    "我升职了！同事们都说我实至名归！",
    "你好，请问今天天气怎么样？",
    "妈妈生病住院了，我好担心她。",
]

# 回响参数（1.5B 扫描表最优 λ=0.08；混合时降为 0.05 防叠加过强）
回响参数 = {"λ": 0.08, "γ": 0.07, "τ": 0.09}
混合参数 = {"λ": 0.05, "γ": 0.07, "τ": 0.09}


def 计算语义熵(logits):
    if logits.dim() == 2:
        logits = logits[0]
    logits = logits.clone().float()
    logits[logits == float('-inf')] = -1e4
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return -(probs * log_probs).sum().item()


def 计算文本重复率(文本, n=2):
    词 = [w for w in re.split(r"[\s，。！？、；：,.!?;:]+", 文本) if w]
    if len(词) < n:
        return 0.0
    grams = [tuple(词[i:i + n]) for i in range(len(词) - n + 1)]
    return 1.0 - len(set(grams)) / max(len(grams), 1)


def 情感命中率(文本, 感知器):
    if not 文本.strip():
        return 0.0
    pos, neg, 命中词, _ = 感知器._扫描情感得分(文本)
    词数 = max(len(list(感知器._分词(文本))), 1)
    return (abs(pos) + abs(neg)) / 词数


def 裸生成(model, tokenizer, 提示, 种子):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    消息 = [{"role": "user", "content": 提示}]
    应用提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(应用提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids, temperature=1.0, top_p=0.9, top_k=50, do_sample=True,
            repetition_penalty=1.05, max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    文本 = tokenizer.decode(新token, skip_special_tokens=True).strip()
    熵, 步 = 0.0, 0
    ids = inputs.input_ids
    with torch.no_grad():
        for i in range(len(新token)):
            out2 = model(ids)
            熵 += 计算语义熵(out2.logits[0, -1, :])
            步 += 1
            ids = torch.cat([ids, 新token[i:i + 1].unsqueeze(0)], dim=-1)
    return 文本, 熵 / max(步, 1)


def 潮汐生成(model, tokenizer, 提示, 种子):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    感知器 = 潮汐感知器()
    决策器 = 潮汐决策器(感知器)
    解码器 = 潮汐解码器(model, tokenizer, 感知器, 决策器)
    消息 = [{"role": "user", "content": 提示}]
    应用提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(应用提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = 解码器.生成(inputs.input_ids, max_new_tokens=128,
                          temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
                          eos_token_id=tokenizer.eos_token_id, 用户文本=提示)
    新token = out[0, inputs.input_ids.shape[1]:]
    文本 = tokenizer.decode(新token, skip_special_tokens=True).strip()
    熵, 步 = 0.0, 0
    ids = inputs.input_ids
    with torch.no_grad():
        for i in range(len(新token)):
            out2 = model(ids)
            熵 += 计算语义熵(out2.logits[0, -1, :])
            步 += 1
            ids = torch.cat([ids, 新token[i:i + 1].unsqueeze(0)], dim=-1)
    return 文本, 熵 / max(步, 1)


def 回响生成(model, tokenizer, 提示, 种子, λ覆盖=None):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    感知器 = 潮汐感知器()  # 复用情感词库
    过滤器 = 情感过滤器()
    过滤器.加载词库()
    池 = 语义回响池(hidden_dim=model.config.hidden_size,
                   decay_gamma=回响参数["γ"])
    注入器 = 回响注入器(model, 池, lambda_strength=λ覆盖 or 回响参数["λ"],
                      情感过滤器实例=过滤器)
    消息 = [{"role": "user", "content": 提示}]
    应用提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(应用提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = 注入器.生成(inputs.input_ids, max_new_tokens=128,
                          temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
                          eos_token_id=tokenizer.eos_token_id, tokenizer=tokenizer)
    新token = out[0, inputs.input_ids.shape[1]:]
    文本 = tokenizer.decode(新token, skip_special_tokens=True).strip()
    熵, 步 = 0.0, 0
    ids = inputs.input_ids
    with torch.no_grad():
        for i in range(len(新token)):
            out2 = model(ids)
            熵 += 计算语义熵(out2.logits[0, -1, :])
            步 += 1
            ids = torch.cat([ids, 新token[i:i + 1].unsqueeze(0)], dim=-1)
    return 文本, 熵 / max(步, 1)


def 混合生成(model, tokenizer, 提示, 种子):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    感知器 = 潮汐感知器()
    决策器 = 潮汐决策器(感知器)
    过滤器 = 情感过滤器()
    过滤器.加载词库()
    池 = 语义回响池(hidden_dim=model.config.hidden_size,
                   decay_gamma=混合参数["γ"])
    注入器 = 混合注入器(model, 池, tokenizer, 感知器, 决策器,
                       lambda_strength=混合参数["λ"],
                       情感过滤器实例=过滤器)
    消息 = [{"role": "user", "content": 提示}]
    应用提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(应用提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = 注入器.生成(inputs.input_ids, max_new_tokens=128,
                          temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
                          eos_token_id=tokenizer.eos_token_id, tokenizer=tokenizer,
                          用户文本=提示)
    新token = out[0, inputs.input_ids.shape[1]:]
    文本 = tokenizer.decode(新token, skip_special_tokens=True).strip()
    熵, 步 = 0.0, 0
    ids = inputs.input_ids
    with torch.no_grad():
        for i in range(len(新token)):
            out2 = model(ids)
            熵 += 计算语义熵(out2.logits[0, -1, :])
            步 += 1
            ids = torch.cat([ids, 新token[i:i + 1].unsqueeze(0)], dim=-1)
    return 文本, 熵 / max(步, 1)


生成函数 = {"裸": 裸生成, "潮汐": 潮汐生成, "回响": 回响生成, "混合": 混合生成}


def 主程序():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", choices=["裸", "潮汐", "回响", "混合", "全部"], default="全部")
    args = ap.parse_args()
    模式列表 = list(生成函数) if args.模式 == "全部" else [args.模式]

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
        该题 = {"提示": 提示, "种子": 种子}
        for 模式 in 模式列表:
            文本, 熵 = 生成函数[模式](模型, 分词器, 提示, 种子)
            命中 = 情感命中率(文本, 感知器统计)
            重复 = 计算文本重复率(文本)
            该题[模式] = {"文本": 文本, "熵": 熵, "命中": 命中, "重复": 重复}
            print(f"  [{模式}] 熵={熵:.3f} 命中={命中:.4f} 重复={重复}")
        print()
        结果列表.append(该题)

    # 汇总
    print("═" * 60)
    print("汇总（均值）")
    汇总 = {}
    for 模式 in 模式列表:
        熵均 = sum(r[模式]["熵"] for r in 结果列表) / len(结果列表)
        命均 = sum(r[模式]["命中"] for r in 结果列表) / len(结果列表)
        重均 = sum(r[模式]["重复"] for r in 结果列表) / len(结果列表)
        汇总[模式] = {"熵": round(熵均, 4), "命中": round(命均, 4), "重复": round(重均, 4)}
        print(f"  [{模式}] 熵={熵均:.3f} 命中={命均:.4f} 重复={重均:.3f}")

    # 相对裸的提升
    if "裸" in 汇总:
        print("\n相对裸：")
        for 模式 in 模式列表:
            if 模式 == "裸":
                continue
            print(f"  [{模式}] 熵Δ={汇总[模式]['熵']-汇总['裸']['熵']:+.3f} "
                  f"命中Δ={汇总[模式]['命中']-汇总['裸']['命中']:+.4f} "
                  f"重复Δ={汇总[模式]['重复']-汇总['裸']['重复']:+.3f}")

    输出路径 = os.path.join(输出目录, f"四模式_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump({"汇总": 汇总, "明细": 结果列表}, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {输出路径}")


if __name__ == "__main__":
    主程序()
