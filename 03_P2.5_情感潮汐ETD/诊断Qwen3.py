# -*- coding: utf-8 -*-
"""
Qwen3-1.7B 快速诊断：潮汐/混合 在 Qwen3 上的参数适应性
======================================================
1.5B 调优的参数直接搬 Qwen3 会退化（AI抑制过度压高频词→重复崩溃）。
扫描：AI抑制 ∈ {0,1,2} × {潮汐, 混合} 在 5 条提示上的输出质量。
"""
import os
import sys
import torch

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

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen3-1.7B-Instruct"

测试提示 = [
    "我总觉得自己不值得被爱。",
    "我真的好累，好想逃。",
    "下班了吗？",
    "昨天那部你选的恐怖片，我都快被吓死。",
    "你为什么总能接我梗？",
]


def 构建提示(tokenizer, 消息):
    return tokenizer.apply_chat_template(
        消息, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def 裸生成(model, tokenizer, 提示, 种子):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    消息 = [{"role": "user", "content": 提示}]
    提示文 = 构建提示(tokenizer, 消息)
    inputs = tokenizer(提示文, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(inputs.input_ids, temperature=1.0, top_p=0.9, top_k=50,
                             do_sample=True, repetition_penalty=1.05, max_new_tokens=64,
                             pad_token_id=tokenizer.eos_token_id)
    新token = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(新token, skip_special_tokens=True).strip()


def 潮汐生成(model, tokenizer, 提示, 种子, AI抑制=0.0):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    感知器 = 潮汐感知器()
    决策器 = 潮汐决策器(感知器)
    解码器 = 潮汐解码器(model, tokenizer, 感知器, 决策器, AI腔抑制强度=AI抑制)
    消息 = [{"role": "user", "content": 提示}]
    提示文 = 构建提示(tokenizer, 消息)
    inputs = tokenizer(提示文, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = 解码器.生成(inputs.input_ids, max_new_tokens=64,
                          temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
                          eos_token_id=tokenizer.eos_token_id, 用户文本=提示)
    新token = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(新token, skip_special_tokens=True).strip()


def 混合生成(model, tokenizer, 提示, 种子, λ=0.08, 倍率=6.0, AI抑制=0.0):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    from semantic_echo.回响池 import 语义回响池
    from semantic_echo.情感过滤器 import 情感过滤器
    from 混合注入器 import 混合注入器
    感知器 = 潮汐感知器()
    决策器 = 潮汐决策器(感知器)
    过滤器 = 情感过滤器()
    过滤器.加载词库()
    池 = 语义回响池(hidden_dim=model.config.hidden_size, decay_gamma=0.07)
    解码器 = 混合注入器(model, 池, tokenizer, 感知器, 决策器,
                       lambda_strength=λ, 引导倍率=倍率,
                       情感过滤器实例=过滤器, AI腔抑制强度=AI抑制)
    消息 = [{"role": "user", "content": 提示}]
    提示文 = 构建提示(tokenizer, 消息)
    inputs = tokenizer(提示文, return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            out = 解码器.生成(inputs.input_ids, max_new_tokens=64,
                              temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
                              eos_token_id=tokenizer.eos_token_id, tokenizer=tokenizer,
                              用户文本=提示)
    finally:
        try:
            解码器._移除钩子()
        except Exception:
            pass
        del 解码器, 池, 过滤器, 感知器, 决策器
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    新token = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(新token, skip_special_tokens=True).strip()


def 计重复(文本):
    """简单退化检测：重复字符段"""
    import re
    m = re.findall(r"(.)\1{4,}", 文本)  # 同一字符连续5次
    return len(m)


def 主程序():
    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[加载] → {设备}", flush=True)
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16 if 设备 == "cuda" else torch.float32,
        trust_remote_code=True).to(设备)
    模型.eval()
    print("[加载] 完成\n", flush=True)

    种子 = 42
    # 裸基线
    print("════ 裸 ════", flush=True)
    for 提示 in 测试提示:
        r = 裸生成(模型, 分词器, 提示, 种子)
        print(f"  [{提示[:14]}] 重复={计重复(r)} | {r[:60]}", flush=True)
    print(flush=True)

    # 潮汐 × AI抑制
    for 抑制 in [0.0, 2.0]:
        print(f"════ 潮汐 AI抑制={抑制} ════", flush=True)
        for 提示 in 测试提示:
            r = 潮汐生成(模型, 分词器, 提示, 种子, AI抑制=抑制)
            print(f"  [{提示[:14]}] 重复={计重复(r)} | {r[:60]}", flush=True)
        print(flush=True)

    # 混合 × AI抑制（去掉 AI 抑制看回响+潮汐是否 OK）
    for 抑制 in [0.0]:
        print(f"════ 混合(λ=0.08,倍率=6) AI抑制={抑制} ════", flush=True)
        for 提示 in 测试提示:
            r = 混合生成(模型, 分词器, 提示, 种子, AI抑制=抑制)
            print(f"  [{提示[:14]}] 重复={计重复(r)} | {r[:60]}", flush=True)
        print(flush=True)

    模型.to("cpu")
    del 模型, 分词器
    torch.cuda.empty_cache()


if __name__ == "__main__":
    主程序()
