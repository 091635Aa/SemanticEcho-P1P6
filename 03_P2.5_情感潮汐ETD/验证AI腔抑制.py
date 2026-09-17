# -*- coding: utf-8 -*-
"""
快速验证 v3 双通道（AI 腔抑制）效果：同一提示对比 裸/潮汐v2/潮汐v3(抑制)
"""
import os
import sys
import torch

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from transformers import AutoModelForCausalLM, AutoTokenizer
from 潮汐感知器 import 潮汐感知器
from 潮汐决策器 import 潮汐决策器
from 潮汐解码器 import 潮汐解码器

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"

测试提示 = [
    "我总觉得自己不值得被爱。",
    "我真的好累，好想逃。",
    "下班了吗？",
    "昨天那部你选的恐怖片，我都快被吓死。",
    "你为什么总能接我梗？",
]


def 裸生成(model, tokenizer, 提示, 种子):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    消息 = [{"role": "user", "content": 提示}]
    提示文 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
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
    提示文 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(提示文, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = 解码器.生成(inputs.input_ids, max_new_tokens=64,
                          temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
                          eos_token_id=tokenizer.eos_token_id, 用户文本=提示)
    新token = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(新token, skip_special_tokens=True).strip()


def 主程序():
    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[加载] → {设备} ...")
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16 if 设备 == "cuda" else torch.float32,
        trust_remote_code=True).to(设备)
    模型.eval()
    print("[加载] 完成\n")

    种子 = 42
    for 提示 in 测试提示:
        print(f"════ 提示: {提示}")
        print(f"[裸]    {裸生成(模型, 分词器, 提示, 种子)[:110]}")
        print(f"[潮v2]  {潮汐生成(模型, 分词器, 提示, 种子, AI抑制=0.0)[:110]}")
        print(f"[潮v3]  {潮汐生成(模型, 分词器, 提示, 种子, AI抑制=2.0)[:110]}")
        print()

    模型.to("cpu")
    del 模型, 分词器
    torch.cuda.empty_cache()


if __name__ == "__main__":
    主程序()
