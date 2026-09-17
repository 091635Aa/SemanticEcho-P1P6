# -*- coding: utf-8 -*-
"""
v5 消融：定位口语化/长度收尾/新词表对混合输出的影响（1.5B，5 条提示快速对比）
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

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
测试提示 = [
    "我总觉得自己不值得被爱。",
    "我真的好累，好想逃。",
    "下班了吗？",
    "你为什么总能接我梗？",
    "你这是在夸我还是挖苦我？",
]


def 混合生成(model, tokenizer, 提示, 种子, AI抑制=4.0, 口语化=1.0, 目标长=34):
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
                       lambda_strength=0.08, 引导倍率=6.0,
                       情感过滤器实例=过滤器, AI腔抑制强度=AI抑制,
                       口语化强度=口语化, 目标长度=目标长)
    消息 = [{"role": "user", "content": 提示}]
    提示文 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
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


def 主程序():
    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[加载] → {设备}", flush=True)
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16 if 设备 == "cuda" else torch.float32,
        trust_remote_code=True).to(设备)
    模型.eval()

    配置 = [
        ("v5完整(口语化1.0,长34)", dict(口语化=1.0, 目标长=34)),
        ("口语化0(长34)", dict(口语化=0.0, 目标长=34)),
        ("口语化0(长200=关收尾)", dict(口语化=0.0, 目标长=200)),
        ("口语化0.5(长34)", dict(口语化=0.5, 目标长=34)),
    ]
    for 名称, kw in 配置:
        print(f"\n════ {名称} ════", flush=True)
        for 提示 in 测试提示:
            r = 混合生成(模型, 分词器, 提示, 42, **kw)
            print(f"  [{提示[:12]}] 长{len(r)} | {r[:58]}", flush=True)

    模型.to("cpu")
    del 模型, 分词器
    torch.cuda.empty_cache()


if __name__ == "__main__":
    主程序()
