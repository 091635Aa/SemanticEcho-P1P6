# -*- coding: utf-8 -*-
"""超融合解码器（UFD）LLM-Judge 对比 —— 生成阶段（30 条样本）

模式：裸 / 锚点P4 / 混合全开 / 超融合DMR（纯净模式，无人设，隔离解码器机制差异）
协议：同种子 42、同采样参数（temperature=1.0, top_p=0.9, top_k=50,
repetition_penalty=1.05, max_new_tokens=256）；输出缓存到 评测结果\超融合_生成_30.json。
"""
import os
import sys
import json
import torch
from datetime import datetime

工作目录 = os.path.dirname(os.path.abspath(__file__))
if 工作目录 not in sys.path:
    sys.path.insert(0, 工作目录)
回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)

from transformers import AutoModelForCausalLM, AutoTokenizer

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器
from 锚点解码器 import 锚点解码器
from 混合锚点器 import 混合锚点器
from 超融合解码器 import 超融合解码器

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
样本路径 = r"i:\Desktop\语义回响\图灵测试\样本_30条.json"
输出目录 = os.path.join(工作目录, "..", "评测结果")
os.makedirs(输出目录, exist_ok=True)
结果路径 = os.path.join(输出目录, "超融合_生成_30.json")

种子 = 42
模式列表 = ["裸", "锚点P4", "混合全开", "超融合DMR"]


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
            repetition_penalty=1.05, max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(新token, skip_special_tokens=True).strip()


def 方案生成(model, tokenizer, 解码器, 提示, 种子):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    消息 = [{"role": "user", "content": 提示}]
    应用提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(应用提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out, 统计 = 解码器.生成(
            inputs.input_ids, max_new_tokens=256,
            temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id, 用户文本=提示)
    新token = out[0, inputs.input_ids.shape[1]:]
    文本 = tokenizer.decode(新token, skip_special_tokens=True).strip()
    统计["文本"] = 文本
    return 文本, 统计


def main():
    print(f"=== UFD 生成阶段 {datetime.now().strftime('%H:%M:%S')} ===", flush=True)
    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16 if 设备 == "cuda" else torch.float32,
        trust_remote_code=True).to(设备)
    print("模型加载完成", flush=True)

    锚库 = 锚点库(模型, 分词器)
    锚库.构建()
    感知器, 潮汐决策 = None, None
    try:
        sys.path.insert(0, r"h:\情感潮汐解码（Emotion Tidal Decoding, ETD）")
        from 潮汐感知器 import 潮汐感知器
        from 潮汐决策器 import 潮汐决策器
        感知器 = 潮汐感知器()
        潮汐决策 = 潮汐决策器(感知器)
    except Exception as e:
        print(f"P3 感知器加载失败（简易 VAD 兜底）：{e}", flush=True)
    目标决策 = 目标决策器(感知器=感知器, 潮汐决策器=潮汐决策, 锚点库=锚库)

    解码器 = {}
    解码器["锚点P4"] = 锚点解码器(模型, 分词器, 锚库, 目标决策,
                                β=0.8, T_anchor=0.3, 句子停止=True)
    解码器["混合全开"] = 混合锚点器(模型, 分词器, 锚库, 目标决策,
                                 锚点β=0.8, 锚点T=0.3, 回响λ=0.08, 潮汐倍率=12.0,
                                 开启A=True, 开启B=True, 开启C=True, 句子停止=True)
    解码器["超融合DMR"] = 超融合解码器(模型, 分词器, 锚库, 目标决策,
                                     开启DSA=True, 开启DMR=True, 开启锚点偏置=False,
                                     α基=0.15, α倍率=1.0, T_emo=0.5, 句子停止=True)

    with open(样本路径, "r", encoding="utf-8") as f:
        数据 = json.load(f)
    样本 = 数据["样本"]

    结果 = {"模型": "Qwen2.5-1.5B-Instruct", "种子": 种子, "模式": 模式列表,
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "回复": []}

    for i, 项 in enumerate(样本):
        提示 = 项["user"]
        print(f"[{i+1}/30] {提示[:20]}...", flush=True)
        条目 = {"序号": 项["序号"], "user": 提示, "girl": 项["girl"], "回复": {}}
        回复, 统计 = 裸生成(模型, 分词器, 提示, 种子), None
        条目["回复"]["裸"] = {"文本": 回复, "统计": {"平均熵": 0.0}}
        for 名称, 解码 in 解码器.items():
            回复, 统计 = 方案生成(模型, 分词器, 解码, 提示, 种子)
            条目["回复"][名称] = {"文本": 回复, "统计": 统计}
        结果["回复"].append(条目)
        if (i + 1) % 10 == 0:
            with open(结果路径, "w", encoding="utf-8") as f:
                json.dump(结果, f, ensure_ascii=False, indent=2)

    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=2)
    print(f"\n生成完成，已保存：{结果路径}", flush=True)


if __name__ == "__main__":
    main()
