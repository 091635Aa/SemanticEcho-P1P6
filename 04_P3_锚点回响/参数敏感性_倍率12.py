# -*- coding: utf-8 -*-
"""
Task8 步骤3：参数敏感性（全开配置 潮汐倍率 6→12，5 条样本观察是否过注入）
=========================================================================
仅跑前 5 条样本（与 运行四模式_锚点.py 同种子 42/同 prompt），全开三通道
（锚点β=0.8 + 回响λ=0.08 + 潮汐倍率=12），对比倍率 6 的健康度/文本，
记录趋势（不过注入判断：熵不塌、重复率不爆升、情感命中率不爆炸、无空回复）。
结果存 评测结果\\参数敏感性_倍率12_样本5.json。
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import json
import sys
import torch

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)
回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)

from transformers import AutoModelForCausalLM, AutoTokenizer

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器
from 混合锚点器 import 混合锚点器
from 运行四模式_锚点 import 构建提示, 文本级情感命中率, 汇总健康度

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
样本路径 = r"i:\Desktop\语义回响\图灵测试\样本_30条.json"
输出目录 = os.path.join(本目录, "评测结果")
os.makedirs(输出目录, exist_ok=True)
输出文件 = os.path.join(输出目录, "参数敏感性_倍率12_样本5.json")
日志路径 = os.path.join(输出目录, "参数敏感性_倍率12_样本5.log")


def 记录日志(msg):
    print(msg, flush=True)
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def 主程序():
    if os.path.exists(日志路径):
        os.remove(日志路径)
    记录日志("=== Task8 步骤3 参数敏感性：全开 潮汐倍率 12（对照 6），前 5 条样本 ===")
    with open(样本路径, encoding="utf-8") as f:
        样本 = json.load(f)["样本"]
    随机样本 = 样本[:5]

    gc.collect()
    torch.cuda.empty_cache()
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
    模型.eval()
    库 = 锚点库(model=模型, tokenizer=分词器)
    基线 = 库.记录只读基线()
    库.构建()
    S = 库.预计算打分表()
    只读 = 库.验证只读(基线)
    记录日志(f"[锚点库] 打分表={list(S.shape)} {S.dtype} 只读校验={只读}")

    决策器 = 目标决策器(锚点库=库, β基=0.8)
    混合器 = 混合锚点器(
        模型, 分词器, 库, 决策器,
        锚点β=0.8, 锚点T=0.3, 回响λ=0.08, 潮汐倍率=12.0,
        开启A=True, 开启B=True, 开启C=True,
        温度=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
    )
    回复列表, 统计列表 = [], []
    for i, r in enumerate(随机样本):
        消息 = [{"role": "user", "content": r["user"]}]
        种子 = 42 + i
        try:
            决策器.感知器.重置轨迹()
        except Exception:  # noqa: BLE001
            pass
        torch.manual_seed(种子)
        torch.cuda.manual_seed(种子)
        提示 = 构建提示(分词器, 消息)
        inputs = 分词器(提示, return_tensors="pt").to(模型.device)
        with torch.no_grad():
            ids, 统计 = 混合器.生成(
                inputs.input_ids, max_new_tokens=256,
                eos_token_id=分词器.eos_token_id, tokenizer=分词器,
                用户文本=r["user"],
            )
        新token = ids[0, inputs.input_ids.shape[1]:]
        回复 = 分词器.decode(新token, skip_special_tokens=True).strip()
        统计["token数"] = int(len(新token))
        统计["情感命中率"] = 文本级情感命中率(回复, 库)
        回复列表.append(回复)
        统计列表.append(统计)
        记录日志(f"[倍率12 {i+1}/5] 长{len(回复)} 熵{统计['平均熵']} 重{统计['重复率']} "
                  f"情{统计['情感命中率']} 兜{统计['触发兜底次数']} "
                  f"{r['user'][:12]} => {回复[:36]}")

    健康度 = 汇总健康度(统计列表, 回复列表, 库)
    记录日志(f"[倍率12] 健康度 {json.dumps(健康度, ensure_ascii=False)}")
    with open(输出文件, "w", encoding="utf-8") as f:
        json.dump({"倍率": 12.0, "样本数": 5, "种子": 42, "配置": "全开 β=0.8 λ=0.08 倍率=12",
                   "健康度": 健康度, "统计": 统计列表, "回复": 回复列表,
                   "user": [r["user"] for r in 随机样本]},
                  f, ensure_ascii=False, indent=2)
    记录日志(f"已保存 -> {输出文件}")


if __name__ == "__main__":
    主程序()
