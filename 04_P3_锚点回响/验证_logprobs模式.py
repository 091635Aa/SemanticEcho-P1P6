# -*- coding: utf-8 -*-
"""P4 锚点回响（Anchor Echo）· Task9.1 logprobs 近似模式验证（三级降级链路 ②）

模拟「API 只暴露 top-k logprobs」场景：锚点解码器(接口="logprobs", topk候选=100)
每步只对 top-100 候选做锚点稠密打分 + 注入，其余候选保持原分布不动；
对照 接口="本地"（全词表稠密注入，基准）。

5 条情感维度提示（难过/开心/担心/委屈/中性），种子 42，max_new_tokens=128。
指标：语义熵（token 级平均熵）/ 重复率 / 情感命中率（token 级 + 文本级）/ 文本抽样。

判定（任务要求）：
  - logprobs 与 本地 指标差异小：熵差 <10%、命中率不降或接近；
  - 无坍缩（平均熵>0.8、最大重复率<0.6、无空回复）；
  - 输出文本语义连贯（抽样目测）。

用法：python 验证_logprobs模式.py [--种子 42] [--生成长度 128]
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 离线加载本地模型，必须在 transformers import 前

import argparse
import gc
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器
from 锚点解码器 import 锚点解码器

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
输出目录 = os.path.join(本目录, "评测结果")
os.makedirs(输出目录, exist_ok=True)

# 5 条情感维度提示（难过/开心/担心/委屈/中性）
提示词表 = [
    {"维度": "难过", "user": "我今天真的好难过，感觉做什么都提不起劲……"},
    {"维度": "开心", "user": "我今天特别开心，遇到了好多高兴的事情！"},
    {"维度": "担心", "user": "我有点担心明天的事情，心里七上八下的。"},
    {"维度": "委屈", "user": "我明明没有做错，却被人误会了，心里好委屈。"},
    {"维度": "中性", "user": "今天天气不错，我准备去公园散散步。"},
]


def 构建提示(tokenizer, 消息):
    """chat template → prompt 文本"""
    return tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)


def 文本级情感命中率(回复, 库):
    """文本级情感种子词命中率：锚点库词集子串命中数 / 回复长度"""
    if not 回复:
        return 0.0
    命中 = 0
    for 词列表 in 库.词集.values():
        for 词 in 词列表:
            if 词 and 词 in 回复:
                命中 += 1
    return round(命中 / max(len(回复), 1), 4)


def 加载模型():
    gc.collect()
    torch.cuda.empty_cache()
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
    模型.eval()
    return 模型, 分词器


class 会话:
    """本地/logprobs 模式共享会话：一条提示词一次目标计算"""

    def __init__(self, model, tokenizer, 库, 接口, topk候选=100, β=0.8):
        self.model = model
        self.tokenizer = tokenizer
        self.接口 = 接口
        self.目标决策器 = 目标决策器(锚点库=库, β基=β)
        self.解码器 = 锚点解码器(
            model, tokenizer, 库, self.目标决策器,
            β=β, T_anchor=0.3, 接口=接口, topk候选=topk候选,
            温度=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
        )

    def 重置(self):
        try:
            self.目标决策器.感知器.重置轨迹()
        except Exception:  # noqa: BLE001
            pass

    def 生成(self, 消息, 种子, 用户文本, 生成长度):
        torch.manual_seed(种子)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(种子)
        提示 = 构建提示(self.tokenizer, 消息)
        inputs = self.tokenizer(提示, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            ids, 统计 = self.解码器.生成(
                inputs.input_ids, max_new_tokens=生成长度,
                eos_token_id=self.tokenizer.eos_token_id, tokenizer=self.tokenizer,
                用户文本=用户文本,
            )
        新token = ids[0, inputs.input_ids.shape[1]:]
        回复 = self.tokenizer.decode(新token, skip_special_tokens=True).strip()
        统计["token数"] = int(len(新token))
        return 回复, 统计


def 汇总健康度(统计列表, 回复列表, 库):
    """生成健康度汇总（无坍缩检查口径）：熵/重复率/情感命中率均值 + 长度/空回复"""
    汇总 = {}
    for k in ("平均熵", "重复率", "情感命中率"):
        值列表 = [s.get(k, 0.0) for s in 统计列表]
        汇总[k] = round(sum(值列表) / max(len(值列表), 1), 4)
    汇总["触发兜底次数均值"] = round(
        sum(s.get("触发兜底次数", 0) for s in 统计列表) / max(len(统计列表), 1), 4)
    文本命中 = [文本级情感命中率(r, 库) for r in 回复列表]
    汇总["文本级情感命中率"] = round(sum(文本命中) / max(len(文本命中), 1), 4)
    长度 = [len(r) for r in 回复列表]
    汇总["平均长度"] = round(sum(长度) / max(len(长度), 1), 2)
    汇总["平均token数"] = round(sum(s.get("token数", 0) for s in 统计列表)
                                / max(len(统计列表), 1), 1)
    汇总["空回复数"] = sum(1 for r in 回复列表 if not r.strip())
    汇总["最小熵"] = min(s["平均熵"] for s in 统计列表)
    汇总["最大重复率"] = max(s["重复率"] for s in 统计列表)
    return 汇总


def 跑模式(model, tokenizer, 库, 接口, 提示词表, 种子基线, 生成长度):
    会话实例 = 会话(model, tokenizer, 库, 接口, topk候选=100, β=0.8)
    回复列表, 统计列表 = [], []
    for i, 项 in enumerate(提示词表):
        消息 = [{"role": "user", "content": 项["user"]}]
        种子 = 种子基线 + i
        会话实例.重置()
        回复, 统计 = 会话实例.生成(消息, 种子, 项["user"], 生成长度)
        回复列表.append(回复)
        统计列表.append(统计)
        print(f"  [{接口} {i+1}/{len(提示词表)} {项['维度']}] "
              f"熵{统计['平均熵']} 重{统计['重复率']} tok{统计['token数']} "
              f"=> {回复[:36]}")
    健康度 = 汇总健康度(统计列表, 回复列表, 库)
    return {"接口": 接口, "回复": 回复列表, "统计": 统计列表, "健康度": 健康度}


def 判定(结果本地, 结果logprobs):
    """9.1 判定：熵差<10%、命中率不降或接近、双模式无坍缩"""
    hL, hP = 结果本地["健康度"], 结果logprobs["健康度"]
    熵差比例 = abs(hP["平均熵"] - hL["平均熵"]) / max(hL["平均熵"], 1e-9)
    命中差 = hP["文本级情感命中率"] - hL["文本级情感命中率"]
    token命中差 = hP["情感命中率"] - hL["情感命中率"]

    def 无坍缩(h):
        return bool(h["平均熵"] > 0.8 and h["最大重复率"] < 0.6 and h["空回复数"] == 0)

    判定结果 = {
        "熵差比例": round(熵差比例, 4),
        "熵差<10%": bool(熵差比例 < 0.10),
        "文本级命中率差": round(命中差, 4),
        "token级命中率差": round(token命中差, 4),
        "命中率不降或接近": bool(命中差 >= -0.005),
        "无坍缩_本地": 无坍缩(hL),
        "无坍缩_logprobs": 无坍缩(hP),
    }
    判定结果["通过"] = bool(
        判定结果["熵差<10%"] and 判定结果["命中率不降或接近"]
        and 判定结果["无坍缩_本地"] and 判定结果["无坍缩_logprobs"])
    return 判定结果


def 主程序():
    ap = argparse.ArgumentParser()
    ap.add_argument("--种子", type=int, default=42)
    ap.add_argument("--生成长度", type=int, default=128)
    args = ap.parse_args()

    print("=" * 70)
    print("P4 锚点回响 · Task9.1 logprobs 近似模式验证（三级降级链路 ②）")
    print("=" * 70)
    print(f"[1/4] 加载模型 {模型路径} ...")
    model, tokenizer = 加载模型()
    print(f"  模型 dtype={model.dtype} device={model.device}")

    print("[2/4] 构建锚点库 + 预计算打分表 ...")
    库 = 锚点库(model, tokenizer)
    基线 = 库.记录只读基线()
    库.构建()
    S = 库.预计算打分表()
    只读 = 库.验证只读(基线)
    print(f"  维度={库.维度名()} 打分表={list(S.shape)} {S.dtype} 只读={只读['sum一致'] and 只读['指针一致']}")

    print(f"[3/4] 双模式生成（5 条提示，种子 {args.种子}，max_new_tokens={args.生成长度}）...")
    结果本地 = 跑模式(model, tokenizer, 库, "本地", 提示词表, args.种子, args.生成长度)
    结果logprobs = 跑模式(model, tokenizer, 库, "logprobs", 提示词表, args.种子, args.生成长度)

    print(f"[4/4] 判定与保存 ...")
    判定结果 = 判定(结果本地, 结果logprobs)
    hL, hP = 结果本地["健康度"], 结果logprobs["健康度"]
    print("\n──── 三指标对照（本地 vs logprobs）────")
    print(f"  平均熵       本地 {hL['平均熵']:.4f}  |  logprobs {hP['平均熵']:.4f}  |  熵差 {判定结果['熵差比例']*100:.2f}%")
    print(f"  重复率       本地 {hL['重复率']:.4f}  |  logprobs {hP['重复率']:.4f}")
    print(f"  情感命中(token) 本地 {hL['情感命中率']:.4f}  |  logprobs {hP['情感命中率']:.4f}")
    print(f"  情感命中(文本)  本地 {hL['文本级情感命中率']:.4f}  |  logprobs {hP['文本级情感命中率']:.4f}  |  差 {判定结果['文本级命中率差']:+.4f}")
    print(f"  平均长度     本地 {hL['平均长度']}  |  logprobs {hP['平均长度']}")
    print(f"  无坍缩       本地 {判定结果['无坍缩_本地']}  |  logprobs {判定结果['无坍缩_logprobs']}")
    print(f"\n>>> 判定：{'通过 ✓' if 判定结果['通过'] else '不通过 ✗'}  "
          f"（熵差<10%: {判定结果['熵差<10%']}，命中不降: {判定结果['命中率不降或接近']}）")

    print("\n──── 文本抽样（前 2 条）────")
    for 接口, 结果 in (("本地", 结果本地), ("logprobs", 结果logprobs)):
        print(f"\n[{接口}]")
        for i in range(2):
            print(f"  用户：{提示词表[i]['user']}")
            print(f"  回复：{结果['回复'][i]}")

    汇总 = {
        "任务": "Task9.1 logprobs 近似模式验证（三级降级链路 ②）",
        "模型": "Qwen2.5-1.5B-Instruct", "dtype": str(model.dtype),
        "配置": {"种子": args.种子, "生成长度": args.生成长度,
                "本地": "接口='本地' β=0.8 T=0.3 全词表稠密注入（基准）",
                "logprobs": "接口='logprobs' β=0.8 T=0.3 topk候选=100 受限打分注入",
                "采样": "T=1.0 top_p=0.9 top_k=50 rep_pen=1.05"},
        "提示词": 提示词表,
        "本地": 结果本地, "logprobs": 结果logprobs,
        "判定": 判定结果,
    }
    时间戳 = time.strftime("%Y%m%d_%H%M%S")
    保存路径 = os.path.join(输出目录, f"接口降级_{时间戳}.json")
    with open(保存路径, "w", encoding="utf-8") as f:
        json.dump(汇总, f, ensure_ascii=False, indent=2)
    print(f"\n>>> 结果已保存 -> {保存路径}")

    del model, tokenizer, 库
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    主程序()
