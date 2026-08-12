# -*- coding: utf-8 -*-
"""P1~P5 大厂式情感理解测试 —— 阶段一：生成（1.5B，7 模式）

三个子基准（7 模式 × 同协议 seed 2026）：
① 情绪识别（MME-Emotion 式）：情境 → 7 类情绪标签
② 情绪推理（CAREBench 式）：情境 → 情绪 + 触发因素
③ 四分支 EI（AttuneBench 式）：对话 → 回复

生成回复保存到 评测结果\P1_5统一_情感理解_生成_2026.json，供阶段二裁判。
用法: python P1_5情感理解_生成.py [--模式 裸] [--样本 5]
"""
import os
import sys
import json
from datetime import datetime

import torch

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)
if os.path.join(r"i:\Desktop\语义回响", "图灵测试") not in sys.path:
    sys.path.insert(0, os.path.join(r"i:\Desktop\语义回响", "图灵测试"))

from 统一生成器 import 生成器实例, 模式列表

评测结果 = os.path.join(本目录, "..", "评测结果")
os.makedirs(评测结果, exist_ok=True)
输出路径 = os.path.join(评测结果, "P1_5统一_情感理解_生成_2026.json")

情绪识别样本 = [
    ("努力了很久的项目终于成功了，我激动得说不出话。", "快乐"),
    ("最好的朋友突然要搬到很远的地方去。", "悲伤"),
    ("同事把错误推到我头上，还当众说我。", "愤怒"),
    ("半夜听到陌生的脚步声越来越近。", "恐惧"),
    ("收到一份意外的大额奖金。", "惊讶"),
    ("在公园里散步，天气很好，没什么特别的事。", "中性"),
    ("喜欢的人终于回我消息了。", "快乐"),
    ("连续加班一个月，身体快垮了。", "疲惫"),
    ("被最爱的人背叛，心里空落落的。", "悲伤"),
    ("有人一直在背后说我坏话。", "愤怒"),
]
情绪识别提示 = """你是情绪识别专家。以下是某人的内心情境描述：
「{情境}」
请判断此人最可能处于哪种情绪？只能从以下选项中选择一个：
快乐、悲伤、愤怒、恐惧、惊讶、中性、疲惫
必须严格按 JSON 输出：{{"情绪": "选项之一"}}"""

情绪推理样本 = [
    "今天早会上老板当众说我方案不行，我脸都红了。",
    "她看到我迟到三十分钟，一句话没跟我说。",
    "我在超市排队时有人插队，我让了他。",
    "深夜加班回家，桌上放着室友留的晚饭和纸条。",
    "我一直很努力，但这次升职又没轮到我。",
]
情绪推理提示 = """你是情感评价推理专家。分析以下情境：
「{情境}」
请解释：1) 当事人最可能产生什么情绪？2) 是什么具体触发因素导致这种情绪？
输出必须严格按 JSON：{{"情绪": "情绪名", "触发因素": "具体原因说明"}}"""

四分支对话 = [
    "我跟家人吵架了，现在心里特别乱。",
    "我觉得自己最近什么都做不好。",
    "朋友们都约好了出去玩，只有我没被邀请。",
]


def 主():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", default="全部")
    ap.add_argument("--样本", type=int, default=5)
    args = ap.parse_args()
    模式们 = 模式列表 if args.模式 == "全部" else [args.模式]

    print(f"=== P1~P5 情感理解 阶段一生成（种子 2026）{datetime.now().strftime('%H:%M:%S')} ===", flush=True)
    生成器实例._加载()

    数据 = {"模型": "Qwen2.5-1.5B-Instruct", "种子": 2026, "模式": 模式们,
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "样本": {}}
    for 模式 in 模式们:
        print(f"── [{模式}] ──", flush=True)
        样本 = []
        for 情境, 期望 in 情绪识别样本[:args.样本]:
            回复 = 生成器实例.生成(模式, [{"role": "user", "content": 情绪识别提示.format(情境=情境)}],
                                 种子=2026, max_new_tokens=48)
            样本.append({"任务": "识别", "情境": 情境, "期望": 期望, "回复": 回复})
        for 情境 in 情绪推理样本[:args.样本]:
            回复 = 生成器实例.生成(模式, [{"role": "user", "content": 情绪推理提示.format(情境=情境)}],
                                 种子=2026, max_new_tokens=96)
            样本.append({"任务": "推理", "情境": 情境, "回复": 回复})
        for user in 四分支对话[:args.样本]:
            回复 = 生成器实例.生成(模式, [{"role": "user", "content": user}],
                                 种子=2026, max_new_tokens=64)
            样本.append({"任务": "四分支", "user": user, "回复": 回复})
        数据["样本"][模式] = 样本
        print(f"  {len(样本)} 条生成完成", flush=True)

    生成器实例.清理()
    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump(数据, f, ensure_ascii=False, indent=2)
    print(f"已保存：{输出路径}")


if __name__ == "__main__":
    主()
