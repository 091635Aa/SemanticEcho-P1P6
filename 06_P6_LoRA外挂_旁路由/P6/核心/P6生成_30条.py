# -*- coding: utf-8 -*-
"""P6 评测·生成阶段：30 条样本 × 三模式（裸 / P6_LoRA裸 / P6_旁路由）

协议与统一基准一致：种子=2026+i、T=1.0/top_p=0.9/top_k=50/rep=1.05、chat 模板（前奏零修改）。
裸 = 未挂 LoRA 的标准采样（协议校验基线）；P6_LoRA裸 = 挂 LoRA 单候选（消融）；
P6_旁路由 = 挂 LoRA + N 候选情感路由选优（完整 P6）。生成缓存供 7B 裁判阶段。
支持断点续跑（每 5 样本增量保存）。
"""
import json
import os
import sys
import time
from datetime import datetime

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from P6旁路由 import P6旁路由生成器

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
目标模型名 = "Qwen2.5-1.5B-Instruct"
模型路径 = os.path.join(模型空间, 目标模型名)
lora路径 = r"f:\lora外挂\lora_adapters\p6_emotion"

样本路径 = r"i:\Desktop\语义回响\图灵测试\样本_30条.json"
输出目录 = os.path.join(本目录, "..", "评测结果")
生成路径 = os.path.join(输出目录, "P6_生成_30.json")

模式列表 = ["裸", "P6_LoRA裸", "P6_旁路由"]
种子基础 = 2026
N候选 = 3


def 保存(回复):
    结果 = {
        "模型": 目标模型名, "LoRA": "p6_emotion", "模式": 模式列表,
        "模式种子": {"基础": 种子基础, "候选步长": 7, "N": N候选},
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "回复": 回复,
    }
    with open(生成路径, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=2)


def 主():
    os.makedirs(输出目录, exist_ok=True)
    with open(样本路径, encoding="utf-8") as f:
        数据 = json.load(f)
    全部样本 = 数据["样本"]
    print(f"=== P6 生成阶段 {datetime.now().strftime('%H:%M:%S')} 样本数={len(全部样本)} ===", flush=True)

    # 断点续跑
    回复 = None
    if os.path.exists(生成路径):
        try:
            旧 = json.load(open(生成路径, encoding="utf-8"))
            if 旧.get("模型") == 目标模型名:
                回复 = 旧["回复"]
                print(f"断点续跑：已有 {sum(1 for s in 回复 if 'P6_旁路由' in s['回复'])}/{len(回复)} 样本完成", flush=True)
        except Exception:  # noqa: BLE001
            pass
    if 回复 is None:
        回复 = [{"序号": 项["序号"], "user": 项["user"], "girl": 项["girl"], "回复": {}}
                for 项 in 全部样本]

    待做 = [s for s in 全部样本 if "P6_旁路由" not in 回复[s["序号"] - 1]["回复"]]
    if not 待做:
        print("全部样本已完成，跳过生成", flush=True)
        return

    # P6 模式（挂 LoRA）
    p6 = P6旁路由生成器(模型路径, lora路径, 挂载=True)
    for i, 项 in enumerate(待做):
        序号 = 项["序号"]
        消息 = [{"role": "user", "content": 项["user"]}]
        种子 = 种子基础 + 序号 - 1
        # P6_LoRA裸（单候选消融）
        if "P6_LoRA裸" not in 回复[序号 - 1]["回复"]:
            t0 = time.time()
            文本, tokens, 统计 = p6.生成(消息, 种子=种子, N=1, 返回候选=False)
            回复[序号 - 1]["回复"]["P6_LoRA裸"] = {"文本": 文本, "统计": 统计}
            print(f"  [{序号}] P6_LoRA裸 {time.time()-t0:.1f}s len={len(文本)}", flush=True)
        # P6_旁路由（完整 P6）
        t0 = time.time()
        文本, tokens, 统计, 候选 = p6.生成(消息, 种子=种子, N=N候选, 返回候选=True)
        回复[序号 - 1]["回复"]["P6_旁路由"] = {
            "文本": 文本, "统计": 统计,
            "候选": [{"种子": c["种子"], "文本": c["文本"], "分数": c["分数"]} for c in 候选],
        }
        print(f"  [{序号}] P6_旁路由 {time.time()-t0:.1f}s len={len(文本)} 路由分={统计['路由分数']}", flush=True)
        if (i + 1) % 5 == 0:
            保存(回复)
    p6.清理()
    保存(回复)

    # 裸基线（不挂 LoRA，协议校验）
    缺裸 = [s for s in 全部样本 if "裸" not in 回复[s["序号"] - 1]["回复"]]
    if 缺裸:
        裸生成器 = P6旁路由生成器(模型路径, lora路径, 挂载=False)
        for 项 in 缺裸:
            序号 = 项["序号"]
            消息 = [{"role": "user", "content": 项["user"]}]
            种子 = 种子基础 + 序号 - 1
            t0 = time.time()
            文本, tokens, 统计 = 裸生成器.生成(消息, 种子=种子, N=1, 返回候选=False)
            回复[序号 - 1]["回复"]["裸"] = {"文本": 文本, "统计": 统计}
            print(f"  [{序号}] 裸 {time.time()-t0:.1f}s len={len(文本)}", flush=True)
        裸生成器.清理()
        保存(回复)

    print(f"生成完成，已保存：{生成路径}", flush=True)


if __name__ == "__main__":
    主()
