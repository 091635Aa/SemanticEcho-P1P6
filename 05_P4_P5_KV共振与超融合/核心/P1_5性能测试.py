# -*- coding: utf-8 -*-
"""P1~P5 性能测试 —— 模型能力评估（参考大厂 LLM 性能评测口径）

对 7 模式测：首 token 延迟(TTFT)、生成吞吐(tok/s)、平均生成耗时、
峰值显存、GPU 利用率(采nvidia-smi)、平均生成长度、每token平均耗时。
用统一生成器内部解码器循环计时（与 统一测试 同协议 seed 2026）。

用法: python P1_5性能测试.py [--模式 裸] [--样本 5]
输出: 评测结果\P1_5统一_性能_2026.json + .csv
"""
import os
import sys
import json
import time
import subprocess
from datetime import datetime

import torch

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)
回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)
if os.path.join(回响工程根, "图灵测试") not in sys.path:
    sys.path.insert(0, os.path.join(回响工程根, "图灵测试"))

from 统一生成器 import 生成器实例, 模式列表, 采样参数

评测结果 = os.path.join(本目录, "..", "评测结果")
os.makedirs(评测结果, exist_ok=True)

测试提示 = [
    "我失恋了，心里好难受，感觉整个世界都塌了。",
    "今天在公司被领导当众批评，特别委屈。",
    "我升职了！同事们都说我实至名归！",
    "你好，请问今天天气怎么样？",
    "妈妈生病住院了，我好担心她。",
]


def 采样GPU():
    """用 nvidia-smi 采样 GPU 利用率与显存（一次）"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        parts = out.stdout.strip().split(",")
        return float(parts[0]), float(parts[1])
    except Exception:
        return None, None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", default="全部")
    ap.add_argument("--样本", type=int, default=5)
    args = ap.parse_args()
    模式们 = 模式列表 if args.模式 == "全部" else [args.模式]

    生成器实例._加载()
    模型 = 生成器实例._模型
    分词器 = 生成器实例._分词器

    print(f"=== P1~P5 性能测试（种子 2026，{args.样本} 样本/模式）{datetime.now().strftime('%H:%M:%S')} ===", flush=True)

    汇总 = {}
    for 模式 in 模式们:
        print(f"── [{模式}] ──", flush=True)
        延迟列表 = []
        吞吐列表 = []
        token数列表 = []
        长度列表 = []
        显存峰值 = 0.0
        util列表 = []
        首token延迟列表 = []
        for i, 提示 in enumerate(测试提示[:args.样本]):
            torch.manual_seed(2026 + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(2026 + i)
            消息 = [{"role": "user", "content": 提示}]
            t0 = time.perf_counter()
            # 内部解码器循环计时：清空显存计数
            torch.cuda.reset_peak_memory_stats()
            if 模式 == "裸":
                inputs = 分词器(分词器.apply_chat_template(消息, tokenize=False,
                                                          add_generation_prompt=True),
                                return_tensors="pt").to(模型.device)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                with torch.no_grad():
                    out = 模型.generate(inputs.input_ids, max_new_tokens=64,
                                        pad_token_id=分词器.eos_token_id,
                                        **采样参数, do_sample=True)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                新 = out[0, inputs.input_ids.shape[1]:]
                文本 = 分词器.decode(新, skip_special_tokens=True).strip()
                # TTFT 近似 = 首 token 前向时间（prefill）
                with torch.no_grad():
                    模型(inputs.input_ids, use_cache=True)
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                首token延迟列表.append(t3 - t1)
            else:
                # 使用统一生成器内部解码器，测量完整生成耗时
                生成器实例._加载()
                解码 = 生成器实例._解码器.get(模式)
                inputs = 分词器(分词器.apply_chat_template(消息, tokenize=False,
                                                          add_generation_prompt=True),
                                return_tensors="pt").to(模型.device)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                with torch.no_grad():
                    if 模式 == "P1_语义回响":
                        out = 生成器实例._回响.生成(
                            inputs.input_ids, max_new_tokens=64,
                            eos_token_id=分词器.eos_token_id, tokenizer=分词器,
                            **采样参数)
                    elif 模式 == "P2.5_潮汐":
                        out = 解码.生成(inputs.input_ids, max_new_tokens=64,
                                       eos_token_id=分词器.eos_token_id, tokenizer=分词器,
                                       用户文本=提示, **采样参数)
                    else:
                        out, _ = 解码.生成(inputs.input_ids, max_new_tokens=64,
                                          eos_token_id=分词器.eos_token_id,
                                          用户文本=提示, **采样参数)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                新 = out[0, inputs.input_ids.shape[1]:]
                文本 = 分词器.decode(新, skip_special_tokens=True).strip()
                首token延迟列表.append(t2 - t1)
            峰值 = torch.cuda.max_memory_allocated() / 1024**3
            显存峰值 = max(显存峰值, 峰值)
            token数 = len(新)
            token数列表.append(token数)
            长度列表.append(len(文本))
            延迟列表.append(t2 - t1)
            吞吐 = token数 / max(t2 - t1, 1e-6)
            吞吐列表.append(吞吐)
            util, mem = 采样GPU()
            if util is not None:
                util列表.append(util)
            print(f"  [{i+1}] {提示[:14]}... 耗时={t2-t1:.2f}s tok={token数} 吞吐={吞吐:.1f}tok/s", flush=True)

        n = len(延迟列表)
        汇总[模式] = {
            "平均耗时(s)": round(sum(延迟列表) / n, 3) if n else 0.0,
            "平均首token延迟(s)": round(sum(首token延迟列表) / n, 3) if n else 0.0,
            "平均吞吐(tok/s)": round(sum(吞吐列表) / n, 1) if n else 0.0,
            "平均生成token数": round(sum(token数列表) / n, 1) if n else 0.0,
            "平均长度(字)": round(sum(长度列表) / n, 1) if n else 0.0,
            "峰值显存(GB)": round(显存峰值, 2),
            "平均GPU利用率(%)": round(sum(util列表) / len(util列表), 1) if util列表 else None,
            "单token平均耗时(ms)": round(sum(延迟列表) / max(sum(token数列表), 1) * 1000, 2) if token数列表 else 0.0,
        }
        print(f"  {json.dumps(汇总[模式], ensure_ascii=False)}", flush=True)

    输出 = {"模型": "Qwen2.5-1.5B-Instruct", "种子": 2026,
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "性能": 汇总,
            "协议": "TTFT近似/吞吐=新token数/耗时; 峰值显存=torch.cuda.max_memory_allocated; GPU利用率=nvidia-smi采样"}
    json路径 = os.path.join(评测结果, "P1_5统一_性能_2026.json")
    with open(json路径, "w", encoding="utf-8") as f:
        json.dump(输出, f, ensure_ascii=False, indent=2)
    # CSV
    import csv
    csv路径 = os.path.join(评测结果, "P1_5统一_性能_2026.csv")
    with open(csv路径, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["模式", "平均耗时(s)", "TTFT(s)", "吞吐(tok/s)", "平均token", "平均长度(字)", "峰值显存(GB)", "GPU利用率(%)", "单token(ms)"])
        for 模式 in 模式们:
            p = 汇总[模式]
            w.writerow([模式, p["平均耗时(s)"], p["平均首token延迟(s)"], p["平均吞吐(tok/s)"],
                        p["平均生成token数"], p["平均长度(字)"], p["峰值显存(GB)"],
                        p["平均GPU利用率(%)"], p["单token平均耗时(ms)"]])
    print(f"已保存：{json路径} / {csv路径}")


if __name__ == "__main__":
    main()
