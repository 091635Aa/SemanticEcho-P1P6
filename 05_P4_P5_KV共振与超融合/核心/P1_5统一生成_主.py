# -*- coding: utf-8 -*-
"""P1~P5 统一测试 —— 生成阶段主控（3 worker 并行 + 合并）

把 30 条样本切成 3 份，3 个进程同时各自加载 1.5B 基座生成（吃满 CUDA），
完成后合并成统一生成 JSON：评测结果\P1_5统一_生成_30_2026.json
"""
import os
import sys
import json
import subprocess
from datetime import datetime

工作目录 = os.path.dirname(os.path.abspath(__file__))
输出目录 = os.path.join(工作目录, "..", "评测结果")
os.makedirs(输出目录, exist_ok=True)
最终路径 = os.path.join(输出目录, "P1_5统一_生成_30_2026.json")
worker脚本 = os.path.join(工作目录, "P1_5统一生成worker.py")
python = sys.executable

切片数 = 2
每片 = 15  # 30 // 2

模式列表 = ["裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响", "P4_KV共振", "P5_超融合"]


def main():
    print(f"=== P1~P5 统一生成（种子 2026，{切片数} worker 并行）{datetime.now().strftime('%H:%M:%S')} ===", flush=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    进程列表 = []
    输出表 = []
    for i in range(切片数):
        start = i * 每片
        end = min(start + 每片, 30)
        part = os.path.join(输出目录, f"P1_5统一_part{i}.json")
        输出表.append(part)
        cmd = [python, worker脚本, "--start", str(start), "--end", str(end), "--out", part]
        print(f"启动 worker{i}: 样本[{start}, {end}) → {part}", flush=True)
        进程列表.append(subprocess.Popen(cmd, cwd=工作目录,
                                         stdout=None, stderr=None,
                                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP))

    # 等待全部完成
    for idx, p in enumerate(进程列表):
        code = p.wait()
        print(f"worker{idx} 退出码={code}", flush=True)
        if code != 0:
            print(f"⚠ worker{idx} 失败，检查其输出日志", flush=True)

    # 合并
    合并结果 = {"模型": "Qwen2.5-1.5B-Instruct", "种子": 2026,
                "模式": 模式列表, "模式种子": {m: 2026 for m in 模式列表},
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "回复": []}
    缺失模式 = set()
    for part in 输出表:
        if not os.path.exists(part):
            print(f"⚠ 缺失 {part}，跳过", flush=True)
            continue
        with open(part, "r", encoding="utf-8") as f:
            数据 = json.load(f)
        for 条目 in 数据["回复"]:
            for m in 模式列表:
                if m not in 条目["回复"]:
                    缺失模式.add(m)
            合并结果["回复"].append(条目)

    合并结果["回复"].sort(key=lambda x: x["序号"])
    with open(最终路径, "w", encoding="utf-8") as f:
        json.dump(合并结果, f, ensure_ascii=False, indent=2)
    print(f"合并完成：{len(合并结果['回复'])} 条样本 → {最终路径}", flush=True)
    if 缺失模式:
        print(f"⚠ 缺失模式：{缺失模式}", flush=True)
    # 清理 part 文件
    for part in 输出表:
        if os.path.exists(part):
            os.remove(part)


if __name__ == "__main__":
    main()
