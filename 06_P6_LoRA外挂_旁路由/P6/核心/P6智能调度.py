# -*- coding: utf-8 -*-
"""P6 智能调度器（GPU 空闲才运行）

规则（按用户要求）：
- 仅当 GPU 空闲（显存 < 2000MiB 且利用率 < 10%，连续 2 次间隔确认）才运行下一步骤；
- 空闲后 10 分钟内自动开始执行；GPU 忙 → 等待，绝不抢占；
- 步骤间状态落盘（P6_调度状态.json），随时可重启续跑；每步均支持断点续跑；
- 若存在"让路"标记文件（C:\\P6临时盘\\让路.flag），当前步骤结束后暂停并等待其删除。

调度步骤：生成(30条) → 裁判(7B盲评) → 汇总(对比)
用法：python P6智能调度.py [--间隔 60] [--空闲阈值MB 2000]
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

本目录 = os.path.dirname(os.path.abspath(__file__))
状态路径 = os.path.join(本目录, "P6_调度状态.json")
让路标记 = r"C:\P6临时盘\让路.flag"
日志路径 = os.path.join(本目录, "P6_调度.log")

步骤 = [
    {"名": "生成", "cmd": [sys.executable, os.path.join(本目录, "P6生成_30条.py")],
     "日志": os.path.join(本目录, "P6生成_30.log"), "需要GPU": True},
    {"名": "裁判", "cmd": [sys.executable, os.path.join(本目录, "P6裁判_30条.py")],
     "日志": os.path.join(本目录, "P6裁判_30.log"), "需要GPU": True},
    {"名": "汇总", "cmd": [sys.executable, os.path.join(本目录, "P6汇总.py")],
     "日志": os.path.join(本目录, "P6汇总.log"), "需要GPU": False},
]


def 写日志(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def 读取GPU():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        m = re.search(r"([\d.]+),\s*([\d.]+)", out)
        if m:
            return float(m.group(1)), float(m.group(2))
    except Exception as e:  # noqa: BLE001
        写日志(f"nvidia-smi 失败：{e}")
    return None, None


def GPU空闲(阈值MB):
    used, util = 读取GPU()
    if used is None:
        return False
    return used < 阈值MB and util < 10.0


def 运行步骤(step, 阈值MB):
    cmd = step["cmd"]
    日志 = step["日志"]
    写日志(f"开始步骤 [{step['名']}] {os.path.basename(cmd[1])}")
    with open(日志, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=本目录)
        try:
            proc.wait(timeout=None)
        except KeyboardInterrupt:
            proc.terminate()
            raise
    ok = (proc.returncode == 0)
    写日志(f"步骤 [{step['名']}] 结束 rc={proc.returncode}")
    return ok


def 读状态():
    if os.path.exists(状态路径):
        try:
            return json.load(open(状态路径, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {s["名"]: False for s in 步骤}


def 保存状态(状态):
    with open(状态路径, "w", encoding="utf-8") as f:
        json.dump(状态, f, ensure_ascii=False, indent=2)


def 主():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--间隔", type=int, default=60, help="GPU 探测间隔秒")
    parser.add_argument("--空闲阈值MB", type=int, default=2000)
    args = parser.parse_args()

    状态 = 读状态()
    写日志(f"P6 调度器启动：间隔={args.间隔}s 空闲阈值={args.空闲阈值MB}MB "
           f"状态={状态}（让路标记：{'存在' if os.path.exists(让路标记) else '无'}）")

    空闲计数 = 0
    while True:
        # 让路标记：用户紧急任务要用 GPU → 暂停
        if os.path.exists(让路标记):
            写日志("检测到让路标记，暂停等待（删除标记后自动恢复）")
            while os.path.exists(让路标记):
                time.sleep(args.间隔)
            写日志("让路标记已解除，恢复探测")
            continue

        未完成 = [s for s in 步骤 if not 状态.get(s["名"], False)]
        if not 未完成:
            写日志("全部步骤完成，退出")
            break

        if GPU空闲(args.空闲阈值MB):
            空闲计数 += 1
            写日志(f"GPU 空闲确认 {空闲计数}/2")
            if 空闲计数 >= 2:
                空闲计数 = 0
                step = 未完成[0]
                if step["需要GPU"]:
                    # 启动前再确认一次仍空闲（避免探测后立刻被抢占）
                    time.sleep(3)
                    if not GPU空闲(args.空闲阈值MB):
                        写日志("启动前 GPU 被占用，让行")
                        continue
                ok = 运行步骤(step, args.空闲阈值MB)
                if ok:
                    状态[step["名"]] = True
                    保存状态(状态)
                    写日志(f"步骤 [{step['名']}] 完成，状态={状态}")
                else:
                    写日志(f"步骤 [{step['名']}] 失败（可断点续跑），等待下一轮重试")
                time.sleep(5)
        else:
            空闲计数 = 0
            used, util = 读取GPU()
            写日志(f"GPU 忙（used={used}MB util={util}%），等待 {args.间隔}s")
            time.sleep(args.间隔)


if __name__ == "__main__":
    主()
