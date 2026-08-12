# -*- coding: utf-8 -*-
"""待办自动执行守护 — 等待系统 RAM 充足后按序完成遗留任务

背景：剪映等应用占用 RAM（2.7GB+），3B/7B 模型加载需 4-5GB CPU 缓冲，
加载分片时 torch_cpu.dll 原生崩溃（0xC0000005）。本脚本每 5 分钟检测
一次可用 RAM，≥5GB 时按序执行：

  1. T5-T6 动态策略校准复测（test_dynamic.py 3B Q4，λ 增量 0.02）
     → 结果备份为 dynamic_策略_v2.csv
  2. T12 7B LoRA 验证（run_emotion_lora_test.py --model 7B --adapter emotion_7B）

完成一项后立即运行下一项；全部完成则退出。
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

工作目录 = Path(r"f:\lora外挂\evaluation")
实验报告 = Path(r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\06_测试文件夹\实验报告")
框架验证 = 实验报告 / "框架验证"
脚本目录 = Path(r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\06_测试文件夹\stress_tests")
日志文件 = 工作目录 / "7B重试.log"
python = r"c:\Users\Administrator\Documents\论文+临时目录\星拟图工程\.venv\Scripts\python.exe"

结果7B = 工作目录 / "emotion_emotion_7B_test_7B.json"
标记校准 = 工作目录 / "dynamic_v2_done.flag"


def 写日志(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(日志文件, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}", flush=True)


def 可用RAM_GB():
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEMORYSTATUSEX(); m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / (1024 ** 3)


def 跑(名称, 参数列表, 输出日志, 输出错误):
    写日志(f"{名称}: 开始")
    try:
        结果 = subprocess.run(
            [python, "-u"] + [str(a) for a in 参数列表],
            cwd=str(脚本目录), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=None,
        )
        写日志(f"{名称}: 退出码 {结果.returncode}")
        if 结果.stdout:
            输出日志.write_text(结果.stdout, encoding="utf-8")
        if 结果.returncode != 0 and 结果.stderr:
            输出错误.write_text(结果.stderr, encoding="utf-8")
            写日志(f"{名称}: stderr 摘要 {结果.stderr[-300:]}")
        return 结果.returncode == 0
    except Exception as e:
        写日志(f"{名称}: 执行异常 {type(e).__name__}: {e}")
        return False


写日志("待办自动执行守护启动（等待 RAM ≥5GB）：①动态策略校准复测 ②7B LoRA 验证")
while True:
    ram = 可用RAM_GB()

    # ① 动态策略校准复测（未完成时）
    if not 标记校准.exists() and ram >= 5.0:
        写日志(f"可用 RAM {ram:.1f}GB，执行动态策略校准复测 ...")
        ok = 跑("T5T6动态校准", [脚本目录 / "test_dynamic.py", "--quant", "4bit"],
                实验报告 / "STEP7c_test_dynamic_v2.log", 实验报告 / "STEP7c_test_dynamic_v2.err")
        # 备份结果（v2 = 校准后最终数据）
        src = 框架验证 / "dynamic_策略.csv"
        if src.exists():
            import shutil
            shutil.copy2(src, 框架验证 / "dynamic_策略_v2.csv")
            写日志("已备份 dynamic_策略_v2.csv")
        标记校准.write_text("done", encoding="utf-8")
        continue

    # ② 7B LoRA 验证（未完成时）
    if not 结果7B.exists() and ram >= 5.0:
        写日志(f"可用 RAM {ram:.1f}GB，执行 7B LoRA 验证 ...")
        ok = 跑("7BLoRA验证", [工作目录 / "run_emotion_lora_test.py", "--model", "7B", "--adapter", "emotion_7B"],
                工作目录 / "test_7B_auto.log", 工作目录 / "test_7B_auto.err")
        if ok and 结果7B.exists():
            写日志("7B LoRA 验证完成 ✅，守护退出")
            break
        else:
            写日志(f"7B 验证失败（RAM {ram:.1f}GB 可能不足），继续等待 ...")
            time.sleep(300)
            continue

    if 标记校准.exists() and 结果7B.exists():
        写日志("全部待办完成，守护退出")
        break

    time.sleep(300)
