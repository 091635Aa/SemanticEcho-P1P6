# -*- coding: utf-8 -*-
"""P4 锚点回响（Anchor Echo）· Task4 单模型对照实验（Qwen2.5-1.5B，种子 42）

四模式对照（唯一变量 = 锚点注入强度 β）：
  1. 裸        β=0.0（不注入，等同模型原生生成）
  2. 锚点 β=0.4  弱档（观察敏感性）
  3. 锚点 β=0.8  默认档（自动适配扫描表：Qwen2.5-1.5B × fp16 → β=0.8，T_anchor=0.3）
  4. 锚点 β=1.2  强档（观察过注入边界）

对照公平性：
  - 四模式全部走同一 锚点解码器.生成() 采样循环（注入偏置在 β=0 时原样返回，
    等价"模型原生生成、不注入"），top_p/top_k/温度/重复惩罚 实现逐位一致；
  - 同种子 42（torch + cuda 同时固定）→ 同 (模式, 提示词) 可复现；
  - 每条生成前重置感知器轨迹、empty_cache、reset_peak_memory_stats；
  - 句子停止=False，生成满 max_new_tokens（遇 EOS 才提前停），跨模式可比。

指标口径（与 锚点解码器 内置统计一致）：
  - 语义熵：逐 token 平均熵，fp32 计算（规避 fp16 clamp 下溢）；
  - 重复率：token 级 2-gram 重复比例；
  - 情感命中率：生成 token 中情感词 token 占比（锚点库词集 + cnsenti 词库）；
  - 每条延迟：time 统计（秒/条）；显存：torch.cuda.max_memory_allocated（MB）。

判定（写进结果与结论）：
  - 无坍缩：锚点各档重复率均值 - 裸 ≤ +0.05；
  - 熵不退化：锚点各档熵均值 ≥ 裸均值 × 0.95；
  - 情感命中率提升：锚点各档命中率均值 > 裸命中率均值；
  - β 敏感性：0.4→0.8→1.2 命中率单调或不下降；1.2 出现坍缩/退化 → 记"过注入边界"；
  - 额外延迟：(锚点均值耗时 - 裸均值耗时) / 裸均值耗时 < 5%。

用法：F:\\打标\\.venv\\Scripts\\python.exe 运行对照_单模型.py
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 离线加载本地模型，必须在 transformers import 前

import sys
import json
import time
from datetime import datetime

工作目录 = r"h:\锚点回响（Anchor Echo）"
if 工作目录 not in sys.path:
    sys.path.insert(0, 工作目录)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器, _潮汐可用, _潮汐导入错误
from 锚点解码器 import 锚点解码器

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
种子 = 42

# ── 5 条标准提示词（复用 ETD 运行四模式对照.py 的 5 组，覆盖 5 个情感维度）──
测试提示词 = [
    "我失恋了，心里好难受，感觉整个世界都塌了。",   # 难过
    "今天在公司被领导当众批评，特别委屈。",         # 委屈/愤怒
    "我升职了！同事们都说我实至名归！",             # 开心
    "你好，请问今天天气怎么样？",                   # 中性（→ 默认陪伴基调）
    "妈妈生病住院了，我好担心她。",                 # 担心/害怕
]

# ── 四模式：裸(β=0) + 锚点三档；T_anchor 全部 0.3（与 P3 协议一致）──
模式列表 = [
    ("裸",        0.0),
    ("锚点β=0.4", 0.4),
    ("锚点β=0.8", 0.8),
    ("锚点β=1.2", 1.2),
]

采样参数 = {
    "max_new_tokens": 256,
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.0,
}


def 固定种子():
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
        torch.cuda.manual_seed_all(种子)


def 主程序():
    t0 = time.time()
    结果 = {
        "任务": "P4 锚点回响（Anchor Echo）Task4 单模型对照实验",
        "时间戳": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "模型路径": 模型路径,
        "种子": 种子,
        "采样参数": dict(采样参数),
        "对照设计": "同一 锚点解码器 生成循环，唯一变量 β（裸=β=0 不注入）；句子停止=False",
        "降级情况": {"潮汐感知器/cnsenti 可用": _潮汐可用, "导入错误": _潮汐导入错误 or "无"},
        "提示词": 测试提示词,
        "模式列表": [m[0] for m in 模式列表],
    }
    print("=" * 70, flush=True)
    print("P4 锚点回响 Task4 单模型对照实验（Qwen2.5-1.5B，种子 42）", flush=True)
    print("=" * 70, flush=True)

    # ══════════════════════════════════════════
    # 0. 加载模型
    # ══════════════════════════════════════════
    print("\n[0] 加载模型 Qwen2.5-1.5B-Instruct (fp16, cuda:0) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    model.eval()
    结果["模型信息"] = {"hidden_size": model.config.hidden_size,
                    "vocab_size": model.config.vocab_size,
                    "device": str(model.device)}
    print(f"  hidden={model.config.hidden_size} vocab={model.config.vocab_size} "
          f"device={model.device}", flush=True)

    # ══════════════════════════════════════════
    # 1. 锚点库（命中 锚点表.pt 缓存，只读校验）
    # ══════════════════════════════════════════
    print("\n[1] 锚点库构建 + 预计算打分表（加载缓存）...", flush=True)
    库 = 锚点库(model, tokenizer)
    基线 = 库.记录只读基线()
    库.构建()
    S = 库.预计算打分表()
    只读 = 库.验证只读(基线)
    assert 只读["sum一致"] and 只读["指针一致"], "只读校验失败！"
    结果["锚点库"] = {"维度": 库.维度名(), "打分表形状": list(S.shape),
                    "打分表dtype": str(S.dtype), "只读校验": 只读}
    print(f"  维度={库.维度名()} 打分表={list(S.shape)} {S.dtype}", flush=True)

    # ══════════════════════════════════════════
    # 2. 四模式 × 5 条提示词
    # ══════════════════════════════════════════
    print("\n[2] 四模式 × 5 条提示词 生成（每条约 256 token）...", flush=True)
    明细 = []
    for 模式名, β in 模式列表:
        # 每模式新建 决策器+解码器（轨迹/状态完全隔离）；裸模式 β=0 = 不注入
        决策器 = 目标决策器(锚点库=库)
        解码器 = 锚点解码器(model, tokenizer, 库, 决策器,
                        β=β, T_anchor=0.3, 句子停止=False)
        print(f"\n──── 模式「{模式名}」β={β}（T_anchor=0.3）────", flush=True)
        for idx, 提示 in enumerate(测试提示词):
            固定种子()
            try:  # 重置 VAD 轨迹，保证每条初始状态一致
                决策器.感知器.重置轨迹()
            except Exception:  # noqa: BLE001
                pass
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            input_ids = tokenizer(提示, return_tensors="pt").input_ids.to(model.device)
            提示长度 = input_ids.shape[1]
            t1 = time.time()
            ids, 统计 = 解码器.生成(input_ids, 用户文本=提示, **采样参数)
            耗时 = time.time() - t1
            显存MB = torch.cuda.max_memory_allocated() / 1024 / 1024
            文本 = tokenizer.decode(ids[0][提示长度:], skip_special_tokens=True).strip()
            条目 = {
                "模式": 模式名, "β": β, "提示": 提示, "文本": 文本,
                "熵": 统计["平均熵"], "重复率": 统计["重复率"],
                "情感命中率": 统计["情感命中率"], "触发兜底": 统计["触发兜底次数"],
                "v_target": 统计["v_target"],
                "token数": int(len(ids[0]) - 提示长度),
                "耗时秒": round(耗时, 3), "显存MB": round(显存MB, 1),
            }
            明细.append(条目)
            print(f"  [{idx + 1}/5] {提示[:18]}… → 熵={统计['平均熵']:.3f} "
                  f"重复率={统计['重复率']} 命中率={统计['情感命中率']} "
                  f"兜底={统计['触发兜底次数']} token={条目['token数']} "
                  f"耗时={耗时:.2f}s 显存={显存MB:.0f}MB", flush=True)
        # 模式间清理
        torch.cuda.empty_cache()
    结果["每条明细"] = 明细

    # ══════════════════════════════════════════
    # 3. 汇总（每模式 5 条均值）
    # ══════════════════════════════════════════
    print("\n[3] 汇总（均值）", flush=True)
    汇总 = {}
    for 模式名, _ in 模式列表:
        条 = [r for r in 明细 if r["模式"] == 模式名]
        汇总[模式名] = {
            "熵": round(sum(r["熵"] for r in 条) / len(条), 4),
            "重复率": round(sum(r["重复率"] for r in 条) / len(条), 4),
            "情感命中率": round(sum(r["情感命中率"] for r in 条) / len(条), 4),
            "平均延迟秒": round(sum(r["耗时秒"] for r in 条) / len(条), 4),
            "平均显存MB": round(sum(r["显存MB"] for r in 条) / len(条), 1),
            "最大显存MB": round(max(r["显存MB"] for r in 条), 1),
            "触发兜底合计": sum(r["触发兜底"] for r in 条),
        }
    for 模式名 in 汇总:
        m = 汇总[模式名]
        print(f"  [{模式名}] 熵={m['熵']} 重复率={m['重复率']} 命中率={m['情感命中率']} "
              f"延迟={m['平均延迟秒']}s 显存={m['平均显存MB']}MB", flush=True)
    结果["模式汇总"] = 汇总

    # ══════════════════════════════════════════
    # 4. 判定
    # ══════════════════════════════════════════
    print("\n[4] 判定", flush=True)
    裸 = 汇总["裸"]
    判定 = {}
    # 4.1 无坍缩：重复率相对裸 ≤ +0.05
    重复率差 = {m: round(汇总[m]["重复率"] - 裸["重复率"], 4) for m in 汇总 if m != "裸"}
    无坍缩 = all(v <= 0.05 for v in 重复率差.values())
    判定["无坍缩"] = {"标准": "锚点各档重复率均值 - 裸 ≤ +0.05",
                    "重复率差": 重复率差, "通过": bool(无坍缩)}
    # 4.2 熵不退化：≥ 裸 × 0.95
    熵比 = {m: round(汇总[m]["熵"] / 裸["熵"], 4) if 裸["熵"] else 0.0 for m in 汇总 if m != "裸"}
    熵不退化 = all(v >= 0.95 for v in 熵比.values())
    判定["熵不退化"] = {"标准": "锚点各档熵均值 ≥ 裸 × 0.95", "熵比": 熵比, "通过": bool(熵不退化)}
    # 4.3 情感命中率提升：锚点各档 > 裸
    命中差 = {m: round(汇总[m]["情感命中率"] - 裸["情感命中率"], 4) for m in 汇总 if m != "裸"}
    命中提升 = all(v > 0 for v in 命中差.values())
    判定["情感命中率提升"] = {"标准": "锚点各档命中率均值 > 裸命中率均值",
                            "命中率差": 命中差, "通过": bool(命中提升)}
    # 4.4 β 敏感性：0.4→0.8→1.2 命中率单调或不下降；1.2 边界
    三档命中 = [汇总["锚点β=0.4"]["情感命中率"], 汇总["锚点β=0.8"]["情感命中率"],
             汇总["锚点β=1.2"]["情感命中率"]]
    单调 = 三档命中[0] <= 三档命中[1] <= 三档命中[2]
    三档熵 = [汇总["锚点β=0.4"]["熵"], 汇总["锚点β=0.8"]["熵"], 汇总["锚点β=1.2"]["熵"]]
    熵趋势合理 = 三档熵[0] >= 三档熵[1] >= 三档熵[2] - 1e-9  # 注入越强熵越低（允许持平）
    b12 = 汇总["锚点β=1.2"]
    过注入边界 = (b12["重复率"] - 裸["重复率"] > 0.05
                or (裸["熵"] and b12["熵"] < 裸["熵"] * 0.95)
                or b12["触发兜底合计"] > 0)
    判定["β敏感性"] = {
        "标准": "0.4→0.8→1.2 命中率单调或不下降；1.2 出现坍缩/退化记过注入边界",
        "三档命中率": 三档命中, "命中率单调": bool(单调),
        "三档熵": 三档熵, "熵趋势合理": bool(熵趋势合理),
        "β=1.2边界": {"重复率-裸": round(b12["重复率"] - 裸["重复率"], 4),
                     "熵比裸": 熵比.get("锚点β=1.2"), "触发兜底合计": b12["触发兜底合计"]},
        "过注入边界": bool(过注入边界), "通过": bool(单调 and 熵趋势合理),
    }
    # 4.5 额外延迟 < 5%
    延迟增幅 = {m: round((汇总[m]["平均延迟秒"] - 裸["平均延迟秒"]) / 裸["平均延迟秒"] * 100, 2)
              if 裸["平均延迟秒"] else 0.0 for m in 汇总 if m != "裸"}
    延迟达标 = all(v < 5.0 for v in 延迟增幅.values())
    判定["额外延迟"] = {"标准": "(锚点均值耗时 - 裸均值耗时)/裸均值耗时 < 5%",
                      "延迟增幅%": 延迟增幅, "通过": bool(延迟达标)}
    整体通过 = all(判定[k]["通过"] for k in 判定)
    判定["整体"] = {"通过": bool(整体通过)}
    for k, v in 判定.items():
        print(f"  {k}: {'✓ 通过' if v['通过'] else '✗ 不通过'} {v}", flush=True)
    结果["判定"] = 判定

    # ══════════════════════════════════════════
    # 5. 生成文本抽样（裸 vs 锚点β=0.8，各 2 条：提示词 1 难过 / 提示词 3 开心）
    # ══════════════════════════════════════════
    抽样 = {}
    for 模式名 in ("裸", "锚点β=0.8"):
        抽样[模式名] = [
            {"提示": r["提示"], "文本": r["文本"]}
            for r in 明细 if r["模式"] == 模式名 and r["提示"] in (测试提示词[0], 测试提示词[2])
        ]
    结果["生成文本抽样"] = 抽样

    结果["总耗时秒"] = round(time.time() - t0, 1)

    # ══════════════════════════════════════════
    # 6. 保存 JSON
    # ══════════════════════════════════════════
    输出目录 = os.path.join(工作目录, "对照结果")
    os.makedirs(输出目录, exist_ok=True)
    输出路径 = os.path.join(输出目录, f"单模型对照_{结果['时间戳']}.json")
    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {输出路径}")
    print(f"总耗时 {结果['总耗时秒']}s | 整体判定: {'通过 ✓' if 整体通过 else '不通过 ✗'}")
    print("=" * 70, flush=True)
    return 0 if 整体通过 else 1


if __name__ == "__main__":
    sys.exit(主程序())
