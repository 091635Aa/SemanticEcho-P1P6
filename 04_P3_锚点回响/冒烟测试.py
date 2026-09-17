# -*- coding: utf-8 -*-
"""P4 锚点回响（Anchor Echo）· Task3 冒烟测试

按任务要求加载 Qwen2.5-1.5B（fp16, cuda:0）跑通：
  1. 锚点库构建 + 预计算打分表 + 只读校验（前后 W_e.sum() 一致）
  2. 目标决策器：悲伤文本 → v_target 主分量落「难过」附近；"温柔陪伴" → 「温柔」
  3. 锚点解码器 裸 vs 锚点 各生成 1 段（种子 42，5 条提示词任选），输出熵/重复率/命中率
  4. 混合锚点器 三通道全开 生成 1 段
  5. 记录结果到 对照结果\\冒烟_<时间戳>.json

降级情况（cnsenti/潮汐感知器 import 失败）自动记录在结果中，不影响主流程。
全部通过（无报错、无 OOM、只读校验通过、v_target 方向正确）后退出码 0。

用法：F:\\打标\\.venv\\Scripts\\python.exe 冒烟测试.py
"""
import os
import sys
import json
import time
from datetime import datetime

os.environ["HF_HUB_OFFLINE"] = "1"
工作目录 = r"h:\锚点回响（Anchor Echo）"
if 工作目录 not in sys.path:
    sys.path.insert(0, 工作目录)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器, 自动适配, _潮汐可用, _潮汐导入错误
from 锚点解码器 import 锚点解码器
from 混合锚点器 import 混合锚点器, _回响可用
from 接口降级 import 判定接口, 构造提示词

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
提示词集 = [
    "我最近工作压力好大，感觉快撑不住了",
    "今天被领导当众批评了，心里特别委屈",
    "我终于通过了考试，太开心了！",
    "最近总是失眠，一个人在家很害怕",
    "我跟他吵架了，气得浑身发抖",
]
悲伤文本 = "我今天特别难过，感觉整个世界都塌了，心里好委屈"
MAX_NEW = 48


def 固定种子():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


def main():
    结果 = {
        "时间戳": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "任务": "P4 锚点回响（Anchor Echo）Task3 冒烟测试",
        "模型路径": 模型路径,
        "降级情况": {"潮汐感知器/cnsenti 可用": _潮汐可用,
                    "导入错误": _潮汐导入错误 or "无",
                    "回响注入器可用": _回响可用},
    }
    t0 = time.time()

    # ══════════════════════════════════════════
    # 0. 加载模型
    # ══════════════════════════════════════════
    print("=== 0. 加载模型 Qwen2.5-1.5B (fp16, cuda:0) ===", flush=True)
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
    # 1. 锚点库构建 + 预计算打分表 + 只读校验
    # ══════════════════════════════════════════
    print("\n=== 1. 锚点库构建 + 预计算打分表 + 只读校验 ===", flush=True)
    库 = 锚点库(model, tokenizer)
    基线 = 库.记录只读基线()
    库.构建()
    S = 库.预计算打分表()
    只读 = 库.验证只读(基线)
    S重算 = 库.预计算打分表(强制重算=True)  # 重算以校验缓存一致性
    一致性 = float(torch.max(torch.abs(S.float() - S重算.float())).item())
    assert 只读["sum一致"] and 只读["指针一致"], "只读校验失败！"
    assert S.shape == (model.config.vocab_size, 6), f"打分表形状异常 {S.shape}"
    assert S.dtype == torch.float16, f"打分表 dtype 异常 {S.dtype}"
    assert bool(torch.isfinite(S).all()), "打分表含 NaN/Inf"
    assert 一致性 < 1e-4, f"打分表缓存与重算不一致 max|Δ|={一致性}"
    每维词数 = {维: len(库.词集[维]) for 维 in 库.维度名()}
    assert all(v >= 45 for v in 每维词数.values()), f"每维词数不足 50 左右: {每维词数}"
    结果["锚点库"] = {
        "维度": 库.维度名(), "每维词数": 每维词数,
        "打分表形状": list(S.shape), "打分表dtype": str(S.dtype),
        "打分表缓存一致maxΔ": round(一致性, 6),
        "只读校验": 只读,
    }
    print(f"  维度={库.维度名()}")
    print(f"  每维词数={每维词数}")
    print(f"  打分表 S={list(S.shape)} {S.dtype}，缓存一致性 maxΔ={一致性:.6f}")
    print(f"  只读校验 sum一致={只读['sum一致']} 指针一致={只读['指针一致']}", flush=True)

    # max_pooling 备选聚合验证
    库max = 锚点库(model, tokenizer, 打分模式="max池化")
    库max.构建()
    S_max = 库max.预计算打分表()
    上头分 = 库max.得分("上头", 打分模式="max池化")
    assert S_max.shape == (model.config.vocab_size, 6)
    结果["锚点库"]["max池化"] = {
        "S_max形状": list(S_max.shape),
        "网络流行语「上头」max池化得分": [round(float(x), 4) for x in 上头分],
    }
    print(f"  max池化: 「上头」得分={[round(float(x),4) for x in 上头分]}", flush=True)

    # ══════════════════════════════════════════
    # 2. 目标决策器
    # ══════════════════════════════════════════
    print("\n=== 2. 目标决策器（悲伤文本 / 温柔指令 / 默认陪伴） ===", flush=True)
    决策器_悲伤 = 目标决策器(锚点库=库)
    目标1 = 决策器_悲伤.计算目标(用户当前=悲伤文本)
    排序1 = np.argsort(目标1.v_target)[::-1]
    主维1 = 库.维度名()[int(排序1[0])]
    次维1 = 库.维度名()[int(排序1[1])]
    结果["目标决策器"] = {
        "悲伤文本": 悲伤文本,
        "v_target": [round(float(x), 4) for x in 目标1.v_target],
        "主分量": 主维1, "次分量": 次维1, "β": 目标1.β,
        "密度目标": 目标1.情感词密度目标, "说明": 目标1.说明,
        "决策日志": {k: v for k, v in 目标1.决策日志.items() if k != "来源权重"},
    }
    print(f"  悲伤文本 → v_target={[round(float(x),3) for x in 目标1.v_target]}")
    print(f"    主分量={主维1}（次={次维1}） β={目标1.β}")
    assert 主维1 in ("难过", "害怕"), f"悲伤文本主分量异常: {主维1}"
    assert "难过" in (主维1, 次维1), f"悲伤文本「难过」未进入前二: {主维1}/{次维1}"

    决策器_指令 = 目标决策器(锚点库=库)
    目标2 = 决策器_指令.计算目标(指令="温柔陪伴")
    主维2 = 库.维度名()[int(np.argmax(目标2.v_target))]
    结果["目标决策器"]["温柔指令"] = {
        "v_target": [round(float(x), 4) for x in 目标2.v_target],
        "主分量": 主维2, "β": 目标2.β, "说明": 目标2.说明,
    }
    print(f"  「温柔陪伴」→ 主分量={主维2} v_target={[round(float(x),3) for x in 目标2.v_target]}")
    assert 主维2 == "温柔", f"温柔指令主分量异常: {主维2}"

    决策器_默认 = 目标决策器(锚点库=库)
    目标3 = 决策器_默认.计算目标()
    主维3 = 库.维度名()[int(np.argmax(目标3.v_target))]
    结果["目标决策器"]["默认无输入"] = {"主分量": 主维3, "说明": 目标3.说明}
    print(f"  无输入 → 主分量={主维3}（默认陪伴基调）")
    assert 主维3 == "温柔", f"默认基调异常: {主维3}"

    适配 = 自动适配(model, "fp16")
    结果["目标决策器"]["自动适配"] = 适配
    print(f"  自动适配(Qwen2.5-1.5B, fp16) → β={适配['β']} T_anchor={适配['T_anchor']}（{适配['来源']}）")
    assert abs(适配["β"] - 0.8) < 1e-6, f"自动适配 β 应命中扫描表 0.8，实得 {适配['β']}"

    # ══════════════════════════════════════════
    # 3. 锚点解码器 裸 vs 锚点
    # ══════════════════════════════════════════
    print("\n=== 3. 锚点解码器 裸(β=0) vs 锚点(β=0.8) ===", flush=True)
    提示 = 提示词集[0]
    input_ids = tokenizer(提示, return_tensors="pt").input_ids.to(model.device)
    解码结果 = {}
    for 名称, β in (("裸", 0.0), ("锚点", 0.8)):
        固定种子()
        解码器 = 锚点解码器(model, tokenizer, 库, 目标决策器(锚点库=库),
                         β=β, T_anchor=0.3, 句子停止=False)
        t1 = time.time()
        ids, 统计 = 解码器.生成(input_ids, max_new_tokens=MAX_NEW, 用户文本=提示)
        耗时 = round(time.time() - t1, 2)
        文本 = tokenizer.decode(ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        解码结果[名称] = {**统计, "文本": 文本, "耗时": 耗时}
        print(f"  [{名称}] β={统计['β']} 熵={统计['平均熵']} 重复率={统计['重复率']} "
              f"命中率={统计['情感命中率']} 兜底={统计['触发兜底次数']} 耗时={耗时}s")
        print(f"      输出：{文本}", flush=True)
    结果["锚点解码器对照"] = 解码结果
    # 无坍缩硬性指标：重复率 ≤ 0.6、熵 > 0
    for 名称, 条 in 解码结果.items():
        assert 条["重复率"] <= 0.6 + 1e-6, f"[{名称}] 重复率异常 {条['重复率']}"
        assert 条["平均熵"] > 0.0, f"[{名称}] 平均熵为 0"

    # 3.5 接口降级验证（不生成，仅验证判定 + logprobs 候选受限注入）
    print("\n=== 3.5 三级接口降级验证 ===", flush=True)
    接口 = 判定接口(model)
    提示模板 = 构造提示词("温柔", 库, "我今天好难过")
    解码器A = 锚点解码器(model, tokenizer, 库, 决策器_悲伤, β=0.8, 句子停止=False)
    解码器A.v_target = np.asarray(目标1.v_target, dtype=np.float32)  # 难过方向
    解码器A.接口 = "logprobs"
    dummy = torch.zeros(1, model.config.vocab_size, device=model.device, dtype=torch.float16)
    注入后 = 解码器A.注入偏置(dummy.clone())
    改变数 = int((注入后 != dummy).sum().item())
    assert 接口 == "本地", f"本地模型判定异常: {接口}"
    assert 0 < 改变数 <= 100, f"logprobs 候选受限注入改变数异常: {改变数}"
    assert bool(torch.isfinite(注入后).all())
    结果["接口降级"] = {"判定接口": 接口, "提示模板": 提示模板,
                     "logprobs候选改变数": 改变数, "topk候选": 解码器A.topk候选}
    print(f"  判定接口={接口}")
    print(f"  logprobs 候选受限注入：{改变数} 个 top-k 候选被偏置（≤100）")
    print(f"  提示模板：{提示模板}", flush=True)

    # ══════════════════════════════════════════
    # 4. 混合锚点器 三通道全开
    # ══════════════════════════════════════════
    print("\n=== 4. 混合锚点器（A锚点+B回响+C潮汐 全开） ===", flush=True)
    固定种子()
    混合 = 混合锚点器(model, tokenizer, 库, 目标决策器(锚点库=库),
                   锚点β=0.8, 锚点T=0.3, 回响λ=0.08, 潮汐倍率=12.0,
                   开启A=True, 开启B=True, 开启C=True, 句子停止=False)
    t2 = time.time()
    ids4, 统计4 = 混合.生成(input_ids, max_new_tokens=MAX_NEW, 用户文本=提示)
    耗时4 = round(time.time() - t2, 2)
    文本4 = tokenizer.decode(ids4[0][input_ids.shape[1]:], skip_special_tokens=True)
    混合结果 = {**统计4, "文本": 文本4, "耗时": 耗时4,
               "通道": {"A": 混合.开启A, "B": 混合.开启B, "C": 混合.开启C},
               "极性token数": len(混合._极性token表),
               "回响池大小": 混合.回响.pool.大小 if 混合.回响 else 0,
               "回响错误": 混合._回响错误 or "无"}
    结果["混合锚点器"] = 混合结果
    print(f"  {混合}")
    print(f"  熵={统计4['平均熵']} 重复率={统计4['重复率']} 命中率={统计4['情感命中率']} "
          f"β={统计4['β']} 兜底={统计4['触发兜底次数']} 耗时={耗时4}s")
    print(f"  输出：{文本4}", flush=True)
    assert 混合.开启A and 混合.开启B and 混合.开启C, "三通道未全开"
    assert 混合结果["重复率"] <= 0.6 + 1e-6, f"混合重复率异常 {混合结果['重复率']}"

    # ══════════════════════════════════════════
    # 5. 记录结果
    # ══════════════════════════════════════════
    结果["总耗时秒"] = round(time.time() - t0, 1)
    结果["显存"] = {"cuda_allocated_GB": round(torch.cuda.memory_allocated() / 1e9, 2),
                  "cuda_reserved_GB": round(torch.cuda.memory_reserved() / 1e9, 2)}
    输出目录 = os.path.join(工作目录, "对照结果")
    os.makedirs(输出目录, exist_ok=True)
    输出路径 = os.path.join(输出目录, f"冒烟_{结果['时间戳']}.json")
    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("冒烟测试全部通过 ✓")
    print(f"结果已记录：{输出路径}")
    print(f"总耗时 {结果['总耗时秒']}s | 显存 {结果['显存']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
