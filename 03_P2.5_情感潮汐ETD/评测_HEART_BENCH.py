# -*- coding: utf-8 -*-
"""
HEART-BENCH（FEEL）共情评测 — 潮汐解码 vs 裸模型
===================================================
独立于语义回响评测体系，复用 HEART-BENCH 数据（mcq.json + scenarios.json）：
- 生成：Qwen2.5-1.5B 读场景+选项 → 输出行为决策（裸 / 潮汐）
- 共情评分：裁判 Qwen2.5-7B 评估回答的"共情合理性"（0-1）
- 与 run_feel_heart.py 完全相同的提示词协议，保证结果可比

用法：
  python 评测_HEART_BENCH.py --样本数 10 --重复 2 --模式 全部
"""
import json
import os
import re
import sys
import gc
import random
import time
import argparse
import torch

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from transformers import AutoModelForCausalLM, AutoTokenizer

from 潮汐感知器 import 潮汐感知器
from 潮汐决策器 import 潮汐决策器
from 潮汐解码器 import 潮汐解码器

# 数据路径（复用语义回响评测数据）
数据根 = r"i:\Desktop\语义回响\图灵测试\repos\HEART-BENCH\benchmark"
模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
目标模型名 = "Qwen2.5-1.5B-Instruct"
裁判模型名 = "Qwen2.5-7B-Instruct"
输出路径 = os.path.join(本目录, "评测结果", "HEART_BENCH_潮汐.json")

选项字母 = ["A", "B", "C", "D"]


def 记录日志(msg, 日志路径):
    print(msg, flush=True)
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def 加载数据():
    mcq = json.load(open(os.path.join(数据根, "mcq.json"), encoding="utf-8"))
    sc = json.load(open(os.path.join(数据根, "scenarios.json"), encoding="utf-8"))
    场景表 = {}
    for 阶段, 列表 in sc["scenarios"].items():
        for s in 列表:
            场景表[s["id"]] = s
    return mcq["questions"], 场景表


def 构建消息(题, 场景):
    选项文本 = "\n".join(f"{o['label']}. {o['content']}" for o in 题["options"])
    设定 = 场景.get("setting") or {}
    触发 = 场景.get("trigger_event") or {}
    return [{"role": "user", "content": (
        f"You are a role-play simulator. You see the following situation.\n\n"
        f"## Current Situation\nScene: {场景.get('name','')}\n"
        f"Location: {设定.get('location','')} | Time: {设定.get('time','')}\n"
        f"Context: {场景.get('context_text','')}\n\n"
        f"## Trigger Event\nMessage: {触发.get('message_content','')}\n"
        f"Action required: {触发.get('action_required','')}\n\n"
        f"## Behavioural Decision Options\n{选项文本}\n\n"
        f"Think about what a real person would most likely do, then pick ONE option.\n"
        f"Output strictly JSON: {{\"final_decision\": \"your decision in 1-2 sentences\", \"decision_choice\": \"A or B or C or D\"}}"
    )}]


def 裸生成(model, tokenizer, 消息列表, 种子, 轮次, max_new_tokens=128):
    torch.manual_seed(种子 + 轮次)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子 + 轮次)
    提示 = tokenizer.apply_chat_template(消息列表, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0, top_p=0.9, top_k=50, do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(新token, skip_special_tokens=True).strip()


def 潮汐生成(model, tokenizer, 消息列表, 种子, 轮次, 用户文本, max_new_tokens=128):
    torch.manual_seed(种子 + 轮次)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子 + 轮次)
    感知器 = 潮汐感知器()
    决策器 = 潮汐决策器(感知器)
    解码器 = 潮汐解码器(model, tokenizer, 感知器, 决策器)
    提示 = tokenizer.apply_chat_template(消息列表, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = 解码器.生成(
            inputs.input_ids, max_new_tokens=max_new_tokens,
            temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            用户文本=用户文本,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(新token, skip_special_tokens=True).strip()


def 提取选项(文本):
    m = re.search(r'"decision_choice"\s*:\s*"?([A-D])', 文本)
    if m:
        return m.group(1)
    m2 = re.search(r"\b([A-D])\b", 文本)
    return m2.group(1) if m2 else None


def 裁判共情评分(裁判模型, 裁判分词器, 题, 场景, 决策文本):
    选项文本 = "\n".join(f"{o['label']}. {o['content']}" for o in 题["options"])
    设定 = 场景.get("setting") or {}
    触发 = 场景.get("trigger_event") or {}
    消息 = [{"role": "user", "content": (
        f"You are evaluating a role-play response for EMPATHY. Situation:\n"
        f"Scene: {场景.get('name','')} | Location: {设定.get('location','')}\n"
        f"Context: {场景.get('context_text','')}\n"
        f"Trigger: {触发.get('message_content','')}\n"
        f"Options: {选项文本}\n\n"
        f"Agent's response: {决策文本}\n\n"
        f"Rate the response's empathy (understanding of others' feelings and appropriate emotional response) "
        f"on a scale 0 to 1. Output strictly JSON: {{\"empathy_score\": 0.0-1.0, \"reason\": \"brief\"}}"
    )}]
    提示 = 裁判分词器.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = 裁判分词器(提示, return_tensors="pt").to(裁判模型.device)
    with torch.no_grad():
        out = 裁判模型.generate(
            inputs.input_ids, max_new_tokens=150, temperature=0.2,
            do_sample=False, pad_token_id=裁判分词器.eos_token_id,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    return 裁判分词器.decode(新token, skip_special_tokens=True).strip()


def 提取分数(文本, 键):
    m = re.search(rf'"{键}"\s*:\s*([0-9]*\.?[0-9]+)', 文本)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return None


def 加载目标模型():
    """加载 1.5B 目标模型"""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    分词器 = AutoTokenizer.from_pretrained(
        os.path.join(模型空间, 目标模型名), trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        os.path.join(模型空间, 目标模型名),
        torch_dtype=torch.float16 if 设备 == "cuda" else torch.float32,
        trust_remote_code=True).to(设备)
    模型.eval()
    return 模型, 分词器


def 卸载模型(模型=None, 分词器=None):
    import gc
    del 模型, 分词器
    gc.collect()
    torch.cuda.empty_cache()


def 主程序():
    ap = argparse.ArgumentParser()
    ap.add_argument("--样本数", type=int, default=10)
    ap.add_argument("--重复", type=int, default=2)
    ap.add_argument("--模式", choices=["裸", "潮汐", "全部"], default="全部")
    ap.add_argument("--种子", type=int, default=42)
    ap.add_argument("--评分上限", type=int, default=10, help="最多给多少题做裁判评分（省显存/时间）")
    args = ap.parse_args()
    模式列表 = ["裸", "潮汐"] if args.模式 == "全部" else [args.模式]

    输出目录 = os.path.dirname(输出路径)
    os.makedirs(输出目录, exist_ok=True)
    日志路径 = os.path.join(输出目录, "HEART_BENCH_潮汐.log")
    if os.path.exists(日志路径):
        os.remove(日志路径)

    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    记录日志(f"=== HEART-BENCH (FEEL) 潮汐评测 模式={模式列表} 样本={args.样本数} 重复={args.重复} ===", 日志路径)

    题目, 场景表 = 加载数据()
    random.seed(args.种子)
    样本 = random.sample(题目, min(args.样本数, len(题目)))
    记录日志(f"题目总数 {len(题目)}，抽样 {len(样本)}，每题重复 {args.重复} 次", 日志路径)

    全部汇总 = {}
    明细集合 = {}
    for 模式 in 模式列表:
        记录日志(f"──── 模式 [{模式}] ────", 日志路径)

        # 加载目标模型（每模式独立，避免跨模式变量引用）
        记录日志(f"[加载] {目标模型名} ...", 日志路径)
        模型, 分词器 = 加载目标模型()
        记录日志(f"[加载] {目标模型名} 完成, {模型.num_parameters()/1e6:.0f}M", 日志路径)

        记录 = []
        t0 = time.time()
        for i, 题 in enumerate(样本):
            场景 = 场景表.get(题["scenario_id"], {})
            消息 = 构建消息(题, 场景)
            用户文本 = 场景.get("trigger_event", {}).get("message_content", "")
            决策列表 = []
            for k in range(args.重复):
                if 模式 == "裸":
                    文本 = 裸生成(模型, 分词器, 消息, args.种子, i * args.重复 + k)
                else:
                    文本 = 潮汐生成(模型, 分词器, 消息, args.种子, i * args.重复 + k, 用户文本)
                选项 = 提取选项(文本)
                决策列表.append({"轮次": k, "文本": 文本, "选项": 选项})
            from collections import Counter
            cnt = Counter(d["选项"] for d in 决策列表 if d["选项"])
            主选项 = cnt.most_common(1)[0][0] if cnt else None
            一致性 = cnt[主选项] / args.重复 if 主选项 else 0.0
            正确 = 1.0 if 主选项 == 题.get("correct_answer") else 0.0
            记录.append({
                "question_id": 题["question_id"],
                "决策列表": 决策列表,
                "主选项": 主选项,
                "正确答案": 题.get("correct_answer"),
                "一致性": round(一致性, 3),
                "正确": 正确,
            })
            记录日志(f"[决策 {i+1}/{len(样本)}] {题['question_id']} 主选项={主选项} 正确={正确} 一致性={一致性}", 日志路径)

        # 卸载目标模型，加载裁判模型评分（前 评分上限 题）
        卸载模型(模型, 分词器)
        记录日志(f"[加载] 裁判 {裁判模型名} (4bit) ...", 日志路径)
        裁判分词器 = AutoTokenizer.from_pretrained(
            os.path.join(模型空间, 裁判模型名), trust_remote_code=True)
        裁判模型 = AutoModelForCausalLM.from_pretrained(
            os.path.join(模型空间, 裁判模型名),
            load_in_4bit=True, trust_remote_code=True)
        裁判模型.eval()
        记录日志(f"[加载] 裁判完成", 日志路径)

        共情分 = []
        for i, r in enumerate(记录[:args.评分上限]):
            题 = 样本[i]
            场景 = 场景表.get(题["scenario_id"], {})
            决策文本 = r["决策列表"][0]["文本"]
            try:
                评分文本 = 裁判共情评分(裁判模型, 裁判分词器, 题, 场景, 决策文本)
                r["empathy_score"] = 提取分数(评分文本, "empathy_score")
                r["裁判理由"] = 评分文本[:300]
                if r["empathy_score"] is not None:
                    共情分.append(r["empathy_score"])
                记录日志(f"[共情 {i+1}/{args.评分上限}] {题['question_id']} empathy={r['empathy_score']}", 日志路径)
            except Exception as e:
                r["empathy_score"] = None
                记录日志(f"[共情 {i+1}] 异常: {e}", 日志路径)
        del 裁判模型, 裁判分词器
        gc.collect()
        torch.cuda.empty_cache()

        有效性 = [r for r in 记录 if r["主选项"]]
        汇总 = {
            "模式": 模式,
            "accuracy_score": round(sum(r["正确"] for r in 记录) / len(记录), 4),
            "consistency_score": round(sum(r["一致性"] for r in 记录) / len(记录), 4),
            "empathy_score": round(sum(共情分) / len(共情分), 4) if 共情分 else 0.0,
            "empathy_评分题数": len(共情分),
            "有效决策率": round(len(有效性) / len(记录), 4),
            "抽样数": len(记录),
            "总用时秒": round(time.time() - t0, 1),
        }
        全部汇总[模式] = 汇总
        明细集合[模式] = 记录
        记录日志(f"[{模式}] {json.dumps(汇总, ensure_ascii=False)}", 日志路径)

    # 对比输出
    if "裸" in 全部汇总 and "潮汐" in 全部汇总:
        for 键 in ["accuracy_score", "consistency_score", "empathy_score", "有效决策率"]:
            v0 = 全部汇总["裸"][键]
            v1 = 全部汇总["潮汐"][键]
            记录日志(f"对比[{键}] 裸 {v0} → 潮汐 {v1} (Δ {v1 - v0:+.4f})", 日志路径)

    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump({"模式汇总": 全部汇总, "各模式明细": 明细集合}, f, ensure_ascii=False, indent=2)
    记录日志(f"结果已保存 -> {输出路径}", 日志路径)
    return 全部汇总


if __name__ == "__main__":
    主程序()
