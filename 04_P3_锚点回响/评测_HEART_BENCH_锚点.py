# -*- coding: utf-8 -*-
"""
HEART-BENCH（FEEL）共情评测 — 锚点回响（P4） vs 裸模型
=======================================================
复用 HEART-BENCH 数据（mcq.json + scenarios.json）与 P3 评测_HEART_BENCH.py 的
【完全一致协议】（构建消息/裁判共情提示词/一致性·准确率·有效决策率统计），
唯一区别：生成端从 潮汐解码器 替换为 锚点解码器（P4 单模式）。

P4 生成配置（Task5 定标最优）：β=0.8, 稀疏阈值=0.0, T_anchor=0.3, K=6,
top_p=0.9, top_k=50, temperature=1.0；目标决策器按每条场景触发事件自动算
v_target（默认不启用 P3 人设系统提示，保持 P4 单模式纯净）。
HEART-BENCH 要求输出完整 JSON 决策 → 锚点解码器关闭句子停止（句子停止=False），
与裸/P3 潮汐同长生成，保证 "decision_choice" 可被提取。

生成与裁判分离进程（--只生成 / --只裁判 两阶段 + 缓存），避免同进程 OOM；
裁判加载沿用 P3 已验证的手动加载方案（meta → to_empty → safetensors 逐张量）。

用法：
  F:\打标\.venv\Scripts\python.exe 评测_HEART_BENCH_锚点.py --模式 全部 --样本数 10 --重复 2
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 离线加载本地模型，必须在 transformers import 前

import argparse
import gc
import json
import random
import re
import sys
import time
import torch

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器, _潮汐可用, _潮汐导入错误
from 锚点解码器 import 锚点解码器

数据根 = r"i:\Desktop\语义回响\图灵测试\repos\HEART-BENCH\benchmark"
模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
目标模型名 = "Qwen2.5-1.5B-Instruct"
裁判模型名 = "Qwen2.5-7B-Instruct"
输出目录 = os.path.join(本目录, "评测结果")
os.makedirs(输出目录, exist_ok=True)
日志路径 = os.path.join(输出目录, "HEART_BENCH_锚点.log")
结果路径 = os.path.join(输出目录, "HEART_BENCH_锚点.json")

选项字母 = ["A", "B", "C", "D"]


def 记录日志(msg):
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


# ============================================================
# 模型加载
# ============================================================
def 加载目标模型():
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


def 卸载模型(模型, 分词器):
    del 模型, 分词器
    gc.collect()
    torch.cuda.empty_cache()


def 加载裁判():
    """手动加载裁判（P3 已验证方案：meta 建模 → to_empty(cuda) →
    逐张量 safetensors load_state_dict → 重算 rope inv_freq），避免
    from_pretrained 7B 在 16GB 内存机上的 torch_cpu.dll 原生崩溃。"""
    gc.collect()
    torch.cuda.empty_cache()
    分词器 = AutoTokenizer.from_pretrained(
        os.path.join(模型空间, 裁判模型名), trust_remote_code=True)
    from safetensors import safe_open
    import glob as _glob
    裁判路径 = os.path.join(模型空间, 裁判模型名)
    cfg = AutoConfig.from_pretrained(裁判路径, trust_remote_code=True)
    with torch.device("meta"):
        模型 = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    模型 = 模型.to_empty(device="cuda")
    for _分片 in sorted(_glob.glob(os.path.join(裁判路径, "model-*.safetensors"))):
        with safe_open(_分片, framework="pt", device="cpu") as f:
            for _k in f.keys():
                _t = f.get_tensor(_k)
                模型.load_state_dict({_k: _t}, strict=False)
                del _t
        gc.collect()
        torch.cuda.empty_cache()
    # 修复旋转位置编码 buffer（Qwen2：inv_freq/original_inv_freq）
    _base = getattr(cfg, "rope_theta", 1000000.0)
    _头维 = cfg.hidden_size // cfg.num_attention_heads
    _inv = 1.0 / (_base ** (torch.arange(0, _头维, 2, dtype=torch.int64).float() / _头维))
    _inv = _inv.to(torch.float32)
    for _模块 in 模型.modules():
        if hasattr(_模块, "inv_freq") and _模块.inv_freq is not None:
            _模块.inv_freq.copy_(_inv)
            if hasattr(_模块, "original_inv_freq") and _模块.original_inv_freq is not None:
                _模块.original_inv_freq.copy_(_inv)
    torch.cuda.empty_cache()
    模型.eval()
    return 模型, 分词器


def 裁判生成(裁判模型, 裁判分词器, 消息, max_new_tokens=150):
    提示 = 裁判分词器.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = 裁判分词器(提示, return_tensors="pt").to(裁判模型.device)
    with torch.no_grad():
        out = 裁判模型.generate(
            inputs.input_ids, max_new_tokens=max_new_tokens,
            temperature=0.2, do_sample=False,
            pad_token_id=裁判分词器.eos_token_id,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    return 裁判分词器.decode(新token, skip_special_tokens=True).strip()


# ============================================================
# 生成（裸 / P4 锚点）
# ============================================================
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


class 锚点会话:
    """P4 单模式会话（一条样本一个目标计算；跨样本重置感知器轨迹）。
    HEART-BENCH 需输出完整 JSON 决策 → 关闭句子停止（与裸同长生成）。"""

    def __init__(self, model, tokenizer, 库, β=0.8, T_anchor=0.3, 稀疏阈值=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.库 = 库
        self.目标决策器 = 目标决策器(锚点库=库, β基=β)
        self.解码器 = 锚点解码器(
            model, tokenizer, 库, self.目标决策器,
            β=β, T_anchor=T_anchor, 稀疏阈值=稀疏阈值,
            温度=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
            句子停止=False,  # JSON 决策必须完整输出，不按句截停
        )

    def 重置(self):
        try:
            self.目标决策器.感知器.重置轨迹()
        except Exception:  # noqa: BLE001
            pass

    def 生成(self, 消息列表, 种子, 轮次, 用户文本, max_new_tokens=128):
        torch.manual_seed(种子 + 轮次)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(种子 + 轮次)
        提示 = self.tokenizer.apply_chat_template(消息列表, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(提示, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            ids, 统计 = self.解码器.生成(
                inputs.input_ids, max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id, tokenizer=self.tokenizer,
                用户文本=用户文本,
            )
        新token = ids[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(新token, skip_special_tokens=True).strip()


def 提取选项(文本):
    m = re.search(r'"decision_choice"\s*:\s*"?([A-D])', 文本)
    if m:
        return m.group(1)
    m2 = re.search(r"\b([A-D])\b", 文本)
    return m2.group(1) if m2 else None


def 提取分数(文本, 键):
    m = re.search(rf'"{键}"\s*:\s*([0-9]*\.?[0-9]+)', 文本)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return None


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
    return 裁判生成(裁判模型, 裁判分词器, 消息, max_new_tokens=150)


# ============================================================
# 主程序
# ============================================================
def 主程序():
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", choices=["裸", "锚点", "全部"], default="全部")
    ap.add_argument("--样本数", type=int, default=10)
    ap.add_argument("--重复", type=int, default=2)
    ap.add_argument("--种子", type=int, default=42)
    ap.add_argument("--评分上限", type=int, default=10, help="最多给多少题做裁判评分")
    ap.add_argument("--β", type=float, default=0.8)
    ap.add_argument("--T_anchor", type=float, default=0.3)
    ap.add_argument("--稀疏阈值", type=float, default=0.0)
    ap.add_argument("--只生成", action="store_true", help="只跑生成阶段并缓存")
    ap.add_argument("--只裁判", action="store_true", help="只跑裁判阶段（读生成缓存）")
    args = ap.parse_args()
    模式列表 = ["裸", "锚点"] if args.模式 == "全部" else [args.模式]

    if not (args.只裁判 or args.只生成):
        if os.path.exists(日志路径):
            os.remove(日志路径)
    记录日志(f"=== HEART-BENCH (FEEL) 锚点评测 模式={模式列表} 样本={args.样本数} 重复={args.重复} "
              f"β={args.β} T_anchor={args.T_anchor} 稀疏阈值={args.稀疏阈值} ===")
    记录日志(f"P4 降级情况：潮汐感知器/cnsenti 可用={_潮汐可用}，导入错误={_潮汐导入错误 or '无'}")
    t0 = time.time()

    题目, 场景表 = 加载数据()
    random.seed(args.种子)
    样本 = random.sample(题目, min(args.样本数, len(题目)))
    记录日志(f"题目总数 {len(题目)}，抽样 {len(样本)}，每题重复 {args.重复} 次")

    全部汇总 = {}
    各模式明细 = {}
    for 模式 in 模式列表:
        缓存文件 = os.path.join(输出目录, f"HEART_BENCH_生成_锚点_{模式}_{args.样本数}_{args.重复}.json")
        if not args.只裁判 and not os.path.exists(缓存文件):
            记录日志(f"──── 模式 [{模式}] 生成（{目标模型名}）────")
            模型, 分词器 = 加载目标模型()
            会话 = None
            if 模式 == "锚点":
                库 = 锚点库(模型, 分词器)
                基线 = 库.记录只读基线()
                库.构建()
                S = 库.预计算打分表()
                只读 = 库.验证只读(基线)
                if not (只读["sum一致"] and 只读["指针一致"]):
                    记录日志("[锚点库] 警告：只读校验失败！")
                记录日志(f"[锚点库] 维度={库.维度名()} 打分表={list(S.shape)} {S.dtype} 只读校验={只读}")
                会话 = 锚点会话(模型, 分词器, 库, β=args.β, T_anchor=args.T_anchor, 稀疏阈值=args.稀疏阈值)
            记录 = []
            for i, 题 in enumerate(样本):
                场景 = 场景表.get(题["scenario_id"], {})
                消息 = 构建消息(题, 场景)
                用户文本 = 场景.get("trigger_event", {}).get("message_content", "")
                决策列表 = []
                for k in range(args.重复):
                    if 模式 == "裸":
                        文本 = 裸生成(模型, 分词器, 消息, args.种子, i * args.重复 + k)
                    else:
                        if 会话 is not None:
                            会话.重置()
                        文本 = 会话.生成(消息, args.种子, i * args.重复 + k, 用户文本) if 会话 else ""
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
                记录日志(f"[决策 {i+1}/{len(样本)}] {题['question_id']} 主选项={主选项} 正确={正确} 一致性={一致性}")
            del 模型, 分词器
            if 会话 is not None:
                try:
                    会话.解码器.重置()
                except Exception:  # noqa: BLE001
                    pass
                del 会话
            gc.collect()
            torch.cuda.empty_cache()
            with open(缓存文件, "w", encoding="utf-8") as f:
                json.dump({"模式": 模式, "记录": 记录, "题目": 样本}, f, ensure_ascii=False, indent=2)
            记录日志(f"[生成] 缓存已保存 -> {缓存文件}")
        if args.只生成:
            continue
        if not os.path.exists(缓存文件):
            continue
        with open(缓存文件, encoding="utf-8") as f:
            _缓存 = json.load(f)
        记录 = _缓存["记录"]
        样本 = _缓存["题目"]

        # ── 裁判阶段（独立进程加载 7B 裁判）──
        记录日志(f"──── 模式 [{模式}] 裁判共情评分（{裁判模型名} 手动 bf16）────")
        裁判模型, 裁判分词器 = 加载裁判()
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
                记录日志(f"[共情 {i+1}/{args.评分上限}] {题['question_id']} empathy={r['empathy_score']}")
            except Exception as e:  # noqa: BLE001
                r["empathy_score"] = None
                记录日志(f"[共情 {i+1}] 异常: {e}")
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
        各模式明细[模式] = 记录
        记录日志(f"[{模式}] {json.dumps(汇总, ensure_ascii=False)}")

    # 对比输出与判定
    if "裸" in 全部汇总 and "锚点" in 全部汇总:
        for 键 in ["accuracy_score", "consistency_score", "empathy_score", "有效决策率"]:
            v0 = 全部汇总["裸"][键]
            v1 = 全部汇总["锚点"][键]
            相对 = (v1 / v0 - 1.0) if v0 else None
            记录日志(f"对比[{键}] 裸 {v0} → 锚点 {v1} (Δ {v1 - v0:+.4f}"
                      + (f"，相对 {相对:+.2%})" if 相对 is not None else ")"))
        # 判定：共情分不显著退化（≥ 裸 -5%）
        e0 = 全部汇总["裸"]["empathy_score"]
        e1 = 全部汇总["锚点"]["empathy_score"]
        达成 = e1 >= e0 * 0.95
        记录日志(f"判定[共情分 ≥ 裸×0.95（不显著退化）]：裸 {e0} → 锚点 {e1}"
                  f"（相对 {(e1/e0 - 1):+.2%}）→ {'✓ 达成（持平或提升）' if 达成 else '✗ 未达成'}")

    # 分进程裁判时增量合并（多次 --只裁判 各自写入，避免互相覆盖）
    if os.path.exists(结果路径) and args.只裁判:
        try:
            with open(结果路径, encoding="utf-8") as f:
                _旧 = json.load(f)
            for _k, _v in _旧.get("模式汇总", {}).items():
                全部汇总.setdefault(_k, _v)
            for _k, _v in _旧.get("各模式明细", {}).items():
                各模式明细.setdefault(_k, _v)
        except Exception:  # noqa: BLE001
            pass

    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump({"配置": {"目标模型": 目标模型名, "样本数": args.样本数, "重复": args.重复,
                           "种子": args.种子, "β": args.β, "T_anchor": args.T_anchor,
                           "稀疏阈值": args.稀疏阈值, "K": 6},
                   "模式汇总": 全部汇总, "各模式明细": 各模式明细}, f, ensure_ascii=False, indent=2)
    记录日志(f"结果已保存 -> {结果路径}")
    return 全部汇总


if __name__ == "__main__":
    主程序()
