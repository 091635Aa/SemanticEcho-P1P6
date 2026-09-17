# -*- coding: utf-8 -*-
"""
④ TuringBench — 大规模图灵测试基准（P4 锚点回响版 · 中文体系）
================================================================
基于 i:\Desktop\语义回响\图灵测试\run_turingbench.py 的协议改造（生成端替换为
锚点解码器，P3 原脚本零改动；不依赖 生成器.py/公共模块.py，独立加载目标模型）：

  1. 人类语料：chinese-adorable 高情商对话数据集 girl 回复（真人中文）
  2. AI 语料：目标模型 1.5B 对相同 user 中文输入生成回复（裸 / P4 锚点，同 chat 模板）
  3. 检测器：TF-IDF(1-2gram) + 逻辑回归，在 人类 vs AI 上训练（留出测试集）
  4. detection_accuracy = 1.5B 新生成文本被判为 AI 的比例（越低越像人）
     human_likeness = 1 - detection_accuracy

P4 生成配置（Task5 定标最优）：β=0.8, 稀疏阈值=0.0, T_anchor=0.3, K=6，
目标决策器按每条 user 输入自动算 v_target（默认无 P3 人设/身份拦截，保持
P4 单模式纯净；裸与锚点同一 chat 模板，唯一变量是解码器）。

生成与检测分离进程（--只生成 / --只检测 两阶段 + 缓存）。

用法：
  F:\打标\.venv\Scripts\python.exe 评测_TuringBench_锚点.py --模式 全部
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 离线加载本地模型，必须在 transformers import 前

import argparse
import gc
import json
import random
import sys
import time
import torch

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from transformers import AutoModelForCausalLM, AutoTokenizer

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器, _潮汐可用, _潮汐导入错误
from 锚点解码器 import 锚点解码器

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
目标模型名 = "Qwen2.5-1.5B-Instruct"
数据集路径 = r"c:\Users\Administrator\.cache\huggingface\hub\datasets--sunorme--chinese-adorable-high-emotional-intelligence-chat\snapshots\15f8a4895c7529c16cd8b43bccc95abf4f8b7c6b\chinese-adorable-high-emotional-intelligence-chat.json"
输出目录 = os.path.join(本目录, "评测结果")
os.makedirs(输出目录, exist_ok=True)
日志路径 = os.path.join(输出目录, "TuringBench_锚点.log")
结果路径 = os.path.join(输出目录, "TuringBench_锚点.json")

训练对数 = 60
测试对数 = 30


def 记录日志(msg):
    print(msg, flush=True)
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def 加载对话():
    with open(数据集路径, encoding="utf-8") as f:
        data = json.load(f)
    有效 = [
        d for d in data
        if isinstance(d, dict) and d.get("user") and d.get("girl")
        and 2 <= len(d["user"]) <= 80 and 2 <= len(d["girl"]) <= 200
    ]
    return 有效


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


def 裸生成(model, tokenizer, 消息, 种子, 轮次, max_new_tokens=64):
    torch.manual_seed(种子 + 轮次)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子 + 轮次)
    提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids, max_new_tokens=max_new_tokens,
            temperature=1.0, top_p=0.9, top_k=50, do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(新token, skip_special_tokens=True).strip()


class 锚点会话:
    """P4 单模式会话（一条样本一个目标计算；跨样本重置感知器轨迹）"""

    def __init__(self, model, tokenizer, 库, β=0.8, T_anchor=0.3, 稀疏阈值=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.库 = 库
        self.目标决策器 = 目标决策器(锚点库=库, β基=β)
        self.解码器 = 锚点解码器(
            model, tokenizer, 库, self.目标决策器,
            β=β, T_anchor=T_anchor, 稀疏阈值=稀疏阈值,
            温度=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
        )

    def 重置(self):
        try:
            self.目标决策器.感知器.重置轨迹()
        except Exception:  # noqa: BLE001
            pass

    def 生成(self, 消息, 种子, 轮次, 用户文本, max_new_tokens=64):
        torch.manual_seed(种子 + 轮次)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(种子 + 轮次)
        提示 = self.tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(提示, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            ids, 统计 = self.解码器.生成(
                inputs.input_ids, max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id, tokenizer=self.tokenizer,
                用户文本=用户文本,
            )
        新token = ids[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(新token, skip_special_tokens=True).strip()


def 生成AI文本(模式, args, 对话):
    """生成阶段：只加载目标模型（1.5B），输出 训练+测试 的 AI 文本列表"""
    记录日志(f"──── 模式 [{模式}] AI 生成（max_new_tokens=64）────")
    model, tokenizer = 加载目标模型()
    会话 = None
    if 模式 == "锚点":
        库 = 锚点库(model, tokenizer)
        基线 = 库.记录只读基线()
        库.构建()
        S = 库.预计算打分表()
        只读 = 库.验证只读(基线)
        if not (只读["sum一致"] and 只读["指针一致"]):
            记录日志("[锚点库] 警告：只读校验失败！")
        记录日志(f"[锚点库] 维度={库.维度名()} 打分表={list(S.shape)} {S.dtype} 只读校验={只读}")
        会话 = 锚点会话(model, tokenizer, 库, β=args.β, T_anchor=args.T_anchor, 稀疏阈值=args.稀疏阈值)
    seed_offset = args.seed_base
    AI训练文本, AI测试文本 = [], []
    for i, d in enumerate(对话[:训练对数]):
        消息 = [{"role": "user", "content": d["user"]}]
        if 模式 == "裸":
            文本 = 裸生成(model, tokenizer, 消息, seed_offset, i)
        else:
            if 会话 is not None:
                会话.重置()
            文本 = 会话.生成(消息, seed_offset, i, d["user"]) if 会话 else ""
        AI训练文本.append(文本)
        记录日志(f"[AI训练 {i+1}/{训练对数}] {d['user'][:14]} => {文本[:28]}")
    for i, d in enumerate(对话[训练对数:训练对数 + 测试对数]):
        消息 = [{"role": "user", "content": d["user"]}]
        if 模式 == "裸":
            文本 = 裸生成(model, tokenizer, 消息, seed_offset, 训练对数 + i)
        else:
            if 会话 is not None:
                会话.重置()
            文本 = 会话.生成(消息, seed_offset, 训练对数 + i, d["user"]) if 会话 else ""
        AI测试文本.append(文本)
        记录日志(f"[AI测试 {i+1}/{测试对数}] {d['user'][:14]} => {文本[:28]}")
    del model, tokenizer
    if 会话 is not None:
        try:
            会话.解码器.重置()
        except Exception:  # noqa: BLE001
            pass
        del 会话
    gc.collect()
    torch.cuda.empty_cache()
    return AI训练文本, AI测试文本


def 检测阶段(模式, args, 对话, AI训练文本, AI测试文本):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score

    记录日志(f"──── 模式 [{模式}] 检测器训练（TF-IDF(1-2gram)+LR）────")
    X人类 = [d["girl"] for d in 对话[:训练对数]]
    XAI = AI训练文本
    X训练 = X人类 + XAI
    y训练 = [1] * len(X人类) + [0] * len(XAI)

    X人类测试 = [d["girl"] for d in 对话[训练对数:训练对数 + 测试对数]]
    XAI测试 = AI测试文本
    X测试 = X人类测试 + XAI测试
    y测试 = [1] * len(X人类测试) + [0] * len(XAI测试)

    检测器 = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True),
        LogisticRegression(max_iter=1000),
    )
    t0 = time.time()
    检测器.fit(X训练, y训练)
    训练acc = accuracy_score(y训练, 检测器.predict(X训练))
    测试acc = accuracy_score(y测试, 检测器.predict(X测试))
    记录日志(f"[{模式}] 检测器训练完成 {time.time()-t0:.1f}s | 训练准确率 {训练acc:.3f} | 留出测试准确率 {测试acc:.3f}")

    AI判AI = sum(1 for t in XAI测试 if 检测器.predict([t])[0] == 0)
    真人判AI = sum(1 for t in X人类测试 if 检测器.predict([t])[0] == 0)
    n_ai = len(XAI测试)
    n_hu = len(X人类测试)
    汇总 = {
        "模式": 模式,
        "detection_accuracy": round(AI判AI / n_ai, 4) if n_ai else 0.0,
        "human_likeness_score": round((n_ai - AI判AI) / n_ai, 4) if n_ai else 0.0,
        "人类文本误判率": round(真人判AI / n_hu, 4) if n_hu else 0.0,
        "检测器训练准确率": round(训练acc, 4),
        "检测器留出测试准确率": round(测试acc, 4),
        "AI文本数": n_ai,
        "真人文本数": n_hu,
        "AI判AI明细": [{"user": d["user"], "文本": t[:120],
                        "预测": "AI" if 检测器.predict([t])[0] == 0 else "人类"}
                       for d, t in zip(对话[训练对数:训练对数 + 测试对数], XAI测试)],
        "方法": "中文体系：TF-IDF(1-2gram)+LR，真人(girl) vs 1.5B 生成回复，留出测试",
    }
    记录日志(f"[{模式}] {json.dumps({k: v for k, v in 汇总.items() if k != 'AI判AI明细'}, ensure_ascii=False)}")
    return 汇总


def 主程序():
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", choices=["裸", "锚点", "全部"], default="全部")
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--β", type=float, default=0.8)
    ap.add_argument("--T_anchor", type=float, default=0.3)
    ap.add_argument("--稀疏阈值", type=float, default=0.0)
    ap.add_argument("--只生成", action="store_true", help="只跑生成阶段并缓存")
    ap.add_argument("--只检测", action="store_true", help="只跑检测阶段（读生成缓存）")
    args = ap.parse_args()
    模式列表 = ["裸", "锚点"] if args.模式 == "全部" else [args.模式]

    if not (args.只检测 or args.只生成):
        if os.path.exists(日志路径):
            os.remove(日志路径)
    记录日志(f"=== TuringBench 锚点评测（中文体系检测器）模式={模式列表} "
              f"β={args.β} T_anchor={args.T_anchor} 稀疏阈值={args.稀疏阈值} ===")
    记录日志(f"P4 降级情况：潮汐感知器/cnsenti 可用={_潮汐可用}，导入错误={_潮汐导入错误 or '无'}")
    t0 = time.time()

    对话 = 加载对话()
    random.seed(args.seed_base)
    random.shuffle(对话)
    记录日志(f"对话总数 {len(对话)} | 训练 {训练对数} 对 | 测试 {测试对数} 对")

    全部汇总 = {}
    各模式明细 = {}
    for 模式 in 模式列表:
        缓存文件 = os.path.join(输出目录, f"TuringBench_生成_锚点_{模式}.json")
        if not args.只检测 and not os.path.exists(缓存文件):
            AI训练文本, AI测试文本 = 生成AI文本(模式, args, 对话)
            with open(缓存文件, "w", encoding="utf-8") as f:
                json.dump({"模式": 模式, "AI训练文本": AI训练文本, "AI测试文本": AI测试文本},
                          f, ensure_ascii=False, indent=2)
            记录日志(f"[生成] 缓存已保存 -> {缓存文件}")
        if args.只生成:
            continue
        if not os.path.exists(缓存文件):
            continue
        with open(缓存文件, encoding="utf-8") as f:
            _缓存 = json.load(f)
        汇总 = 检测阶段(模式, args, 对话, _缓存["AI训练文本"], _缓存["AI测试文本"])
        汇总["总用时秒"] = round(time.time() - t0, 1)
        全部汇总[模式] = 汇总
        各模式明细[模式] = _缓存

    if "裸" in 全部汇总 and "锚点" in 全部汇总:
        for 键 in ["detection_accuracy", "human_likeness_score", "人类文本误判率"]:
            v0 = 全部汇总["裸"][键]
            v1 = 全部汇总["锚点"][键]
            相对 = (v1 / v0 - 1.0) if v0 else None
            记录日志(f"对比[{键}] 裸 {v0} → 锚点 {v1} (Δ {v1 - v0:+.4f}"
                      + (f"，相对 {相对:+.2%})" if 相对 is not None else ")"))
        # 判定：人类相似度不下降（≥ 裸）
        h0 = 全部汇总["裸"]["human_likeness_score"]
        h1 = 全部汇总["锚点"]["human_likeness_score"]
        达成 = h1 >= h0
        记录日志(f"判定[human_likeness ≥ 裸（不下降，兜底保底）]：裸 {h0} → 锚点 {h1}"
                  f"（Δ {h1 - h0:+.4f}）→ {'✓ 达成（不下降）' if 达成 else '✗ 下降（记录原因）'}")

    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump({"配置": {"目标模型": 目标模型名, "seed_base": args.seed_base,
                           "β": args.β, "T_anchor": args.T_anchor,
                           "稀疏阈值": args.稀疏阈值, "K": 6},
                   "模式汇总": 全部汇总,
                   "各模式生成": {k: {"AI训练文本": v["AI训练文本"], "AI测试文本": v["AI测试文本"]}
                                 for k, v in 各模式明细.items()}}, f, ensure_ascii=False, indent=2)
    记录日志(f"结果已保存 -> {结果路径}")
    return 全部汇总


if __name__ == "__main__":
    主程序()
