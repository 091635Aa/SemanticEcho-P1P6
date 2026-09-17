# -*- coding: utf-8 -*-
"""P4 锚点回响（Anchor Echo）· Task9.2 纯黑盒锚点提示退化模式验证（三级降级链路 ③）

用 锚点解码器(接口="提示")：完全不访问模型内部（模拟纯黑盒 API），只把当前
维度锚点词构造进 prompt（接口降级.py 的 构造提示词()），模型生成时自然带情感方向。

对照：裸（接口="本地" β=0，无注入、无提示改写）——同提示/同种子/同统计口径。

验证点：
  a) 构造提示词() 输出合理（提示模板可见、锚点词注入正确）；
  b) 5 条提示词生成 + 指标（熵/重复率/命中率）+ 文本抽样；
  c) 代码层面确认全程不访问模型内部接口：
       - 黑盒锚点库（纯文本词集，零 embedding），预计算打分表 打点计数应为 0；
       - model.get_input_embeddings 打点计数应为 0（模型前向内部用 embed_tokens，
         不经 get_input_embeddings；本地模式构建打分表时该计数 >0，形成对照）；
       - 生成期 注入偏置 的分发入口 _注入偏置稠密/_注入偏置logprobs 打点计数应为 0
         （提示模式 logits 原样返回，零注入）。

判定（任务要求）：
  - 生成无异常（无空回复、无坍缩）；
  - 输出带目标情感方向（目测 + 文本级情感词命中率 > 裸对照 或用情感词出现）；
  - 全程不访问模型内部接口（代码层面打点确认）。

用法：python 验证_提示模式.py [--种子 42] [--生成长度 128]
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 离线加载本地模型，必须在 transformers import 前

import argparse
import gc
import glob
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from 锚点库 import 默认词集
from 目标决策器 import 目标决策器
from 锚点解码器 import 锚点解码器
from 接口降级 import 构造提示词

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
输出目录 = os.path.join(本目录, "评测结果")
os.makedirs(输出目录, exist_ok=True)

# 5 条情感维度提示（难过/开心/担心/委屈/中性）
提示词表 = [
    {"维度": "难过", "user": "我今天真的好难过，感觉做什么都提不起劲……"},
    {"维度": "开心", "user": "我今天特别开心，遇到了好多高兴的事情！"},
    {"维度": "担心", "user": "我有点担心明天的事情，心里七上八下的。"},
    {"维度": "委屈", "user": "我明明没有做错，却被人误会了，心里好委屈。"},
    {"维度": "中性", "user": "今天天气不错，我准备去公园散散步。"},
]


class 黑盒锚点库:
    """纯黑盒锚点库：只提供文本词集与维度名（零 embedding 访问）。

    预计算打分表() 打点计数——提示模式按设计不应调用（调用即代表访问了
    模型内部接口的入口），计数 >0 视为违规。
    """

    def __init__(self):
        self.词集 = 默认词集
        self.预计算打分表调用次数 = 0

    def 维度名(self):
        return list(self.词集.keys())

    def 预计算打分表(self, 缓存路径=None):
        self.预计算打分表调用次数 += 1
        return None


def 构建提示(tokenizer, 消息):
    return tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)


def 文本级情感命中率(回复, 词集):
    """文本级情感种子词命中率：锚点库词集子串命中数 / 回复长度"""
    if not 回复:
        return 0.0
    命中 = 0
    for 词列表 in 词集.values():
        for 词 in 词列表:
            if 词 and 词 in 回复:
                命中 += 1
    return round(命中 / max(len(回复), 1), 4)


def 加载模型():
    gc.collect()
    torch.cuda.empty_cache()
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
    模型.eval()
    return 模型, 分词器


class 会话:
    """裸 / 提示模式共享会话：一条提示词一次目标计算。

    裸   ：接口="本地" β=0 —— 零注入、prompt 不改写（统计口径与提示模式一致）；
    提示 ：接口="提示" β=0.8 —— 零内部访问、当前维度锚点词注入 prompt。
    """

    def __init__(self, model, tokenizer, 库, 接口, β=0.8):
        self.model = model
        self.tokenizer = tokenizer
        self.接口 = 接口
        self.库 = 库
        self.目标决策器 = 目标决策器(锚点库=库, β基=β)
        self.解码器 = 锚点解码器(
            model, tokenizer, 库, self.目标决策器,
            β=β, T_anchor=0.3, 接口=接口, topk候选=100,
            温度=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
        )

    def 重置(self):
        try:
            self.目标决策器.感知器.重置轨迹()
        except Exception:  # noqa: BLE001
            pass

    def 生成(self, 消息, 种子, 用户文本, 生成长度):
        torch.manual_seed(种子)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(种子)
        提示 = 构建提示(self.tokenizer, 消息)
        inputs = self.tokenizer(提示, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            ids, 统计 = self.解码器.生成(
                inputs.input_ids, max_new_tokens=生成长度,
                eos_token_id=self.tokenizer.eos_token_id, tokenizer=self.tokenizer,
                用户文本=用户文本,
            )
        # 提示模式：生成() 内部用 构造提示词(用户文本) 重写 prompt（变长），
        # 切片点须按重写后的 prompt token 数计算
        if self.接口 == "提示":
            prompt_ids = self.tokenizer(self.解码器.构造提示词(用户文本),
                                        return_tensors="pt").input_ids
            新token = ids[0, prompt_ids.shape[1]:]
        else:
            新token = ids[0, inputs.input_ids.shape[1]:]
        回复 = self.tokenizer.decode(新token, skip_special_tokens=True).strip()
        统计["token数"] = int(len(新token))
        return 回复, 统计


def 汇总健康度(统计列表, 回复列表, 词集):
    汇总 = {}
    for k in ("平均熵", "重复率", "情感命中率"):
        值列表 = [s.get(k, 0.0) for s in 统计列表]
        汇总[k] = round(sum(值列表) / max(len(值列表), 1), 4)
    汇总["触发兜底次数均值"] = round(
        sum(s.get("触发兜底次数", 0) for s in 统计列表) / max(len(统计列表), 1), 4)
    文本命中 = [文本级情感命中率(r, 词集) for r in 回复列表]
    汇总["文本级情感命中率"] = round(sum(文本命中) / max(len(文本命中), 1), 4)
    长度 = [len(r) for r in 回复列表]
    汇总["平均长度"] = round(sum(长度) / max(len(长度), 1), 2)
    汇总["平均token数"] = round(sum(s.get("token数", 0) for s in 统计列表)
                                / max(len(统计列表), 1), 1)
    汇总["空回复数"] = sum(1 for r in 回复列表 if not r.strip())
    汇总["最小熵"] = min(s["平均熵"] for s in 统计列表)
    汇总["最大重复率"] = max(s["重复率"] for s in 统计列表)
    return 汇总


def 跑模式(model, tokenizer, 库, 接口, β, 提示词表, 种子基线, 生成长度, 黑盒检查=None):
    会话实例 = 会话(model, tokenizer, 库, 接口, β=β)
    # 注入入口打点：_注入偏置稠密/_注入偏置logprobs 是 logits 注入的唯一入口，
    # 提示模式（接口="提示"）按设计 注入偏置() 原样返回 logits，不应触达
    if 黑盒检查 is not None:
        原始稠密 = 会话实例.解码器._注入偏置稠密
        原始受限 = 会话实例.解码器._注入偏置logprobs

        def 打点稠密(logits):
            黑盒检查["注入入口计数"] += 1
            return 原始稠密(logits)

        def 打点受限(logits):
            黑盒检查["注入入口计数"] += 1
            return 原始受限(logits)

        会话实例.解码器._注入偏置稠密 = 打点稠密
        会话实例.解码器._注入偏置logprobs = 打点受限
    回复列表, 统计列表 = [], []
    for i, 项 in enumerate(提示词表):
        消息 = [{"role": "user", "content": 项["user"]}]
        种子 = 种子基线 + i
        会话实例.重置()
        回复, 统计 = 会话实例.生成(消息, 种子, 项["user"], 生成长度)
        回复列表.append(回复)
        统计列表.append(统计)
        print(f"  [{接口} {i+1}/{len(提示词表)} {项['维度']}] "
              f"熵{统计['平均熵']} 重{统计['重复率']} tok{统计['token数']} "
              f"=> {回复[:36]}")
    健康度 = 汇总健康度(统计列表, 回复列表, 库.词集)
    return {"接口": 接口, "回复": 回复列表, "统计": 统计列表, "健康度": 健康度}


def 判定(结果裸, 结果提示, 黑盒检查):
    """9.2 判定：生成无异常 + 情感方向（命中率>裸或用情感词）+ 零内部访问"""
    hN, hT = 结果裸["健康度"], 结果提示["健康度"]
    命中差 = hT["文本级情感命中率"] - hN["文本级情感命中率"]
    用情感词出现 = 结果提示["健康度"]["文本级情感命中率"] > 0.0

    判定结果 = {
        "生成无异常": bool(hT["空回复数"] == 0 and hT["平均熵"] > 0.8
                            and hT["最大重复率"] < 0.6),
        "文本级命中率差(提示-裸)": round(命中差, 4),
        "命中率>裸": bool(命中差 > 0.0),
        "用情感词出现": bool(用情感词出现),
        "黑盒_零get_input_embeddings": bool(黑盒检查["get_input_embeddings计数"] == 0),
        "黑盒_零预计算打分表": bool(黑盒检查["预计算打分表计数"] == 0),
        "黑盒_零注入入口": bool(黑盒检查["注入入口计数"] == 0),
    }
    判定结果["零内部访问"] = bool(
        判定结果["黑盒_零get_input_embeddings"]
        and 判定结果["黑盒_零预计算打分表"] and 判定结果["黑盒_零注入入口"])
    判定结果["情感方向"] = bool(判定结果["命中率>裸"] or 判定结果["用情感词出现"])
    判定结果["通过"] = bool(判定结果["生成无异常"] and 判定结果["情感方向"]
                            and 判定结果["零内部访问"])
    return 判定结果


def 主程序():
    ap = argparse.ArgumentParser()
    ap.add_argument("--种子", type=int, default=42)
    ap.add_argument("--生成长度", type=int, default=128)
    args = ap.parse_args()

    print("=" * 70)
    print("P4 锚点回响 · Task9.2 纯黑盒锚点提示退化模式验证（三级降级链路 ③）")
    print("=" * 70)
    print(f"[1/5] 加载模型 {模型路径} ...")
    model, tokenizer = 加载模型()
    print(f"  模型 dtype={model.dtype} device={model.device}")

    # ── 验证 a：构造提示词() 模板（无需模型，先展示）──
    # 注意：任务提示词标签（担心/委屈/中性）是口语情感标签，锚点库维度为
    # 温柔/开心/难过/愤怒/害怕/平静；实际注入用的是 主导维度()（v_target 的
    # argmax 维，经 VAD 原型表自动映射），故模板样例按每条提示的真实主导维度展示。
    print("[2/5] 构造提示词() 模板验证（提示模板可见、锚点词注入正确）...")
    模板样例 = {}
    for 项 in 提示词表:
        临时库 = 黑盒锚点库()
        决策 = 目标决策器(锚点库=临时库)
        目标 = 决策.计算目标(用户当前=项["user"])
        主导 = 临时库.维度名()[int(np.argmax(目标.v_target))]
        提示 = 构造提示词(主导, 临时库, 项["user"])
        模板样例[项["维度"]] = {"用户提示标签": 项["维度"], "主导维度": 主导, "模板": 提示}
        print(f"  [{项['维度']} → 主导维度「{主导}」] {提示}")

    # ── 代码层面黑盒打点 ──
    print("[3/5] 黑盒打点（get_input_embeddings / 预计算打分表 / 注入入口）...")
    黑盒检查 = {"get_input_embeddings计数": 0, "预计算打分表计数": 0, "注入入口计数": 0}
    原始get = model.get_input_embeddings

    def 打点get():
        黑盒检查["get_input_embeddings计数"] += 1
        return 原始get()

    model.get_input_embeddings = 打点get  # 实例属性遮蔽方法（仅本进程）

    库 = 黑盒锚点库()

    print(f"[4/5] 双模式生成（5 条提示，种子 {args.种子}，max_new_tokens={args.生成长度}）...")
    结果裸 = 跑模式(model, tokenizer, 库, "本地", β=0.0, 提示词表=提示词表,
                 种子基线=args.种子, 生成长度=args.生成长度, 黑盒检查=黑盒检查)
    print(f"  [裸对照完成] get_input_embeddings 计数 = {黑盒检查['get_input_embeddings计数']}"
          f"（黑盒锚点库零 embedding，两模式都应保持 0）")
    注入前 = 黑盒检查["注入入口计数"]
    打分表前 = 库.预计算打分表调用次数
    结果提示 = 跑模式(model, tokenizer, 库, "提示", β=0.8, 提示词表=提示词表,
                 种子基线=args.种子, 生成长度=args.生成长度, 黑盒检查=黑盒检查)
    # 提示模式阶段增量：预计算打分表 / 注入入口 在该模式期间必须为 0
    黑盒检查["预计算打分表计数"] = 库.预计算打分表调用次数 - 打分表前
    黑盒检查["注入入口计数"] = 黑盒检查["注入入口计数"] - 注入前

    print(f"[5/5] 判定与保存 ...")
    判定结果 = 判定(结果裸, 结果提示, 黑盒检查)
    hN, hT = 结果裸["健康度"], 结果提示["健康度"]
    print("\n──── 三指标对照（裸 vs 提示）────")
    print(f"  平均熵          裸 {hN['平均熵']:.4f}  |  提示 {hT['平均熵']:.4f}")
    print(f"  重复率          裸 {hN['重复率']:.4f}  |  提示 {hT['重复率']:.4f}")
    print(f"  情感命中(token) 裸 {hN['情感命中率']:.4f}  |  提示 {hT['情感命中率']:.4f}")
    print(f"  情感命中(文本)  裸 {hN['文本级情感命中率']:.4f}  |  提示 {hT['文本级情感命中率']:.4f}  |  差 {判定结果['文本级命中率差(提示-裸)']:+.4f}")
    print(f"  平均长度        裸 {hN['平均长度']}  |  提示 {hT['平均长度']}")
    print(f"  空回复          裸 {hN['空回复数']}  |  提示 {hT['空回复数']}")
    print("\n──── 黑盒检查（代码层面）────")
    print(f"  get_input_embeddings 调用次数 = {黑盒检查['get_input_embeddings计数']}（0 = 提示模式零 embedding 访问）")
    print(f"  预计算打分表调用次数 = {黑盒检查['预计算打分表计数']}（0 = 提示模式零打分表）")
    print(f"  注入入口调用次数 = {黑盒检查['注入入口计数']}（0 = 提示模式零 logits 注入）")
    print(f"\n>>> 判定：{'通过 ✓' if 判定结果['通过'] else '不通过 ✗'}  "
          f"（生成无异常: {判定结果['生成无异常']}，情感方向: {判定结果['情感方向']}，"
          f"零内部访问: {判定结果['零内部访问']}）")

    print("\n──── 文本抽样（全部 5 条）────")
    for 项, rN, rT in zip(提示词表, 结果裸["回复"], 结果提示["回复"]):
        print(f"\n  用户：{项['user']}")
        print(f"  裸   ：{rN}")
        print(f"  提示 ：{rT}")

    汇总 = {
        "任务": "Task9.2 纯黑盒锚点提示退化模式验证（三级降级链路 ③）",
        "模型": "Qwen2.5-1.5B-Instruct", "dtype": str(model.dtype),
        "配置": {"种子": args.种子, "生成长度": args.生成长度,
                "裸": "接口='本地' β=0（零注入、prompt 不改写，统计口径对照）",
                "提示": "接口='提示' β=0.8（零内部访问，构造提示词注入 prompt）",
                "采样": "T=1.0 top_p=0.9 top_k=50 rep_pen=1.05"},
        "提示词": 提示词表,
        "模板样例": 模板样例,
        "裸": 结果裸, "提示": 结果提示,
        "黑盒检查": 黑盒检查,
        "判定": 判定结果,
    }
    时间戳 = time.strftime("%Y%m%d_%H%M%S")
    保存路径 = os.path.join(输出目录, f"接口降级_{时间戳}.json")

    # 合并 9.1 结果（若存在最近一个 接口降级_*.json）
    已有9_1 = glob.glob(os.path.join(输出目录, "接口降级_*.json"))
    for 旧 in 已有9_1:
        if 旧 == 保存路径:
            continue
        try:
            with open(旧, encoding="utf-8") as f:
                旧内容 = json.load(f)
            if 旧内容.get("任务", "").startswith("Task9.1"):
                汇总["Task9.1"] = 旧内容
                break
        except Exception:  # noqa: BLE001
            continue
    with open(保存路径, "w", encoding="utf-8") as f:
        json.dump(汇总, f, ensure_ascii=False, indent=2)
    print(f"\n>>> 结果已保存 -> {保存路径}")

    del model, tokenizer, 库
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    主程序()
