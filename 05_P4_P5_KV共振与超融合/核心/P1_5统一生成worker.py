# -*- coding: utf-8 -*-
"""P1~P5 统一测试 —— 生成阶段 worker（样本切片并行）

每个 worker 加载一次 1.5B 基座，为 [start, end) 区间的样本跑全部 7 模式
（裸 / P1语义回响 / P1.5兼容层 / P2.5潮汐 / P3锚点回响 / P4 KV共振 / P5超融合），
统一种子 2026（控制变量）。输出到指定 part JSON。

用法: python P1_5统一生成worker.py --start 0 --end 10 --out part0.json
"""
import os
import sys
import json
import argparse
import math
from collections import Counter

import numpy as np
import torch

工作目录 = os.path.dirname(os.path.abspath(__file__))
if 工作目录 not in sys.path:
    sys.path.insert(0, 工作目录)
回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)
ETD目录 = r"h:\情感潮汐解码（Emotion Tidal Decoding, ETD）"
if ETD目录 not in sys.path:
    sys.path.insert(0, ETD目录)

from transformers import AutoModelForCausalLM, AutoTokenizer

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器
from 锚点解码器 import 锚点解码器, 计算熵, 计算重复率
from 超融合解码器 import 超融合解码器
from 情感共振解码器 import 情感共振解码器

from semantic_echo.回响池 import 语义回响池
from semantic_echo.采样处理器 import 回响注入器
from semantic_echo.情感过滤器 import 情感过滤器

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
样本路径 = r"i:\Desktop\语义回响\图灵测试\样本_30条.json"
种子 = 2026
最大长度 = 256

模式列表 = ["裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响", "P4_KV共振", "P5_超融合"]


class _GPU回响注入器(回响注入器):
    """GPU 直分配投影矩阵（父类 CPU 分配 933MB，且每步 to(device) 会拖慢）"""

    def _初始化投影(self, seed: int) -> None:
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        scale = math.sqrt(2.0 / self.hidden_dim)
        self.投影矩阵 = torch.randn(
            self.hidden_dim, self.vocab_size,
            generator=rng, dtype=torch.float32, device=self.device,
        ) * scale
        self.投影矩阵.requires_grad_(False)


def 裸生成(model, tokenizer, 提示, 种子):
    """裸：无注入自回归循环，逐 token 前向记录熵（与各解码器同口径）"""
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    消息 = [{"role": "user", "content": 提示}]
    应用提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(应用提示, return_tensors="pt").to(model.device)
    past = None
    已生成 = inputs.input_ids.clone()
    新token列表 = []
    熵列表 = []
    eos = tokenizer.eos_token_id
    with torch.no_grad():
        for _ in range(最大长度):
            输入 = 已生成[:, -1:] if past is not None else 已生成
            out = model(输入, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            past = out.past_key_values
            熵列表.append(计算熵(logits))
            for tid in set(新token列表):
                logits[0, tid] /= 1.05
            logits = logits / 1.0
            # top-p
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, stable=True)
            cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cum > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            # top-k
            topk_v, _ = torch.topk(logits, min(50, logits.size(-1)), dim=-1)
            logits[logits < topk_v[:, -1].unsqueeze(-1)] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, num_samples=1)
            已生成 = torch.cat([已生成, tok], dim=-1)
            新token列表.append(tok.item())
            if tok.item() == eos:
                break
    文本 = tokenizer.decode(新token列表, skip_special_tokens=True).strip()
    统计 = {"平均熵": round(sum(熵列表) / len(熵列表), 4) if 熵列表 else 0.0,
            "重复率": 计算重复率(新token列表),
            "触发兜底次数": 0, "种子": 种子}
    return 文本, 统计, 新token列表


def 统一情感token集(tokenizer, 锚库, 目标决策):
    集 = set()
    for 维, 词列表 in 锚库.词集.items():
        for 词 in 词列表:
            ids = tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                集.add(ids[0])
    try:
        感知器 = 目标决策.感知器
        for 词 in getattr(感知器, "_正面词", set()) | getattr(感知器, "_负面词", set()):
            ids = tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                集.add(ids[0])
    except Exception:
        pass
    return 集


def 情感命中率(token列表, 情感集):
    if not token列表:
        return 0.0
    return round(sum(1 for t in token列表 if t in 情感集) / len(token列表), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    print(f"[worker {args.start}-{args.end}] 加载模型 {模型路径} fp16 ...", flush=True)
    设备 = "cuda"
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16, trust_remote_code=True).to(设备)
    模型.eval()
    torch.cuda.empty_cache()
    print("[worker] 模型加载完成", flush=True)

    锚库 = 锚点库(模型, 分词器)
    锚库.构建()
    print(f"[worker] 锚点库 K={锚库.锚点矩阵.shape[0]}", flush=True)

    感知器, 潮汐决策 = None, None
    潮汐可用 = False
    try:
        from 潮汐感知器 import 潮汐感知器
        from 潮汐决策器 import 潮汐决策器
        感知器 = 潮汐感知器()
        潮汐决策 = 潮汐决策器(感知器)
        潮汐可用 = True
    except Exception as e:
        print(f"[worker] P2.5 潮汐感知器加载失败（该模式跳过）：{e}", flush=True)
    目标决策 = 目标决策器(感知器=感知器, 潮汐决策器=潮汐决策, 锚点库=锚库)
    print(f"[worker] 目标决策器就绪（简易模式={目标决策._简易模式}）", flush=True)

    # ── P1 回响注入器（GPU 投影 + 情感过滤器）──
    池 = 语义回响池(int(模型.config.hidden_size))
    过滤 = None
    try:
        过滤 = 情感过滤器()
        过滤.加载词库()
    except Exception as e:
        print(f"[worker] 情感过滤器不可用（P1 走无筛选回响）：{e}", flush=True)
    回响 = _GPU回响注入器(模型, 池, lambda_strength=0.29,
                          情感过滤器实例=过滤)
    print("[worker] P1 回响注入器就绪", flush=True)

    # ── 各方案解码器 ──
    解码器 = {}
    解码器["P1.5_兼容层"] = 锚点解码器(模型, 分词器, 锚库, 目标决策,
                                   β=None, 句子停止=True)
    解码器["P3_锚点回响"] = 锚点解码器(模型, 分词器, 锚库, 目标决策,
                                    β=0.8, T_anchor=0.3, 句子停止=True)
    解码器["P4_KV共振"] = 情感共振解码器(模型, 分词器, 锚库, 目标决策,
                                       开启KV调制=True, 开启V调制=False, 开启DSA=True,
                                       κ基=0.15, 情感阈值=0.08, 调制层数=4, 句子停止=True)
    解码器["P5_超融合"] = 超融合解码器(模型, 分词器, 锚库, 目标决策,
                                     开启DSA=True, 开启DMR=True, 开启锚点偏置=False,
                                     α基=0.15, α倍率=1.0, T_emo=0.5, 句子停止=True)
    if 潮汐可用:
        from 潮汐解码器 import 潮汐解码器
        解码器["P2.5_潮汐"] = 潮汐解码器(模型, 分词器, 感知器, 潮汐决策)
    print(f"[worker] 方案解码器：{list(解码器.keys())}", flush=True)

    # ── 释放锚点库大矩阵（W_e fp32 ≈933MB），各解码器打分表已构建完成 ──
    锚库.W_e = None
    锚库._有效权重 = None
    锚库.权重 = None
    torch.cuda.empty_cache()

    情感集 = 统一情感token集(分词器, 锚库, 目标决策)
    print(f"[worker] 情感token集大小={len(情感集)}", flush=True)

    with open(样本路径, "r", encoding="utf-8") as f:
        数据 = json.load(f)
    样本 = 数据["样本"]

    结果 = {"模型": "Qwen2.5-1.5B-Instruct", "种子": 种子,
            "模式": 模式列表, "时间": "", "回复": []}

    for i in range(args.start, args.end):
        项 = 样本[i]
        提示 = 项["user"]
        print(f"[worker] [{i+1}/30] {提示[:16]}...", flush=True)
        条目 = {"序号": 项["序号"], "user": 提示, "girl": 项["girl"], "回复": {}}
        # 裸
        文本, 统计, token列表 = 裸生成(模型, 分词器, 提示, 种子)
        统计["情感命中率"] = 情感命中率(token列表, 情感集)
        统计["长度(字)"] = len(文本)
        条目["回复"]["裸"] = {"文本": 文本, "统计": 统计}
        # P1 语义回响
        torch.manual_seed(种子)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(种子)
        熵列表 = []
        out = 回响.生成(
            分词器(分词器.apply_chat_template([{"role": "user", "content": 提示}],
                                        tokenize=False, add_generation_prompt=True),
                   return_tensors="pt").input_ids.to(模型.device),
            max_new_tokens=最大长度, temperature=1.0, top_p=0.9, top_k=50,
            repetition_penalty=1.05, eos_token_id=分词器.eos_token_id,
            logits_callback=lambda 步, logits: 熵列表.append(计算熵(logits)),
            tokenizer=分词器)
        # 提取新生成部分：按 prompt 长度切片
        _提示ids = 分词器(分词器.apply_chat_template([{"role": "user", "content": 提示}],
                                      tokenize=False, add_generation_prompt=True),
                          return_tensors="pt").input_ids.to(模型.device)
        P1文本 = 分词器.decode(out[0][_提示ids.shape[1]:], skip_special_tokens=True).strip()
        P1tokens = out[0][_提示ids.shape[1]:].tolist()
        条目["回复"]["P1_语义回响"] = {
            "文本": P1文本,
            "统计": {"平均熵": round(sum(熵列表) / len(熵列表), 4) if 熵列表 else 0.0,
                     "重复率": 计算重复率(P1tokens),
                     "情感命中率": 情感命中率(P1tokens, 情感集),
                     "长度(字)": len(P1文本), "触发兜底次数": 0, "种子": 种子}}
        # 其余方案
        for 名称, 解码 in 解码器.items():
            torch.manual_seed(种子)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(种子)
            if 名称 == "P2.5_潮汐":
                熵列表2 = []
                out2 = 解码.生成(
                    分词器(分词器.apply_chat_template([{"role": "user", "content": 提示}],
                                                tokenize=False, add_generation_prompt=True),
                           return_tensors="pt").input_ids.to(模型.device),
                    max_new_tokens=最大长度, temperature=1.0, top_p=0.9, top_k=50,
                    repetition_penalty=1.05, eos_token_id=分词器.eos_token_id,
                    logits_callback=lambda 步, logits: 熵列表2.append(计算熵(logits)),
                    tokenizer=分词器, 用户文本=提示)
                _ids = 分词器(分词器.apply_chat_template([{"role": "user", "content": 提示}],
                                      tokenize=False, add_generation_prompt=True),
                              return_tensors="pt").input_ids.to(模型.device)
                _文本 = 分词器.decode(out2[0][_ids.shape[1]:], skip_special_tokens=True).strip()
                _tokens = out2[0][_ids.shape[1]:].tolist()
                条目["回复"][名称] = {
                    "文本": _文本,
                    "统计": {"平均熵": round(sum(熵列表2) / len(熵列表2), 4) if 熵列表2 else 0.0,
                             "重复率": 计算重复率(_tokens),
                             "情感命中率": 情感命中率(_tokens, 情感集),
                             "长度(字)": len(_文本),
                             "触发兜底次数": int(解码.统计.get("触发兜底", 0))
                             if hasattr(解码, "统计") else 0,
                             "种子": 种子}}
            else:
                out3, 统计3 = 解码.生成(
                    分词器(分词器.apply_chat_template([{"role": "user", "content": 提示}],
                                                tokenize=False, add_generation_prompt=True),
                           return_tensors="pt").input_ids.to(模型.device),
                    max_new_tokens=最大长度, temperature=1.0, top_p=0.9, top_k=50,
                    repetition_penalty=1.05, eos_token_id=分词器.eos_token_id,
                    用户文本=提示)
                _ids = 分词器(分词器.apply_chat_template([{"role": "user", "content": 提示}],
                                      tokenize=False, add_generation_prompt=True),
                              return_tensors="pt").input_ids.to(模型.device)
                _文本 = 分词器.decode(out3[0][_ids.shape[1]:], skip_special_tokens=True).strip()
                统计3["长度(字)"] = len(_文本)
                统计3["种子"] = 种子
                条目["回复"][名称] = {"文本": _文本, "统计": 统计3}
        结果["回复"].append(条目)
        if (i + 1) % 5 == 0:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(结果, f, ensure_ascii=False, indent=2)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=2)
    print(f"[worker] 完成，已保存：{args.out}", flush=True)


if __name__ == "__main__":
    main()
