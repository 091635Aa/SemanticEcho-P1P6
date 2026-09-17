# -*- coding: utf-8 -*-
"""P6 旁路由选优生成器（ELS 核心·推理态）

设计：主路径零注入 —— 仅标准采样生成 N 条候选，侧路路由器按情感质量评分选优。
- 多候选：LoRA 挂载模型按 N 个独立种子各生成一条
- 路由评分：情感命中 + 长度约束 + AI腔惩罚 + 重复惩罚
- 兜底：全部候选不合格 → 裸采样兜底

指标口径与前辈一致：
- 情感命中率 = 情感token命中数 / 生成token总数（情感token集 = 静态情感词 + cnsenti 词库单token词）
- 重复率 = 2-gram token 重复率 = 1 - 唯一/总数
- 平均熵 = 生成 token 的 softmax 熵均值（一次前向）
"""
import math
import os
import re
import sys
from collections import Counter

# ── 内存约束（低占用运行，避免挤占同机其他 AI 进程）──
os.environ.setdefault("HF_HOME", r"C:\P6临时盘\hf")
os.environ.setdefault("TORCH_HOME", r"C:\P6临时盘\torch")
os.environ.setdefault("TEMP", r"D:\P6临时盘\tmp")
os.environ.setdefault("TMP", r"D:\P6临时盘\tmp")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

torch.set_num_threads(4)

# ── 情感 token 集（单 token 情感词）──
静态情感词 = [
    # 积极
    "开心", "高兴", "幸福", "喜欢", "爱", "温暖", "感动", "希望", "加油", "恭喜",
    "棒", "好", "真", "值得", "放心", "安心", "舒服", "轻松", "快乐", "甜蜜",
    "治愈", "可爱", "温柔", "心疼", "心软", "灿烂", "笑容", "谢谢", "感谢",
    # 消极/共情
    "难过", "伤心", "痛苦", "害怕", "担心", "焦虑", "委屈", "孤独", "失落",
    "沮丧", "失望", "心累", "疲惫", "累", "哭", "痛", "烦", "慌", "迷茫",
    "后悔", "遗憾", "心疼你", "抱抱", "陪你", "理解", "明白", "懂", "在乎",
    "珍惜", "勇气", "努力", "坚持", "支持", "相信",
]

AI腔词 = [
    "我无法", "无法提供", "对不起，我无法", "抱歉，我无法",
    "作为AI", "作为AI助手", "我是一个AI", "我是AI", "AI助手", "语言模型",
    "被编程", "温馨提示", "请注意", "如果您有任何", "请随时告诉我",
    "设定清晰的目标", "设定目标", "作为一个AI",
    "帮助您更好地", "我的目的是", "我会尽力", "我无法理解", "我不明白",
]


class P6旁路由生成器:
    """P6 生成器：LoRA 外挂 + 旁路由选优"""

    采样默认 = dict(temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05)

    def __init__(self, 模型路径, lora路径, 设备=None, 挂载=True):
        self._设备 = 设备 or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[P6] 加载基座 {os.path.basename(模型路径)} fp16 ...")
        self._分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
        if self._分词器.pad_token is None:
            self._分词器.pad_token = self._分词器.eos_token
        self._模型 = AutoModelForCausalLM.from_pretrained(
            模型路径, torch_dtype=torch.float16, trust_remote_code=True,
            low_cpu_mem_usage=True).to(self._设备)
        self._模型.eval()
        self._挂载 = 挂载
        if 挂载:
            print(f"[P6] 挂载 LoRA: {os.path.basename(lora路径)}")
            self._模型 = PeftModel.from_pretrained(self._模型, lora路径)
            self._模型 = self._模型.merge_and_unload()
            self._模型.eval()
        self._情感token集 = self._构建情感token集()

    # ────────────────────────────────
    # 情感 token 集
    # ────────────────────────────────
    def _构建情感token集(self):
        集 = set()
        for 词 in 静态情感词:
            ids = self._分词器.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                集.add(ids[0])
        try:
            from cnsenti import Sentiment
            s = Sentiment()
            正面 = set(getattr(s, "Poss", []) or [])
            负面 = set(getattr(s, "Negs", []) or [])
            for 词 in 正面 | 负面:
                ids = self._分词器.encode(词, add_special_tokens=False)
                if len(ids) == 1:
                    集.add(ids[0])
        except Exception as e:  # noqa: BLE001
            print(f"[P6] cnsenti 词库不可用：{e}")
        print(f"[P6] 情感token集大小：{len(集)}")
        return 集

    # ────────────────────────────────
    # 指标
    # ────────────────────────────────
    @staticmethod
    def 计算重复率(token列表, 阶数=2):
        if len(token列表) < 阶数 + 1:
            return 0.0
        ngrams = [tuple(token列表[i:i + 阶数]) for i in range(len(token列表) - 阶数 + 1)]
        return round(1.0 - len(set(ngrams)) / max(len(ngrams), 1), 4)

    def 计算情感命中率(self, token列表):
        if not token列表:
            return 0.0
        命中 = sum(1 for t in token列表 if t in self._情感token集)
        return round(命中 / len(token列表), 4)

    def 计算平均熵(self, token列表):
        if len(token列表) < 2:
            return 1.0
        ids = torch.tensor([token列表[:-1]], device=self._设备)
        with torch.no_grad():
            logits = self._模型(input_ids=ids).logits[0].float()
        probs = F.softmax(logits, dim=-1)
        logp = torch.log(probs.clamp_min(1e-9))
        熵 = -(probs * logp).sum(-1)
        return round(float(熵.mean().cpu()), 4)

    # ────────────────────────────────
    # 路由评分
    # ────────────────────────────────
    def 路由评分(self, 文本, token列表):
        长度 = len(文本)
        情感命中 = self.计算情感命中率(token列表)
        # 长度分：15~80 字为优
        if 10 <= 长度 <= 80:
            长度分 = 1.0 - abs(长度 - 40) / 80.0
        elif 长度 < 10:
            长度分 = -0.5
        else:
            长度分 = -1.0
        # AI腔惩罚
        ai数 = sum(1 for w in AI腔词 if w in 文本)
        ai惩罚 = min(ai数, 3) * 1.0
        # 重复惩罚
        重复 = self.计算重复率(token列表)
        重复惩罚 = 重复 * 3.0
        # 综合分
        分数 = 2.5 * 情感命中 + 1.2 * max(长度分, 0) - ai惩罚 - 重复惩罚
        return 分数, {"情感命中": 情感命中, "长度": 长度, "ai腔": ai数, "重复": 重复, "分数": round(分数, 4)}

    # ────────────────────────────────
    # 生成
    # ────────────────────────────────
    def _单次生成(self, 提示, 种子, max_new_tokens, 采样参数):
        torch.manual_seed(种子)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(种子)
        with torch.no_grad():
            out = self._模型.generate(
                提示, max_new_tokens=max_new_tokens,
                pad_token_id=self._分词器.eos_token_id,
                do_sample=True, **采样参数)
        新 = out[0, 提示.shape[1]:]
        文本 = self._分词器.decode(新, skip_special_tokens=True).strip()
        return 文本, 新.tolist()

    def 生成(self, 消息, 种子=2026, N=3, max_new_tokens=128,
            候选种子步长=7, 采样参数=None, 返回候选=False):
        """P6 旁路由生成：N 候选 → 情感路由选优 → 兜底裸采样"""
        采样 = 采样参数 or dict(self.采样默认)
        提示文本 = self._分词器.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
        inputs = self._分词器(提示文本, return_tensors="pt").to(self._设备)
        提示 = inputs.input_ids

        候选列表 = []
        兜底 = 0
        for i in range(N):
            候选种子 = 种子 + i * 候选种子步长
            文本, tokens = self._单次生成(提示, 候选种子, max_new_tokens, 采样)
            if not 文本.strip() or len(文本.strip()) < 4:
                兜底 += 1
                continue
            分数, 明细 = self.路由评分(文本, tokens)
            候选列表.append({"种子": 候选种子, "文本": 文本, "tokens": tokens,
                             "分数": 明细["分数"], "明细": 明细})

        if not 候选列表:
            # 全空 → 裸兜底
            兜底 += 1
            文本, tokens = self._单次生成(提示, 种子, max_new_tokens, 采样)
            候选列表.append({"种子": 种子, "文本": 文本, "tokens": tokens,
                             "分数": 0.0, "明细": {"情感命中": 0, "长度": len(文本), "ai腔": 0, "重复": 0, "分数": 0.0}})

        最佳 = max(候选列表, key=lambda c: c["分数"])
        文本, tokens, 明细 = 最佳["文本"], 最佳["tokens"], 最佳["明细"]

        统计 = {
            "平均熵": self.计算平均熵(tokens),
            "重复率": self.计算重复率(tokens),
            "情感命中率": 明细["情感命中"],
            "长度": len(文本),
            "触发兜底次数": 兜底,
            "路由分数": 明细["分数"],
            "候选数": len(候选列表),
            "ai腔": 明细["ai腔"],
        }
        if 返回候选:
            return 文本, tokens, 统计, 候选列表
        return 文本, tokens, 统计

    def 清理(self):
        import gc
        self._模型 = None
        self._分词器 = None
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
