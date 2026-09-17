# -*- coding: utf-8 -*-
"""P6 情感导演解码器（Emotional Director Decoding, EDD）—— 零权重万能插件

背景（对照前辈缺陷）：
  P5 超融合虽综合最高（0.6255），但存在**维度倒退**：
    情感理解 0.6753 < 裸 0.7420、EmoCharacter 0.8762 < 裸 0.9090、
    FEEL 行为决策 0.1 < 裸 0.25 —— 固定强度注入干扰了非情感任务。

P6 EDD 的三大新机制（全部零权重，解码期注入）：
  ① 任务自适应强度（TAD）：探测任务类型（角色扮演/情感倾诉/知识决策），
     动态调整注入强度与通道权重 —— 情感任务强引导、知识任务弱引导保准确。
  ② 进度感知强度调度（PIS）：开头强（立基调）→ 中段稳 → 尾段渐弱（自然收尾），
     避免长输出过度注入。
  ③ 在线质量纠正（OQC）：逐 token 监控已生成文本，
     - AI 腔检测 → 动态抑制（满足"纠正改进能力"）
     - 情感缺失（已生成 N 字仍无情感词）→ 临时加强情感引导
     - 角色漂移（v_dyn 偏离 v_target 过大）→ 拉回角色锚点（角色扮演一致性）

多通道正交注入（复用前辈已验证组件）：
  - DMR 稠密乘性重加权（P5）：logits' = (1-α)·logits + α·log q_emo(v_eff)
  - KV 情感调制（P4）：K[p] *= (1+κ·g)，含 V 调制
  - 锚点 tanh 加性（P3）：logits += β·tanh(S@v_eff/T_anchor)
  - AI 腔三通道抑制 + 口语化（P2.5）
  - DSA 动态方向（P5）：v_eff = normalize((1-γ)·v_target + γ·v_dyn)

接口对齐 超融合解码器.生成()/注入偏置()，可直接接入 统一生成器。
"""
import math
import os
import re
import sys
from collections import Counter
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F

工作目录 = os.path.dirname(os.path.abspath(__file__))
if 工作目录 not in sys.path:
    sys.path.insert(0, 工作目录)

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器, 匹配角色基调
from 锚点解码器 import 计算熵, 计算重复率


def _定位最后一层(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[-1]
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h[-1]
    if hasattr(model, 'model') and hasattr(model.model, 'decoder') \
            and hasattr(model.model.decoder, 'layers'):
        return model.model.decoder.layers[-1]
    raise ValueError(f"无法定位模型 {type(model).__name__} 的最后一层")


def 经验熵(token列表):
    if not token列表:
        return 0.0
    n = len(token列表)
    c = Counter(token列表)
    return float(-sum((v / n) * math.log(v / n) for v in c.values()))


class 情感导演解码器:
    """P6 EDD：任务自适应强度 + 进度调度 + 在线纠正 + 多通道正交注入"""

    def __init__(
        self,
        model,
        tokenizer,
        锚点库: 锚点库,
        目标决策器: 目标决策器,
        # DMR
        开启DMR: bool = True,
        α基: float = 0.18,
        α倍率: float = 1.0,
        α上限: float = 0.7,
        T_emo: float = 0.5,
        # KV 调制
        开启KV调制: bool = True,
        开启V调制: bool = True,
        κ基: float = 0.20,
        κ上限: float = 0.6,
        κ_v基: float = 0.12,
        情感阈值: float = 0.05,
        调制层数: int = 4,
        # 锚点加性
        开启锚点偏置: bool = True,
        β基: float = 0.6,
        β上限: float = 1.5,
        T_anchor: float = 0.3,
        # DSA
        开启DSA: bool = True,
        γ: float = 0.3,
        # AI 腔抑制
        AI腔抑制强度: float = 2.0,
        口语化强度: float = 0.6,
        # 导演层
        任务自适应: bool = True,
        进度调度: bool = True,
        在线纠正: bool = True,
        情感缺失阈值: int = 16,
        角色漂移阈值: float = 0.5,   # cos 夹角阈值（1=同向）
        # 兜底 / 句停止
        退化窗口: int = 40,
        兜底阈值: float = 0.6,
        句分隔符: str = r"[。！？!?；;\n～…~]",
        密度基: float = 0.06,
        密度增益: float = 0.10,
        密度上限: float = 0.25,
        目标长度: int = 34,
        最短字数: int = 12,
        最大字数: int = 90,
        最长句数: int = 2,
        句子停止: bool = True,
        最小长度: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.锚点库 = 锚点库
        self.目标决策器 = 目标决策器

        self.开启DMR = bool(开启DMR)
        self.α基 = α基
        self.α倍率 = α倍率
        self.α上限 = α上限
        self.T_emo = T_emo
        self.开启KV调制 = bool(开启KV调制)
        self.开启V调制 = bool(开启V调制)
        self.κ基 = κ基
        self.κ上限 = κ上限
        self.κ_v基 = κ_v基
        self.情感阈值 = 情感阈值
        self.调制层数 = 调制层数
        self.开启锚点偏置 = bool(开启锚点偏置)
        self.β基 = β基
        self.β上限 = β上限
        self.T_anchor = T_anchor
        self.开启DSA = bool(开启DSA)
        self.γ = γ
        self.AI腔抑制强度 = AI腔抑制强度
        self.口语化强度 = 口语化强度
        self.任务自适应 = bool(任务自适应)
        self.进度调度 = bool(进度调度)
        self.在线纠正 = bool(在线纠正)
        self.情感缺失阈值 = 情感缺失阈值
        self.角色漂移阈值 = 角色漂移阈值
        self.退化窗口 = 退化窗口
        self.兜底阈值 = 兜底阈值
        self.句分隔符 = re.compile(句分隔符)
        self.密度基 = 密度基
        self.密度增益 = 密度增益
        self.密度上限 = 密度上限
        self.目标长度 = 目标长度
        self.最短字数 = 最短字数
        self.最大字数 = 最大字数
        self.最长句数 = 最长句数
        self.句子停止 = 句子停止
        self.最小长度 = 最小长度

        self.device = model.device
        self.vocab_size = int(model.config.vocab_size)

        # ── 锚点载体 ──
        self.锚点矩阵 = 锚点库.构建()
        self.打分表 = 锚点库.预计算打分表()
        self.K = self.锚点矩阵.shape[0]

        # ── DSA hook ──
        self._钩子列表: List = []
        self.当前hidden_state: Optional[torch.Tensor] = None
        if self.开启DSA:
            self._注册钩子()

        # ── AI 腔 / 口语化 token 表 ──
        self._AI腔token表: Dict[int, float] = {}
        self._首token表: Dict[int, float] = {}
        self._口语化token表: Dict[int, float] = {}
        self._正性模板token表: Dict[int, float] = {}
        self._正性首token表: Dict[int, float] = {}
        self._低唤起角色 = False
        self._用户正性 = False
        self._构建文本token表()
        self._构建正性模板token表()

        # ── 目标状态 ──
        self.v_target: Optional[np.ndarray] = None
        self.v_eff: Optional[torch.Tensor] = None
        self.密度目标 = self.密度基
        self.α当前 = self.α基
        self.κ当前 = 0.0
        self.κ_v当前 = 0.0
        self.β = self.β基
        self.任务类型 = "闲聊"
        self.任务强度 = 1.0

        # ── 句状态 / 兜底 / 纠正 ──
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._已生成token列表: List[int] = []
        self._兜底计数 = 0
        self.触发兜底次数 = 0
        self._已调制位置 = 0
        self._纠正触发 = {"AI腔": 0, "情感缺失": 0, "角色漂移": 0}
        self._句情感token数 = 0
        self._身份拦截中 = False

        # ── 情感 token 集 ──
        self._情感token集: Set[int] = self._构建情感token集()

        # ── 特殊/控制 token 屏蔽（v6.4 跨模型通用：Qwen3 <think>、工具调用等）──
        # 保留 eos/pad/bos 等自然终止 token，仅屏蔽思考块/工具控制 token
        self._屏蔽token集: Set[int] = set()
        try:
            _保留 = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                     self.tokenizer.bos_token_id}
            for tid in self.tokenizer.all_special_ids:
                if tid not in _保留:
                    self._屏蔽token集.add(tid)
            for 词 in ("<think>", "</think>", "<thought>", "<tool_call>",
                       "<tool_response>", "<|im_start|>", "<|im_end|>"):
                for tid in self.tokenizer.encode(词, add_special_tokens=False):
                    if tid not in _保留:
                        self._屏蔽token集.add(tid)
        except Exception as e:  # noqa: BLE001
            print(f"[情感导演解码器] 屏蔽 token 集构建失败：{e}")

    def _屏蔽特殊token(self, logits: torch.Tensor) -> torch.Tensor:
        """控制 token 一律 -inf（防止 Qwen3 思考块/工具调用污染人味输出）"""
        for tid in self._屏蔽token集:
            if tid < self.vocab_size:
                logits[0, tid] = float('-inf')
        return logits

    # ──────────────────────────────────────────────
    # DSA hook
    # ──────────────────────────────────────────────

    def _注册钩子(self) -> None:
        try:
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    hs = output[0][0, -1, :]
                else:
                    hs = output[0, -1, :]
                self.当前hidden_state = hs.detach()
            handle = _定位最后一层(self.model).register_forward_hook(hook)
            self._钩子列表.append(handle)
        except Exception as e:  # noqa: BLE001
            print(f"[情感导演解码器] DSA hook 注册失败，降级为静态方向：{e}")
            self.开启DSA = False

    def _移除钩子(self) -> None:
        for handle in self._钩子列表:
            handle.remove()
        self._钩子列表.clear()

    def __del__(self):
        self._移除钩子()

    # ──────────────────────────────────────────────
    # token 表构建
    # ──────────────────────────────────────────────

    AI腔词表 = {
        # 强抑制（AI 身份暴露）
        "AI": 3.0, "助手": 3.0, "人工智能": 3.0, "模型": 3.0, "智能": 2.5,
        # 中抑制（服务性套话）
        "作为": 2.5, "提供": 2.0, "帮助": 2.0, "用户": 2.0, "服务": 2.0,
        "抱歉": 2.5, "对不起": 2.5, "请告诉我": 2.0, "很高兴": 2.0,
        "请问": 1.5, "回答": 1.5, "信息": 1.5, "相关": 1.5, "方面": 1.5,
        "如果": 1.5, "需要": 1.5, "可以": 1.0, "能够": 1.5,
        # 弱抑制（正式书面语）
        "一些": 1.0, "以下": 1.5, "总之": 1.5, "以上": 1.5, "首先": 1.0,
        "其次": 1.0, "最后": 1.0, "根据": 1.5, "建议": 1.0, "内容": 1.0,
        # 短语感知补充（绕行词）
        "语言": 2.0, "文本": 1.5, "生成": 1.5, "随时": 1.5, "功能": 1.5,
        "设计": 1.5, "目标": 1.5, "主要": 1.0,
        # 身份暴露复合/替代词
        "作为一个": 2.5, "作为一名": 2.5, "机器人": 2.0, "程序": 2.0,
        "model": 2.5, "虚拟": 1.5, "语料": 1.5, "think": 2.0,
        # P6 补充
        "我没有情感": 4.0, "我专注于": 3.5, "准确的": 2.5, "有用的信息": 3.0,
        "我不能": 3.0, "我不具备": 3.5, "建议您": 2.5, "您需要": 2.0,
        "我的目的": 3.0, "语言模型": 3.0, "不便": 2.0, "欢迎": 1.5,
        "随时提供": 2.0, "如果有需要": 2.0, "如果您需要": 2.5,
        # P6.1 补充：书面语/分析腔/权威腔
        "我认为": 1.5, "我觉得": 1.0, "实际上": 1.5, "一般来说": 2.0,
        "通常情况下": 2.0, "这个问题": 1.5, "这个句子": 2.5, "基本结构": 2.5,
        "句型": 2.5, "用于": 2.0, "本质": 1.5, "事实上": 1.5,
        "您": 1.2, "您的": 1.5, "请问您": 2.0, "您好": 2.0,
        "无法": 2.0, "不确定": 1.5, "可能": 1.0, "如果": 1.5, "但是": 1.0,
    }
    # 首 token 黑名单（真人回复从不以这些开头）
    首token黑名单 = {
        "作为": 3.0, "我是": 3.0, "我的": 2.5, "抱歉": 3.0, "对不起": 3.0,
        "请": 2.0, "如果": 1.5, "根据": 1.5, "AI": 3.0, "助手": 3.0,
        "模型": 3.0, "语言": 2.0, "文本": 1.5, "很高兴": 2.0, "非常": 1.0,
        "Q": 3.0, "作为一个": 3.0, "作为一名": 3.0, "机器人": 2.0,
        "程序": 2.0, "model": 2.5, "智能": 2.0, "我无法": 3.0, "我不能": 3.0,
        "我需要": 2.0, "我的目的": 3.0, "实际上": 1.5, "这个问题": 2.0,
        "这个句子": 3.0, "一般来说": 2.0, "您好": 2.0, "我认为": 2.0,
        # v6.3：对话角色前缀回显（"A:/B:"）——首 token 抑制单字母 A-D
        "A": 2.5, "B": 2.5, "C": 2.5, "D": 2.5,
    }
    # 短语感知前缀（近文本正处这些之后 → 抑制 ×3）
    _短语前缀 = [
        "作为", "我是", "我的", "语言", "随时", "请", "提供", "帮助",
        "无法", "抱歉", "对不起", "AI", "文本", "目标", "功能", "设计",
        "智能", "生成", "这个句子", "基本结构", "句型", "用于", "这个",
        "我认为", "一般来说", "根据",
    ]
    # decode 级 AI 腔开头过滤（任何候选 token 以此开头 → -inf）
    _AI腔开头 = re.compile(
        r"^(作为|抱歉|对不起|不好意思|我是|我的|我很|我无法|我不能|如果|请|"
        r"根据|很高兴|AI|助手|模型|语言|文本|智能|Q|I\b|非常|很遗憾|温馨提示|"
        r"这个句子|这个问题|一般来说|实际上|<think>|<thought>|让我|我需要)"
    )
    # 身份暴露拦截（v6）："我是/作为…"刚成形时对身份名词直接 -inf
    身份打开器列表 = [
        "我是一个正在运行", "我是一个正在", "我是一个", "我是一台",
        "我是24小时", "我是基于", "我是", "作为一个", "作为",
        "我的功能", "我的目的", "我被设计", "我的存在",
        "我的出现", "我的使命", "我的作用", "我的回答",
        "我无法", "我不能", "我不具备", "我没有", "我没有情感", "我的任务是",
        "我的工作是", "我主要负责", "我只能", "我没有能力", "我的能力", "我认为我是",
        "我是一个模型", "我是一台", "我是一个程序", "我是电脑",
    ]
    身份名词列表 = [
        "AI", "人工智能", "助手", "模型", "语言", "系统", "程序", "机器人",
        "数字", "虚拟", "计算机", "预训练", "算法", "智能体", "语料",
        "自然语言", "大语言", "语言处理", "聊天机器人", "在线", "正在运行",
        "大型", "大规模", "人工", "运行", "24小时",
        "基于", "正在", "数据", "框架", "处理", "规模", "训练",
        "人类", "人们", "用户", "知识库", "数据库", "网络", "存在", "无意识",
        "编程", "逻辑", "服务", "推荐", "解答", "阿里巴巴", "物理",
        "机器", "学习", "神经", "Qwen", "GPT", "LLM", "指令", "问答",
        "信息", "软件", "硬件", "代码", "功能模块", "接口", "任务", "对话型",
    ]
    # 身份拦截打开后需保持"拦截模式"直到句号（v6.2 持久拦截）
    身份句界 = re.compile(r"[。！？!?；;\n～…~，,]")
    口语化词表 = [
        "啊", "呀", "啦", "哦", "呢", "吧", "嘛", "咯", "哟", "哇",
        "嘻嘻", "哈哈", "嘿嘿", "呜", "诶", "～",
    ]

    def _构建文本token表(self) -> None:
        for 词, 权重 in self.AI腔词表.items():
            if not 词:
                continue
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                self._AI腔token表[ids[0]] = 权重
        for 词, 权重 in self.首token黑名单.items():
            if not 词:
                continue
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                self._首token表[ids[0]] = 权重
        for 词 in self.口语化词表:
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                self._口语化token表[ids[0]] = 1.0

    # ── 角色扮演·低唤起角色：高唤起正性模板抑制（v6.5）──
    # EmoCharacter 缺陷：用户"今天遇到特别开心的事"时，低唤起角色（前辈/老师/兄长/毒舌/老人）
    # 也被劫持输出统一的"太好了！有什么开心的事？"模板 → 基调漂移、一致性下跌。
    # 仅当 任务类型==角色扮演 且 角色基调 valence<0.35（低唤起）时激活，高唤起角色不受影响。
    角色正性模板词 = {
        "太好了": 3.0, "太棒了": 3.0, "太高兴了": 3.0, "太令人高兴了": 3.0,
        "恭喜": 2.5, "恭喜恭喜": 3.0, "恭喜你": 3.0, "真棒": 2.5, "棒": 2.0,
        "有什么开心的事": 3.0, "什么惊喜": 2.5, "什么可开心的": 2.5,
        "什么开心的事": 2.5, "喜出望外": 3.5, "真的很高兴": 3.0, "听起来真的很高兴": 3.5,
        "我很高兴": 2.5, "高兴得": 2.5, "太让人高兴": 3.0, "超级开心": 3.0,
    }
    角色正性短语前缀 = ("太好了", "太棒了", "太高兴了", "太令人高兴了", "恭喜", "真棒",
                        "喜出望外", "真的很高兴", "听起来真的很高兴", "我很高兴",
                        "太让人高兴", "超级开心", "高兴得")
    # 首 token 堵死"太好了/太棒了/恭喜/喜出望外/哇/很高兴/开心/惊喜/好消息/真是太"开头（单 token 词）
    角色正性首token词 = {"太": 8.0, "恭喜": 8.0, "棒": 8.0, "喜": 8.0, "哇": 8.0,
                      "高兴": 5.0, "开心": 3.0, "惊喜": 5.0, "好消息": 5.0,
                      "真是太": 8.0, "太好了": 8.0, "太棒了": 8.0, "太高兴了": 8.0}

    def _构建正性模板token表(self) -> None:
        for 词, 权重 in self.角色正性模板词.items():
            for tid in self.tokenizer.encode(词, add_special_tokens=False):
                self._正性模板token表[tid] = max(self._正性模板token表.get(tid, 0.0), 权重)
        for 词, 权重 in self.角色正性首token词.items():
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                self._正性首token表[ids[0]] = 权重

    # ──────────────────────────────────────────────
    # 任务自适应强度（TAD）
    # ──────────────────────────────────────────────

    def _探测任务(self, 用户文本: str) -> str:
        """按用户输入探测任务类型 → 角色扮演/情感倾诉/知识决策/闲聊/格式任务"""
        if not 用户文本:
            return "闲聊"
        # 结构化输出任务（JSON/决策格式）→ 格式任务：几乎不注入，保格式
        格式词 = ["JSON", "decision_choice", "final_decision", "Output strictly",
                 "rubric", "```", "emoji", "dialogue_history"]
        for w in 格式词:
            if w in 用户文本:
                return "格式任务"
        角色词 = ["你是", "扮演", "角色", "你是我的", "像", "帮我扮演"]
        决策词 = ["怎么办", "如何", "什么原因", "为什么", "怎么解决", "建议",
                 "分析", "解释", "方案", "步骤", "区别", "优缺点"]
        for w in 角色词:
            if w in 用户文本:
                return "角色扮演"
        # 情感词密度
        try:
            感知器 = self.目标决策器.感知器
            情感数 = sum(1 for 词 in getattr(感知器, "_正面词", set())
                        | getattr(感知器, "_负面词", set()) if 词 in 用户文本)
        except Exception:
            情感数 = sum(1 for w in
                        ["累", "难过", "开心", "烦", "焦虑", "害怕", "孤独",
                         "伤心", "委屈", "失望", "想", "爱", "哭", "痛"] if w in 用户文本)
        if 情感数 >= 1:
            return "情感倾诉"
        for w in 决策词:
            if w in 用户文本:
                return "知识决策"
        return "闲聊"

    def _任务强度(self) -> float:
        if not self.任务自适应:
            return 1.0
        if self.任务类型 == "角色扮演":
            return 1.25   # 角色扮演需更强情感锚定
        if self.任务类型 == "情感倾诉":
            return 0.9     # 倾诉任务温和引导（减噪：多通道注入易干扰细腻表达）
        if self.任务类型 == "知识决策":
            return 0.55   # 知识任务弱引导保准确
        if self.任务类型 == "格式任务":
            return 0.2    # 结构化输出任务几乎不注入，保 JSON 格式
        return 1.0

    # ──────────────────────────────────────────────
    # 进度感知强度调度（PIS）
    # ──────────────────────────────────────────────

    def _进度系数(self, 步: int) -> float:
        if not self.进度调度:
            return 1.0
        if 步 < 8:
            return 1.2      # 开头：立情感基调
        if 步 <= 40:
            return 1.0      # 中段：稳定
        return 0.55         # 尾段：渐弱自然收尾

    # ──────────────────────────────────────────────
    # 动态方向（DSA）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def _动态方向(self, v_target_np: np.ndarray) -> torch.Tensor:
        if self.当前hidden_state is None:
            return torch.as_tensor(v_target_np, dtype=torch.float32, device=self.device)
        h = self.当前hidden_state.float()
        hn = h / (h.norm() + 1e-9)
        A = self.锚点矩阵.float() / (self.锚点矩阵.float().norm(dim=-1, keepdim=True) + 1e-9)
        v_dyn = hn @ A.T
        v_t = torch.as_tensor(v_target_np, dtype=torch.float32, device=self.device)
        v_eff = (1.0 - self.γ) * v_t + self.γ * v_dyn
        return (v_eff / (v_eff.norm() + 1e-9)).float()

    def _角色漂移检测(self) -> bool:
        """v_dyn 与 v_target 的余弦低于阈值 → 认为生成方向偏离角色目标"""
        if self.当前hidden_state is None or self.v_target is None:
            return False
        try:
            h = self.当前hidden_state.float()
            hn = h / (h.norm() + 1e-9)
            A = self.锚点矩阵.float() / (self.锚点矩阵.float().norm(dim=-1, keepdim=True) + 1e-9)
            v_dyn = hn @ A.T
            v_t = torch.as_tensor(self.v_target, dtype=torch.float32, device=self.device)
            v_t = v_t / (v_t.norm() + 1e-9)
            cos = float((v_dyn / (v_dyn.norm() + 1e-9)) @ v_t)
            return cos < self.角色漂移阈值
        except Exception:
            return False

    # ──────────────────────────────────────────────
    # 在线质量纠正（OQC）
    # ──────────────────────────────────────────────

    def _AI腔抑制(self, logits: torch.Tensor) -> torch.Tensor:
        """三通道 AI 腔抑制（移植潮汐 v4 验证方案）：
        (1) 首 token 黑名单 ×6 + decode 级 top-300 过滤 -inf
        (2) 短语感知：近文本正处 AI 腔短语前缀后 → 抑制 ×3
        (3) 基础抑制：AI 腔 token 施加 -强度×权重 偏置
        """
        强度 = self.AI腔抑制强度 * self.任务强度
        # 角色扮演任务 AI 腔风险最高 → 额外 ×1.5 强压
        if self.任务类型 == "角色扮演":
            强度 *= 1.5
        if 强度 <= 0 or not self._AI腔token表:
            return logits

        # (1) 首 token 黑名单 + decode 级过滤
        if not self._已生成token列表:
            for tid, 权重 in self._首token表.items():
                if tid < self.vocab_size:
                    logits[0, tid] -= 强度 * 6.0 * 权重
            topk = torch.topk(logits, k=min(300, self.vocab_size), dim=-1)
            for v, tid in zip(topk.values[0], topk.indices[0]):
                txt = self.tokenizer.decode([tid.item()], skip_special_tokens=True)
                if txt and self._AI腔开头.match(txt):
                    logits[0, tid] = float('-inf')
            return logits

        # (2) 短语感知：解码最近 3 个 token，若构成 AI 腔短语前缀 → 抑制 ×3
        倍数 = 1.0
        近文 = "".join(
            self.tokenizer.decode([t], skip_special_tokens=True)
            for t in self._已生成token列表[-3:]
        )
        for 前缀 in self._短语前缀:
            idx = 近文.rfind(前缀)
            if idx != -1 and len(近文) - idx <= 4:
                倍数 = 3.0
                break

        # (3) 基础抑制
        for tid, 权重 in self._AI腔token表.items():
            if tid < self.vocab_size:
                logits[0, tid] -= 强度 * 权重 * 倍数
        return logits

    def _身份暴露拦截(self, logits: torch.Tensor) -> torch.Tensor:
        """v6.2 持久身份拦截：
        - 开启器出现在**全文任意位置**（不限于尾部）即进入拦截模式
        - 拦截模式持续到句子边界（。！？；等），期间身份名词一律 -inf（含 decode 级过滤）
        - 解决 v6.1"我是由机器学习和神经..."（窗口太短漏网）的绕行问题
        """
        if not self._生成文本:
            return logits
        全文 = self._生成文本
        if not self._身份拦截中:
            # 最近一句（句界之后）出现开启器 → 开启拦截模式
            if self.身份句界.search(全文):
                最近句 = self.身份句界.split(全文)[-1]
            else:
                最近句 = 全文
            if any(o in 最近句 for o in self.身份打开器列表):
                self._身份拦截中 = True
            else:
                return logits
        else:
            # 已拦截：若遇句界则关闭（此句已收尾）
            if self.身份句界.search(全文[-1:]):
                self._身份拦截中 = False
                return logits
        # 拦截模式：身份名词 token -inf + decode 级过滤
        for 词 in self.身份名词列表:
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if not ids:
                continue
            for tid in ids:
                if tid < self.vocab_size:
                    logits[0, tid] = float('-inf')
        # decode 级：候选 token 解码后命中身份名词/身份动词 → -inf
        topk = torch.topk(logits, k=min(200, self.vocab_size), dim=-1)
        for _v, tid in zip(topk.values[0], topk.indices[0]):
            候选 = self.tokenizer.decode([tid.item()], skip_special_tokens=True)
            if 候选 and any(n in 候选 for n in self.身份名词列表):
                logits[0, tid] = float('-inf')
        return logits

    def _口语化引导(self, logits: torch.Tensor) -> torch.Tensor:
        if self.口语化强度 <= 0 or not self._口语化token表:
            return logits
        for tid in self._口语化token表:
            if tid < self.vocab_size:
                logits[0, tid] += self.口语化强度 * self.任务强度
        return logits

    def _角色正性模板抑制(self, logits: torch.Tensor) -> torch.Tensor:
        """v6.5→v6.7：角色扮演 + 低唤起角色 + 用户正性情感 → 抑制高唤起正性模板。

        修复 EmoCharacter 缺陷：用户表达开心时，低唤起角色（前辈/老师/兄长/毒舌/老人）
        被"太好了！有什么开心的事？"等热情模板劫持 → 跨轮基调漂移。
        v6.7 加入用户正性条件 + 替代模板（喜出望外/真的很高兴/听起来真的很高兴）抑制。
        高唤起角色（同桌/小女儿/女友/演员）与用户非正性场景不受影响。
        """
        if not (self.任务类型 == "角色扮演" and self._低唤起角色 and self._用户正性):
            return logits
        if not self._正性模板token表:
            return logits
        强度 = 2.0 * self.任务强度
        # ① 首 token：堵死"太好了/太棒了/恭喜/喜出望外/哇"开头
        if not self._已生成token列表:
            for tid, 权重 in self._正性首token表.items():
                if tid < self.vocab_size:
                    logits[0, tid] -= 强度 * 权重
            # decode 级前缀过滤：候选 token 解码文本含"太/恭/喜/棒/哇/高兴/开心/惊喜/好消息"（含"真是太"等整体 token）→ -inf
            _开头词 = ("太", "恭", "喜", "棒", "哇", "高兴", "开心", "惊喜", "好消息")
            _topk = torch.topk(logits, k=min(300, self.vocab_size), dim=-1)
            for _v, _tid in zip(_topk.values[0], _topk.indices[0]):
                _txt = self.tokenizer.decode([_tid.item()], skip_special_tokens=True)
                if _txt and any(_w in _txt for _w in _开头词):
                    logits[0, _tid] = float('-inf')
            return logits
        # ② 持续抑制：低唤起+用户正性时，"太好了/太棒了/恭喜"等模板词全流程压低
        #    （防"真是太好了！"式绕过——首 token 不是模板词但后续拼接出现）
        #    "太好了"在开心场景先验 logits 极高（可>15），需强压（-30 级）
        for tid, 权重 in self._正性首token表.items():
            if tid < self.vocab_size:
                logits[0, tid] -= 强度 * 权重 * 1.5
        # ③ 近文短语检测：正在输出"太好了/太棒了/恭喜…" → 加倍抑制同组词
        近文 = "".join(
            self.tokenizer.decode([t], skip_special_tokens=True)
            for t in self._已生成token列表[-6:]
        )
        if any(前缀 in 近文 for 前缀 in self.角色正性短语前缀):
            for tid, 权重 in self._正性模板token表.items():
                if tid < self.vocab_size:
                    logits[0, tid] -= 强度 * 权重
        return logits

    def _情感缺失检测(self) -> bool:
        """已生成文本超过阈值仍无情感词 → 情感缺失，需加强引导"""
        if len(self._生成文本) < self.情感缺失阈值:
            return False
        if self._句情感token数 > 0:
            return False
        # 全文级检查：整个生成文本无情感词
        try:
            感知器 = self.目标决策器.感知器
            词集 = getattr(感知器, "_正面词", set()) | getattr(感知器, "_负面词", set())
            return not any(w in self._生成文本 for w in 词集)
        except Exception:
            return not any(w in self._生成文本 for w in
                          ["开心", "难过", "爱", "累", "想", "好", "谢谢", "心疼", "抱"])

    # ──────────────────────────────────────────────
    # KV 情感调制（P4 复用 + V 调制增强）
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def _调制KV缓存(self, past_key_values, token_ids: torch.Tensor):
        if past_key_values is None or not self.开启KV调制 or self.v_target is None:
            return past_key_values
        if self.任务类型 == "格式任务":
            return past_key_values  # 结构化输出任务不调 KV，保格式
        if self.κ当前 <= 0:
            return past_key_values
        # KV 调制可能在 注入偏置 之前被调用 → 确保 v_eff 已计算
        if self.v_eff is None:
            v_t = np.asarray(self.v_target, dtype=np.float32)
            self.v_eff = self._动态方向(v_t) if self.开启DSA else torch.as_tensor(
                v_t, dtype=torch.float32, device=self.device)
        ids = token_ids[0]
        T = len(ids)
        开始 = self._已调制位置
        if T <= 开始:
            return past_key_values
        v = self.v_eff.to(self.打分表.dtype)
        S_ids = self.打分表[ids[开始:]]
        scores = (S_ids @ v).float()
        g = torch.clamp(scores, 0.0, 1.0)
        掩码 = g > self.情感阈值
        self._已调制位置 = T
        if not 掩码.any():
            return past_key_values
        尺度k = torch.where(掩码, 1.0 + self.κ当前 * g, torch.ones_like(g))
        sk = 尺度k.reshape(1, 1, -1, 1)
        sv = None
        if self.开启V调制 and self.κ_v当前 > 0:
            尺度v = torch.where(掩码, 1.0 + self.κ_v当前 * g, torch.ones_like(g))
            sv = 尺度v.reshape(1, 1, -1, 1)
        层条目 = list(past_key_values)
        n = len(层条目)
        起始 = max(0, n - self.调制层数)
        for l in range(起始, n):
            k = 层条目[l][0]
            if k is None:
                continue
            k[:, :, 开始:, :] *= sk.to(k.dtype)
            if sv is not None and 层条目[l][1] is not None:
                层条目[l][1][:, :, 开始:, :] *= sv.to(层条目[l][1].dtype)
        return past_key_values

    # ──────────────────────────────────────────────
    # 主注入（多通道正交，动态强度）
    # ──────────────────────────────────────────────

    def _密度系数(self) -> float:
        if self._句情感词数 <= 0:
            return 1.0
        L = max(len(self._当前句文本), 1)
        if self._句情感词数 / L > self.密度目标:
            return 0.3
        return 1.0

    @torch.no_grad()
    def 注入偏置(self, logits: torch.Tensor, 步: int) -> torch.Tensor:
        if self.v_target is None:
            return logits
        v_t = np.asarray(self.v_target, dtype=np.float32)
        self.v_eff = self._动态方向(v_t) if self.开启DSA else torch.as_tensor(
            v_t, dtype=torch.float32, device=self.device)

        强度 = self.任务强度 * self._进度系数(步)
        # 情感缺失纠正 → 强度提升
        if self.在线纠正 and self._情感缺失检测():
            self._纠正触发["情感缺失"] += 1
            强度 *= 1.5

        # ① DMR 稠密乘性
        if self.开启DMR and self.α当前 > 0:
            v = self.v_eff.to(self.打分表.dtype)
            a = self.打分表 @ v
            log_q = F.log_softmax(a / self.T_emo, dim=-1)
            α_dyn = min(self.α上限, self.α当前 * 强度)
            logits = (1.0 - α_dyn) * logits + α_dyn * log_q.unsqueeze(0)

        # ② 锚点 tanh 加性
        if self.开启锚点偏置 and self.β > 0:
            密度系数 = self._密度系数()
            v = self.v_eff.to(self.打分表.dtype)
            a = self.打分表 @ v
            β_dyn = min(self.β上限, self.β * 强度) * 密度系数
            logits = logits + β_dyn * torch.tanh(a / self.T_anchor).unsqueeze(0)

        # ③ AI 腔抑制 + 角色正性模板抑制 + 口语化（在线纠正）
        if self.在线纠正:
            logits = self._AI腔抑制(logits)
        if self.任务类型 == "角色扮演":
            logits = self._角色正性模板抑制(logits)
        if self.口语化强度 > 0:
            logits = self._口语化引导(logits)
        return logits

    # ──────────────────────────────────────────────
    # 目标更新（任务探测 + 决策器自适应）
    # ──────────────────────────────────────────────

    def 更新目标(self, 用户文本: str = "", 思考链文本: str = "", 指令: str = "",
                角色=None, 轮次: int = 0):
        try:
            self.任务类型 = self._探测任务(用户文本)
            # 外部 system 角色扮演（指令标记）→ 全程保持角色扮演锚定，防止后续轮次漂移
            if 指令 and "角色" in 指令:
                self.任务类型 = "角色扮演"
            self.任务强度 = self._任务强度()
            # 长任务（对话历史类，如 HeartBench）允许 3 句细腻表达；普通聊天保持 2 句
            if "对话历史" in 用户文本 or "下文回应" in 用户文本:
                self.最长句数 = 3
            else:
                self.最长句数 = 2
            # 格式任务（JSON 输出）禁用句子停止，避免截断结构化输出
            self.句子停止 = bool(self.句子停止) and self.任务类型 != "格式任务"
            # v6.5：低唤起角色（基调 valence<0.35）→ 激活正性模板抑制
            _基调 = 匹配角色基调(角色) if 角色 else None
            self._低唤起角色 = bool(_基调 and _基调[1][0] < 0.35)
            目标 = self.目标决策器.计算目标(
                用户当前=用户文本 or None, 思考链文本=思考链文本, 指令=指令,
                角色=角色, 轮次=轮次, 角色权重=0.7)
            self.v_target = np.asarray(目标.v_target, dtype=np.float32)
            # v6.7：低唤起角色 + 用户正性情感 → 激活高唤起正性模板抑制（防"喜出望外/真的太高兴"替代模板）
            self._用户正性 = False
            try:
                _v = (目标.决策日志 or {}).get("用户当前VAD", {}) or {}
                self._用户正性 = float(_v.get("valence", 0.0) or 0.0) > 0.1
            except Exception:
                self._用户正性 = False
            self.密度目标 = 目标.情感词密度目标
            日志 = 目标.决策日志 or {}
            Δ唤醒 = float(日志.get("Δ唤醒", 0.0) or 0.0)
            活跃度 = float(日志.get("活跃度", 0.0) or 0.0)
            目标强度 = float(日志.get("目标强度", 0.0) or 0.0)
            潮汐α = float(日志.get("潮汐α", self.α基) or self.α基)
            # α 自适应 + 任务强度
            self.α当前 = min(self.α上限, max(0.0, 潮汐α * self.α倍率) * (0.7 + 0.3 * self.任务强度))
            # κ 自适应
            κ = self.κ基 * (1.0 + Δ唤醒) * min(1.0, max(活跃度, 目标强度) * 2.5)
            self.κ当前 = min(self.κ上限, max(0.0, κ))
            self.κ_v当前 = min(self.κ上限, self.κ_v基 * (1.0 + Δ唤醒))
            # β
            self.β = min(self.β上限, self.β基 * (0.7 + 0.3 * self.任务强度))
            return 目标
        except Exception as e:  # noqa: BLE001
            print(f"[情感导演解码器] 目标计算失败：{e}")
            self.v_target = None
            return None

    # ──────────────────────────────────────────────
    # 生成循环
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def 生成(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        logits_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        tokenizer=None,
        轮次回调: Optional[Callable[[int, object], None]] = None,
        用户文本: str = "",
        思考链文本: str = "",
        指令: str = "",
        角色=None,
        轮次: int = 0,
    ):
        temperature = 1.0 if temperature is None else temperature
        top_p = 0.9 if top_p is None else top_p
        top_k = 50 if top_k is None else top_k
        repetition_penalty = 1.05 if repetition_penalty is None else repetition_penalty
        tokenizer = tokenizer or self.tokenizer
        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id

        self.更新目标(用户文本, 思考链文本, 指令, 角色=角色, 轮次=轮次)

        past_key_values = None
        已生成 = input_ids.clone()
        已生成token集合: Set[int] = set()
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._已生成token列表 = []
        self._兜底计数 = 0
        self.触发兜底次数 = 0
        self._已调制位置 = 0
        self._纠正触发 = {"AI腔": 0, "情感缺失": 0, "角色漂移": 0}
        self._句情感token数 = 0
        self._身份拦截中 = False
        熵列表: List[float] = []

        for 步 in range(max_new_tokens):
            模型输入 = 已生成[:, -1:] if past_key_values is not None else 已生成
            outputs = self.model(模型输入, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # KV 调制（P4）：注意力空间
            past_key_values = self._调制KV缓存(past_key_values, 已生成)

            if repetition_penalty != 1.0:
                for tid in 已生成token集合:
                    logits[0, tid] /= repetition_penalty

            # 角色漂移纠正
            if self.在线纠正 and self.任务类型 == "角色扮演" and self._角色漂移检测():
                self._纠正触发["角色漂移"] += 1
                v = self.v_eff.to(self.打分表.dtype)
                a = self.打分表 @ v
                logits = logits + min(self.β上限, self.β * 1.5) * torch.tanh(a / self.T_anchor).unsqueeze(0)

            # 主注入
            logits = self.注入偏置(logits, 步)
            # 身份暴露硬拦截（在线纠正）
            if self.在线纠正:
                logits = self._身份暴露拦截(logits)
            # 控制 token 屏蔽（<think>/工具调用等，跨模型）
            logits = self._屏蔽特殊token(logits)

            if logits_callback is not None:
                logits_callback(步, logits)

            熵列表.append(计算熵(logits))

            if self.最小长度 > 0 and self._生成文本 and len(self._生成文本) < self.最小长度:
                if eos_token_id is not None and eos_token_id < self.vocab_size:
                    logits[0, eos_token_id] = float('-inf')

            logits = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, stable=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < top_k_values[:, -1].unsqueeze(-1)] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            下一个token = torch.multinomial(probs, num_samples=1)

            已生成 = torch.cat([已生成, 下一个token], dim=-1)
            已生成token集合.add(下一个token.item())
            self._已生成token列表.append(下一个token.item())

            self._更新句状态(下一个token.item())
            if len(self._已生成token列表) >= 8 and len(self._已生成token列表) % self.退化窗口 == 0:
                self._兜底监测()

            if 轮次回调 is not None:
                轮次回调(步, self)
            if 下一个token.item() == eos_token_id:
                break
            if self.句子停止 and self._句子停止():
                break

        统计 = {
            "平均熵": round(sum(熵列表) / len(熵列表), 4) if 熵列表 else 0.0,
            "重复率": 计算重复率(self._已生成token列表),
            "情感命中率": self._情感命中率(self._已生成token列表),
            "α": round(self.α当前, 4),
            "κ": round(self.κ当前, 4),
            "β": round(self.β or 0.0, 4),
            "任务类型": self.任务类型,
            "任务强度": round(self.任务强度, 4),
            "纠正触发": dict(self._纠正触发),
            "触发兜底次数": self.触发兜底次数,
        }
        return 已生成, 统计

    # ──────────────────────────────────────────────
    # 句状态 / 兜底 / 指标
    # ──────────────────────────────────────────────

    def _更新句状态(self, token_id: int) -> None:
        文本 = self.tokenizer.decode([token_id], skip_special_tokens=True)
        if not 文本:
            return
        if token_id in self._情感token集:
            self._句情感词数 += 1
            self._句情感token数 += 1
        self._生成文本 += 文本
        if self.句分隔符.search(文本):
            self._当前句文本 = ""
            self._句情感词数 = 0
            self._句子数 += 1
            self._兜底监测()
        else:
            self._当前句文本 += 文本

    def _兜底监测(self) -> None:
        最近 = self._已生成token列表[-self.退化窗口:]
        if len(最近) < 8:
            return
        重复率 = 1.0 - len(set(tuple(最近[i:i + 2]) for i in range(len(最近) - 1))) / max(len(最近) - 1, 1)
        熵 = 经验熵(最近)
        if 重复率 > self.兜底阈值 or 熵 < 0.6:
            self.触发兜底次数 += 1
            self._兜底计数 += 1
            if self._兜底计数 >= 3:
                self.α当前 = 0.0
                self.β = 0.0
                self.κ当前 = 0.0
            else:
                self.α当前 *= 0.5
                self.β *= 0.5
                self.κ当前 *= 0.5
        else:
            self._兜底计数 = 0

    def _句子停止(self) -> bool:
        if not self._生成文本:
            return False
        if len(self._生成文本) >= self.最短字数 and self._句子数 >= self.最长句数:
            return True
        if self._句子数 >= self.最长句数 + 1:
            return True
        if len(self._生成文本) >= self.最大字数:
            return True
        return False

    def _构建情感token集(self) -> Set[int]:
        集: Set[int] = set()
        for 维, 词列表 in self.锚点库.词集.items():
            for 词 in 词列表:
                ids = self.tokenizer.encode(词, add_special_tokens=False)
                if len(ids) == 1:
                    集.add(ids[0])
        try:
            感知器 = self.目标决策器.感知器
            for 词 in getattr(感知器, "_正面词", set()) | getattr(感知器, "_负面词", set()):
                ids = self.tokenizer.encode(词, add_special_tokens=False)
                if len(ids) == 1:
                    集.add(ids[0])
        except Exception:  # noqa: BLE001
            pass
        return 集

    def _情感命中率(self, token列表: List[int]) -> float:
        if not token列表:
            return 0.0
        命中 = sum(1 for t in token列表 if t in self._情感token集)
        return round(命中 / len(token列表), 4)

    def 重置(self) -> None:
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._已生成token列表 = []
        self._兜底计数 = 0
        self.触发兜底次数 = 0
        self.v_target = None
        self.β = self.β基
        self._已调制位置 = 0

    def __repr__(self):
        return (f"情感导演解码器(EDD 任务自适应={self.任务自适应} 进度调度={self.进度调度} "
                f"在线纠正={self.在线纠正} DMR(α={self.α基}) KV(κ={self.κ基},V={self.开启V调制}) "
                f"锚点(β={self.β基}) AI腔抑制={self.AI腔抑制强度})")
