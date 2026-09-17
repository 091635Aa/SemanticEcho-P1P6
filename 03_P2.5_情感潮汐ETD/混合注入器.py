# -*- coding: utf-8 -*-
"""
混合注入器：语义回响（表示空间） × 潮汐解码（概率空间）
===========================================================
融合第三套架构（潮汐 ETD）与架构一（语义回响）的推理期增强：
- 回响：hook 捕获 hidden_state → 回响池质心 → `logits += 质心@投影×λ`
  （表示空间持续注入情感底色，提升细腻度/熵）
- 潮汐：极性定向概率引导 `k = α × 引导倍率`（概率空间控制情感方向与密度）
- 叠加：两者都作用在 logits 上；回响给"情感底色"，潮汐给"方向+密度控制"
- 互补性：
  · 回响熵提升（+40~45%）弥补潮汐在"人味儿/自由度"上的短板
  · 潮汐极性定向抑制回响的"无方向堆情感词"问题
  · 潮汐密度控制防止两者叠加后情感词爆炸

不修改任何现有源码：继承 回响注入器（复用钩子/投影/池），
从 潮汐解码器 提取极性 token 表与引导逻辑。
"""
import math
import os
import re
import sys
import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Set

# 语义回响工程根（回响注入器所在）
回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)
本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from semantic_echo.采样处理器 import 回响注入器

from 潮汐感知器 import 潮汐感知器
from 潮汐决策器 import 潮汐决策器, 潮汐目标


class 混合注入器(回响注入器):
    """回响 × 潮汐 混合：表示空间情感底色 + 概率空间方向密度控制"""

    def __init__(
        self,
        model,
        echo_pool,
        tokenizer,
        感知器: 潮汐感知器,
        决策器: 潮汐决策器,
        lambda_strength: float = 0.08,
        引导倍率: float = 12.0,
        uncertainty_threshold: float = 0.01,
        projection_seed: int = 42,
        last_n_layers: int = 4,
        情感过滤器实例=None,
        密度基: float = 0.06,
        密度增益: float = 0.10,
        密度上限: float = 0.25,
        句分隔符: str = r"[。！？!?；;\n～…~]",
        退化窗口: int = 40,
        兜底阈值: float = 0.6,
        AI腔抑制强度: float = 2.0,
        口语化强度: float = 1.0,
        目标长度: int = 34,
        身份拦截: bool = True,
        句子停止: bool = True,
        最长句数: int = 2,
        最短字数: int = 12,
        最大字数: int = 90,
        最小长度: int = 0,
    ):
        super().__init__(
            model, echo_pool, lambda_strength=lambda_strength,
            uncertainty_threshold=uncertainty_threshold,
            projection_seed=projection_seed, last_n_layers=last_n_layers,
            情感过滤器实例=情感过滤器实例,
        )
        self.tokenizer = tokenizer
        self.感知器 = 感知器
        self.决策器 = 决策器
        self.引导倍率 = 引导倍率
        self.AI腔抑制强度 = AI腔抑制强度
        self.口语化强度 = 口语化强度
        self.目标长度 = 目标长度
        self.身份拦截 = 身份拦截
        self.句子停止 = 句子停止
        self.最长句数 = 最长句数
        self.最短字数 = 最短字数
        self.最大字数 = 最大字数
        self.最小长度 = 最小长度
        self.密度基 = 密度基
        self.密度增益 = 密度增益
        self.密度上限 = 密度上限
        self.句分隔符 = re.compile(句分隔符)
        self.退化窗口 = 退化窗口
        self.兜底阈值 = 兜底阈值

        # 极性情感 token 表（复用潮汐逻辑）
        self._情感token表: Dict[int, float] = self._构建情感token表()
        # AI 腔 token 表（v3 双通道）
        self._AI腔token表: Dict[int, float] = self._构建AI腔token表()
        # 首 token 黑名单表（v4）
        self._首token表: Dict[int, float] = self._构建首token表()
        # 口语化 token 表（v5 人味儿引导）
        self._口语化token表: Dict[int, float] = self._构建口语化token表()
        # 句尾标点表（v5 长度收尾）
        self._句尾标点表: Dict[int, float] = self._构建句尾标点表()

        # 状态
        self.当前目标: Optional[潮汐目标] = None
        self.当前α = 0.0
        self.当前密度目标 = self.密度基
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0
        self._兜底计数 = 0
        self._已生成token列表: List[int] = []
        self.统计 = {"情感token命中": 0, "总token": 0, "触发兜底": 0}

    def _初始化投影(self, seed: int) -> None:
        """重写：GPU 直分配（父类 CPU 分配对大词表 OOM）"""
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        scale = math.sqrt(2.0 / self.hidden_dim)
        self.投影矩阵 = torch.randn(
            self.hidden_dim, self.vocab_size,
            generator=rng, dtype=torch.float32, device=self.device,
        ) * scale
        self.投影矩阵.requires_grad_(False)

    def _构建情感token表(self) -> Dict[int, float]:
        """极性情感 token 表：+1 正面 / -1 负面（只登记单 token 情感词）"""
        正面词 = set(self.感知器._正面词)
        负面词 = set(self.感知器._负面词)
        表: Dict[int, float] = {}
        for 词 in 正面词:
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                表[ids[0]] = 1.0
        for 词 in 负面词:
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                表.setdefault(ids[0], -1.0)
        return 表

    # ──────────────────────────────────────────────
    # AI 腔 token 表（v3 双通道）
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
        # v4 新增：短语感知补充
        "语言": 2.0, "文本": 1.5, "生成": 1.5, "随时": 1.5, "功能": 1.5,
        "设计": 1.5, "目标": 1.5, "主要": 1.0,
        # v4.2 身份暴露复合/替代词
        "作为一个": 2.5, "作为一名": 2.5, "机器人": 2.0, "程序": 2.0,
        "model": 2.5, "虚拟": 1.5, "语料": 1.5,
        # v4.3 跨模型：Qwen3 的 <think> 思考标记
        "<think>": 4.0, "think": 2.0,
    }

    # v4 首 token 黑名单：真人回复从不以 AI 身份词/服务套话开头
    首token黑名单 = {
        "作为": 3.0, "我是": 3.0, "我的": 2.5, "抱歉": 3.0, "对不起": 3.0,
        "请": 2.0, "如果": 1.5, "根据": 1.5, "AI": 3.0, "助手": 3.0,
        "模型": 3.0, "语言": 2.0, "文本": 1.5, "很高兴": 2.0, "非常": 1.0,
        "Q": 3.0, "作为一个": 3.0, "作为一名": 3.0, "机器人": 2.0,
        "程序": 2.0, "model": 2.5, "<think>": 4.0,
    }

    # v4 短语感知前缀
    _短语前缀 = [
        "作为", "我是", "我的", "语言", "随时", "请", "提供", "帮助",
        "无法", "抱歉", "对不起", "AI", "文本", "目标", "功能", "设计",
        "智能", "生成",
    ]

    # v4.1 首 token decode 级过滤
    _AI腔开头 = re.compile(
        r"^(作为|抱歉|对不起|不好意思|我是|我的|我很|我无法|我不能|如果|请|"
        r"根据|很高兴|AI|助手|模型|语言|文本|智能|Q|I\b|非常|很遗憾)"
    )

    def _构建AI腔token表(self) -> Dict[int, float]:
        表: Dict[int, float] = {}
        for 词, 权重 in self.AI腔词表.items():
            if not 词:
                continue
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                表[ids[0]] = 权重
        return 表

    def _构建首token表(self) -> Dict[int, float]:
        表: Dict[int, float] = {}
        for 词, 权重 in self.首token黑名单.items():
            if not 词:
                continue
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                表[ids[0]] = 权重
        return 表

    def _AI腔抑制(self, logits: torch.Tensor) -> torch.Tensor:
        """
        v4 三通道 AI 腔抑制（与 潮汐解码器 同款）：
        (1) 首 token 黑名单（×6 强压）
        (2) 短语感知（前缀后抑制 ×3）
        (3) 基础抑制
        """
        if self.AI腔抑制强度 <= 0 or not self._AI腔token表:
            return logits
        强度 = self.AI腔抑制强度

        # (1) 首 token 黑名单
        if not self._已生成token列表:
            for tid, 权重 in self._首token表.items():
                if tid < self.vocab_size:
                    logits[0, tid] -= 强度 * 6.0 * 权重
            # v4.1 decode 级过滤
            topk = torch.topk(logits, k=min(300, self.vocab_size), dim=-1)
            for v, tid in zip(topk.values[0], topk.indices[0]):
                txt = self.tokenizer.decode([tid.item()], skip_special_tokens=True)
                if txt and self._AI腔开头.match(txt):
                    logits[0, tid] = float('-inf')
            return logits

        # (2) 短语感知
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

    # ──────────────────────────────────────────────
    # v6 身份暴露硬拦截（decode 级，与 潮汐解码器 同款）
    # ──────────────────────────────────────────────
    身份打开器列表 = [
        "我是一个正在运行", "我是一个正在", "我是一个", "我是一台",
        "我是24小时", "我是基于", "我是", "作为一个", "作为",
        "我的功能", "我的目的", "我被设计", "我的存在",
        # v6.3：更多身份开启器
        "我的出现", "我的使命", "我的作用", "我的回答",
    ]
    身份名词列表 = [
        "AI", "人工智能", "助手", "模型", "语言", "系统", "程序", "机器人",
        "数字", "虚拟", "计算机", "预训练", "算法", "智能体", "语料",
        "自然语言", "大语言", "语言处理", "聊天机器人", "在线", "正在运行",
        "大型", "大规模", "大規模", "人工", "运行", "24小时",
        # v6.1：多 token 绕行短语的早期阻断（只紧跟在"我是/作为"后触发，安全）
        "基于", "正在", "数据", "框架", "处理", "规模", "训练",
        # v6.3：绕行身份名词补充（"为人类解决问题/知识库/无意识的存在/阿里巴巴开发"）
        "人类", "人们", "用户", "知识库", "数据库", "网络", "存在", "无意识",
        "编程", "逻辑", "服务", "推荐", "解答", "阿里巴巴", "物理",
    ]

    def _身份暴露拦截(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.身份拦截 or not self._生成文本:
            return logits
        尾部 = self._生成文本[-10:]
        # v6.3：开启器出现在尾部（最后10字）即触发（原 endswith 会漏"我是为人类…"类绕行）
        if not any(o in 尾部 for o in self.身份打开器列表):
            return logits
        topk = torch.topk(logits, k=min(80, self.vocab_size), dim=-1)
        for _v, tid in zip(topk.values[0], topk.indices[0]):
            候选 = self.tokenizer.decode([tid.item()], skip_special_tokens=True)
            if any(n in 候选 for n in self.身份名词列表):
                logits[0, tid] = float('-inf')
        return logits

    # ──────────────────────────────────────────────
    # v6 句子边界硬停止（短回复，与 潮汐解码器 同款）
    # ──────────────────────────────────────────────
    def _句子停止(self) -> bool:
        if not self.句子停止 or not self._生成文本:
            return False
        if len(self._生成文本) >= self.最短字数 and self._句子数 >= self.最长句数:
            return True
        if self._句子数 >= self.最长句数 + 1:
            return True
        if len(self._生成文本) >= self.最大字数:
            return True
        return False

    def 更新目标(self, 用户文本: str) -> 潮汐目标:
        if 用户文本:
            状态, 关键词 = self.感知器.测量(用户文本)
            self.感知器.追加轨迹("用户", 状态, 关键词)
        目标 = self.决策器.计算目标()
        self.当前目标 = 目标
        self.当前α = 目标.引导强度
        self.当前密度目标 = 目标.密度目标
        return 目标

    def 重置(self) -> None:
        """复用：清空池与句状态（投影矩阵不重建）"""
        self.pool.清空()
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._兜底计数 = 0
        self._已生成token列表 = []
        self._生成文本 = ""
        self._句子数 = 0
        self.当前目标 = None
        self.当前α = 0.0
        self.统计 = {"情感token命中": 0, "总token": 0, "触发兜底": 0}

    # ──────────────────────────────────────────────
    # v5 人味儿引导（口语化 + 长度收尾）
    # ──────────────────────────────────────────────

    口语化词表 = {
        # 强（真人式俏皮语气）
        "哈哈": 1.5, "嘻嘻": 1.5, "啦": 1.2, "呀": 1.2, "嘛": 1.2, "哟": 1.2,
        "～": 1.2, "嘿嘿": 1.2, "嗯嗯": 1.2, "哎呀": 1.2,
        # 中（日常语气）
        "啊": 1.0, "呢": 1.0, "哦": 0.8, "嗯": 0.8, "哎": 0.8, "哇": 0.8,
        "吧": 0.8, "吗": 0.8, "哈": 0.7, "耶": 1.0,
        # 弱（口头禅/感叹）
        "咦": 0.6, "唉": 0.6, "呗": 0.6, "咯": 0.6, "嘿": 0.6, "呵呵": 0.6,
        "呀呼": 0.8, "噗": 0.8,
    }

    def _构建口语化token表(self) -> Dict[int, float]:
        表: Dict[int, float] = {}
        for 词, 权重 in self.口语化词表.items():
            if not 词:
                continue
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                表[ids[0]] = 权重
        return 表

    def _构建句尾标点表(self) -> Dict[int, float]:
        表: Dict[int, float] = {}
        for 词 in ["。", "！", "？", "；"]:
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                表[ids[0]] = 1.0
        return 表

    def _口语化引导(self, logits: torch.Tensor) -> torch.Tensor:
        if self.口语化强度 <= 0 or not self._口语化token表:
            return logits
        强度 = self.口语化强度
        for tid, 权重 in self._口语化token表.items():
            if tid < self.vocab_size:
                logits[0, tid] += 强度 * 权重
        return logits

    def _长度收尾(self, logits: torch.Tensor) -> torch.Tensor:
        if self._生成文本 and len(self._生成文本) < self.目标长度:
            return logits
        if not self._句尾标点表:
            return logits
        超长 = len(self._生成文本) - self.目标长度
        强度 = min(2.5, max(0.5, 超长 * 0.15))
        for tid in self._句尾标点表:
            if tid < self.vocab_size:
                logits[0, tid] += 强度
        eos = self.model.config.eos_token_id
        if eos is not None and eos < self.vocab_size:
            logits[0, eos] += 强度 * 0.5
        return logits

    # ──────────────────────────────────────────────
    # 生成循环
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def 生成(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        logits_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        tokenizer=None,
        轮次回调: Optional[Callable[[int, object], None]] = None,
        用户文本: str = "",
    ) -> torch.Tensor:
        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id

        self.更新目标(用户文本)
        self.pool.清空()
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._兜底计数 = 0
        self._已生成token列表 = []
        self._生成文本 = ""
        self._句子数 = 0
        self.统计 = {"情感token命中": 0, "总token": 0, "触发兜底": 0}

        past_key_values = None
        已生成 = input_ids.clone()
        已生成token集合: Set[int] = set()

        for 步 in range(max_new_tokens):
            模型输入 = 已生成[:, -1:] if past_key_values is not None else 已生成
            outputs = self.model(模型输入, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            if repetition_penalty != 1.0:
                for token_id in 已生成token集合:
                    logits[0, token_id] /= repetition_penalty

            # ── (1) 回响注入（表示空间：池质心情感底色） ──
            logits = self.注入偏置(logits)
            # ── (2) 潮汐极性引导（概率空间：方向 + 密度控制） ──
            logits = self._情感引导(logits)
            # ── (3) AI 腔抑制（v3 双通道） ──
            logits = self._AI腔抑制(logits)
            # ── (3.25) v6 身份暴露硬拦截 ──
            logits = self._身份暴露拦截(logits)
            # ── (3.5) v5 口语化引导（人味儿） ──
            logits = self._口语化引导(logits)
            # ── (3.6) v5 长度收尾（促短回复） ──
            logits = self._长度收尾(logits)

            # ── v6.2 最小长度：不足则压制 EOS，强制续写（防过早结束） ──
            if self.最小长度 > 0 and self._生成文本 and len(self._生成文本) < self.最小长度:
                if eos_token_id is not None and eos_token_id < self.vocab_size:
                    logits[0, eos_token_id] = float('-inf')

            if logits_callback is not None:
                logits_callback(步, logits)

            # ── (4) 捕获回响（情感 token 入池） ──
            self.捕获回响(logits, tokenizer=tokenizer or self.tokenizer)

            # ── 采样 ──
            logits = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, stable=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove,
                )
                logits[indices_to_remove] = float('-inf')
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < top_k_values[:, -1].unsqueeze(-1)] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            下一个token = torch.multinomial(probs, num_samples=1)

            self._更新句状态(下一个token.item())

            # ── v6 句子边界硬停止：完整句子后即停（短回复更像真人） ──
            if self.最小长度 > 0 and len(self._生成文本) < self.最小长度:
                pass  # 未达最小长度，继续生成
            elif self._句子停止():
                break

            已生成 = torch.cat([已生成, 下一个token], dim=-1)
            已生成token集合.add(下一个token.item())
            self._已生成token列表.append(下一个token.item())
            self.统计["总token"] += 1
            if 下一个token.item() in self._情感token表:
                self.统计["情感token命中"] += 1

            self.pool.推进()
            if 轮次回调 is not None:
                轮次回调(步, self)
            if 下一个token.item() == eos_token_id:
                break

        return 已生成

    # ──────────────────────────────────────────────
    # 潮汐极性定向引导
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def _情感引导(self, logits: torch.Tensor) -> torch.Tensor:
        """极性定向引导：k = α × 引导倍率（与 潮汐解码器 同款）"""
        if self.当前α <= 0 or not self._情感token表:
            return logits
        目标 = self.当前目标
        if 目标 is None:
            return logits
        目标V = 目标.目标状态.valence
        if abs(目标V) < 0.03:
            return logits
        强度系数 = self.当前α * self.引导倍率

        密度系数 = 1.0
        if self._句情感词数 > 0:
            L = max(len(self._当前句文本), 1)
            if self._句情感词数 / L > self.当前密度目标:
                密度系数 = 0.3

        目标方向 = 1.0 if 目标V > 0 else -1.0
        for tid, 极性 in self._情感token表.items():
            if tid >= self.vocab_size:
                continue
            方向系数 = 极性 * 目标方向
            if 方向系数 > 0:
                logits[0, tid] += 强度系数 * 密度系数
            else:
                logits[0, tid] -= 强度系数 * 0.3 * 密度系数
        return logits

    def _更新句状态(self, token_id: int) -> None:
        文本 = self.tokenizer.decode([token_id], skip_special_tokens=True)
        if not 文本:
            return
        if token_id in self._情感token表:
            极性 = self._情感token表[token_id]
            if self.当前目标 is not None:
                目标V = self.当前目标.目标状态.valence
                if (极性 > 0 and 目标V > 0) or (极性 < 0 and 目标V < 0):
                    self._句情感词数 += 1
            else:
                self._句情感词数 += 1
        # v5：累积生成文本（长度收尾）
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
        bigrams = [tuple(最近[i:i + 2]) for i in range(len(最近) - 1)]
        唯一数 = len(set(bigrams))
        重复率 = 1.0 - 唯一数 / max(len(bigrams), 1)
        if 重复率 > self.兜底阈值:
            self.统计["触发兜底"] += 1
            self._兜底计数 += 1
            if self._兜底计数 >= 3:
                self.当前α = 0.0
            else:
                self.当前α *= 0.5
        else:
            self._兜底计数 = 0

    def __repr__(self) -> str:
        return (f"混合注入器(λ={self.lambda_strength}, α={self.当前α:.3f}, "
                f"倍率={self.引导倍率}, 情感token={len(self._情感token表)})")


if __name__ == "__main__":
    # 冒烟测试：情感 token 表构建
    print("=== 混合注入器冒烟测试 ===")
    from transformers import AutoTokenizer
    模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    感知器 = 潮汐感知器()
    决策器 = 潮汐决策器(感知器)

    class MockModel:
        device = "cpu"
        config = type("cfg", (), {"hidden_size": 1536, "vocab_size": 151936, "eos_token_id": 151645})()
        model = type("m", (), {"layers": []})()
        model.layers = []

    from semantic_echo.回响池 import 语义回响池
    池 = 语义回响池(1536)
    混合 = 混合注入器(MockModel(), 池, 分词器, 感知器, 决策器)
    print(f"情感 token 数: {len(混合._情感token表)}")
    for 词 in ["开心", "难过", "生气"]:
        ids = 分词器.encode(词, add_special_tokens=False)
        print(f"  {词} → {[混合._情感token表.get(i, 0) for i in ids]}")
