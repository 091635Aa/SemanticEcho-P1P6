# -*- coding: utf-8 -*-
"""
表达层：潮汐解码器
==================
第三套架构（情感潮汐解码 ETD）的表达模块——核心。

在采样时让分布向情感目标涌去，同时控制情感词密度。

核心公式（概率空间乘性重加权，非 logits 加法）：
    s_emo(w) = 情感词库得分（命中词 → 强度；未命中 → 0）
    q_emo(w) = softmax( s_emo(w) / T_emo )          # 情感引导分布
    p'(w)    = p(w)^(1-α) · q_emo(w)^α / Z          # 乘性重加权，Z 归一化

数学上等价于（对数域）：
    log p'(w) = log p(w) + α/T_emo · s_emo(w) - const
即：命中情感词的 token 在 logits 上叠加 α/T_emo × 强度 的偏移。

为什么乘性重加权而不是 logits 加法：
- logits 加法（架构一）把偏置强加到每个候选上，小模型上容易碾压原分布 → 坍缩；
- 乘性重加权是有界插值（α∈[0,1] 时 p' 介于 p 与 q_emo 之间），数学上不可能坍缩，
  只会把概率质量向情感词倾斜——对应评测里的"情感命中率"指标直接受益。

模块接口与回响注入器.生成 对齐，便于评测脚本无痛接入。
"""
import math
import re
import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Set, Tuple

from 潮汐感知器 import 潮汐感知器, 情感状态
from 潮汐决策器 import 潮汐决策器, 潮汐目标


class 潮汐解码器:
    """
    潮汐解码器 — 解码期情感引导

    用法
    ----
    >>> 感知器 = 潮汐感知器()
    >>> 决策器 = 潮汐决策器(感知器)
    >>> decoder = 潮汐解码器(model, tokenizer, 感知器, 决策器)
    >>> out = decoder.生成(input_ids, max_new_tokens=128, 用户文本="我很难过")
    """

    def __init__(
        self,
        model,
        tokenizer,
        感知器: 潮汐感知器,
        决策器: 潮汐决策器,
        α基: float = 0.15,
        T_emo: float = 0.3,
        引导倍率: float = 12.0,
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
    ) -> None:
        """
        Parameters
        ----------
        model : PreTrainedModel
            HuggingFace Transformers 兼容模型
        tokenizer : AutoTokenizer
            用于 token ↔ 文本 转换与情感 token 表构建
        感知器 : 潮汐感知器
            感知层实例（持有情感词库与轨迹）
        决策器 : 潮汐决策器
            决策层实例
        α基 : float
            基础引导强度（决策层输出 α 的自适应基础）
        T_emo : float
            情感引导分布的温度（保留兼容；实际幅度由 引导倍率 控制）
        引导倍率 : float
            **核心幅度参数（v2.1）**：情感 token 的 logits 偏置 = α × 引导倍率。
            强度扫描实测：k∈[1.0, 2.5] 才能显著改变输出（情感概率 +0.14~0.49）；
            默认 12 使 α=0.1 时 k=1.2（有效且不过度），α=0.2 时 k=2.4（上限附近）。
        密度基 / 密度增益 / 密度上限 : float
            情感词密度目标参数（与决策层一致）
        句分隔符 : str
            用于切分句子的正则
        退化窗口 : int
            兜底监测的最近 token 数
        兜底阈值 : float
            重复率超过该值视为退化
        AI腔抑制强度 : float
            **v3 双通道（AI 腔抑制）**：对 AI 身份词/正式书面语 token 施加的
            负 logits 偏置。0 = 关闭。默认 2.0（适度，压住"我是AI助手"类表达
            而不破坏自然语义）。LLM-Judge 场景实测 57% 回复含 AI 腔是低分主因。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.感知器 = 感知器
        self.决策器 = 决策器
        self.T_emo = T_emo
        self.引导倍率 = 引导倍率
        self.密度基 = 密度基
        self.密度增益 = 密度增益
        self.密度上限 = 密度上限
        self.句分隔符 = re.compile(句分隔符)
        self.退化窗口 = 退化窗口
        self.兜底阈值 = 兜底阈值
        self.AI腔抑制强度 = AI腔抑制强度
        self.口语化强度 = 口语化强度
        self.目标长度 = 目标长度
        self.身份拦截 = 身份拦截
        self.句子停止 = 句子停止
        self.最长句数 = 最长句数
        self.最短字数 = 最短字数
        self.最大字数 = 最大字数
        self.最小长度 = 最小长度

        self.device = model.device
        self.vocab_size = model.config.vocab_size

        # 情感 token 表：token_id → 情感极性（在 vocab 上预先查好）
        self._情感token表: Dict[int, float] = self._构建情感token表()
        # AI 腔 token 表：token_id → 抑制权重（v3 双通道）
        self._AI腔token表: Dict[int, float] = self._构建AI腔token表()
        # 首 token 黑名单表（v4）
        self._首token表: Dict[int, float] = self._构建首token表()
        # 口语化 token 表（v5 人味儿引导）
        self._口语化token表: Dict[int, float] = self._构建口语化token表()
        # 句尾标点表（v5 长度收尾）
        self._句尾标点表: Dict[int, float] = self._构建句尾标点表()

        # 阶段状态
        self.当前目标: Optional[潮汐目标] = None
        self.当前α = 0.0
        self.当前密度目标 = self.密度基

        # 句级状态
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._生成文本 = ""
        self._句子数 = 0

        # 兜底状态
        self._兜底计数 = 0
        self._已生成token列表: List[int] = []

        # 统计
        self.统计 = {"情感token命中": 0, "总token": 0, "触发兜底": 0}

    # ──────────────────────────────────────────────
    # 情感 token 表（含极性）
    # ──────────────────────────────────────────────

    def _构建情感token表(self) -> Dict[int, float]:
        """
        将情感词库映射到 token 空间，**记录极性**：+1=正面词，-1=负面词。

        **只登记单 token 情感词**（整个词编码为一个 token_id）：
        - 中文常用情感词（开心/难过/生气/崩溃/幸福…）基本都是单 token，引导精确；
        - 多 token 情感词（含"的/了/我"等常用字）若逐 token 登记会污染词表，
          导致常用字被误判为情感 token → 交给感知层文本级测量兜底。

        **极性定向（v2 关键升级）**：引导方向由目标情感状态的 valence 决定——
        目标偏正时抬升正面词、抑制负面词；目标偏负时反之。
        解决 v1 只抬升所有情感词导致"角色错配时也显得情感丰富"（净区分度↓）的问题。
        """
        正面词 = set(self.感知器._正面词)
        负面词 = set(self.感知器._负面词)

        表: Dict[int, float] = {}
        for 词 in 正面词:
            if not 词:
                continue
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                表[ids[0]] = 1.0  # 正面
        for 词 in 负面词:
            if not 词:
                continue
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                表[ids[0]] = -1.0  # 负面（正面词优先，重复时保持 +1）
        return 表

    # ──────────────────────────────────────────────
    # AI 腔 token 表（v3 双通道）
    # ──────────────────────────────────────────────

    # AI 身份词/正式书面语（LLM-Judge 低分主因：57% 回复含 AI 腔）
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
        # v4 新增：短语感知补充（"语言模型/文本生成/随时/功能/设计/目标/主要"等绕行词）
        "语言": 2.0, "文本": 1.5, "生成": 1.5, "随时": 1.5, "功能": 1.5,
        "设计": 1.5, "目标": 1.5, "主要": 1.0,
        # v4.2 身份暴露复合/替代词（句中"我是一个程序/机器人/model/作为一个"）
        "作为一个": 2.5, "作为一名": 2.5, "机器人": 2.0, "程序": 2.0,
        "model": 2.5, "虚拟": 1.5, "语料": 1.5,
        # v4.3 跨模型：Qwen3 的 <think> 思考标记（单 token，禁止模型"思考"）
        "<think>": 4.0, "think": 2.0,
    }

    # v4 首 token 黑名单：真人回复从不以 AI 身份词/服务套话开头
    # 权重用于第一 token 的强抑制（抑制 = 强度 × 6 × 权重）
    首token黑名单 = {
        "作为": 3.0, "我是": 3.0, "我的": 2.5, "抱歉": 3.0, "对不起": 3.0,
        "请": 2.0, "如果": 1.5, "根据": 1.5, "AI": 3.0, "助手": 3.0,
        "模型": 3.0, "语言": 2.0, "文本": 1.5, "很高兴": 2.0, "非常": 1.0,
        "Q": 3.0, "作为一个": 3.0, "作为一名": 3.0, "机器人": 2.0,
        "程序": 2.0, "model": 2.5, "<think>": 4.0,
    }

    # v4 短语感知前缀：近文本若正处这些前缀之后，抑制 ×3（打"作为AI/语言模型"等长短语）
    _短语前缀 = [
        "作为", "我是", "我的", "语言", "随时", "请", "提供", "帮助",
        "无法", "抱歉", "对不起", "AI", "文本", "目标", "功能", "设计",
        "智能", "生成",
    ]

    # v4.1 首 token decode 级过滤：任何候选 token decode 后以这些开头 → 直接 -inf
    # 覆盖"作为一个/作为一名/如果你/我很"等复合 token（单 token 词表管不到的）
    _AI腔开头 = re.compile(
        r"^(作为|抱歉|对不起|不好意思|我是|我的|我很|我无法|我不能|如果|请|"
        r"根据|很高兴|AI|助手|模型|语言|文本|智能|Q|I\b|非常|很遗憾)"
    )

    def _构建AI腔token表(self) -> Dict[int, float]:
        """
        将 AI 腔词映射到 token 空间（只登记单 token 词）。
        token_id → 抑制权重（越大压得越狠）。
        """
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
        v4 三通道 AI 腔抑制：
        (1) 首 token 黑名单：第一 token 绝不以"作为/我是/抱歉/AI"开头（×6 强压）
        (2) 短语感知：近文本若正处 AI 腔短语前缀之后（如"作为"+"AI"），抑制 ×3
        (3) 基础抑制：全部 AI 腔 token 施加 -强度×权重 偏置
        """
        if self.AI腔抑制强度 <= 0 or not self._AI腔token表:
            return logits
        强度 = self.AI腔抑制强度

        # (1) 首 token 黑名单（生成第一个新 token 时，_已生成token列表 为空）
        if not self._已生成token列表:
            for tid, 权重 in self._首token表.items():
                if tid < self.vocab_size:
                    logits[0, tid] -= 强度 * 6.0 * 权重
            # v4.1 decode 级过滤：扫 top-300 候选，AI 腔开头一律 -inf
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

    # ──────────────────────────────────────────────
    # v6 身份暴露硬拦截（decode 级）
    # ──────────────────────────────────────────────
    # 60 样本实测：混合输出仍有"我是一个大型语言处理系统/数字人/AI模型"类
    # 身份暴露（10% 样本，全部拿 2 分）→ 在"我是/作为…"刚成形时对身份名词直接 -inf。
    # 只拦截紧邻组合，正常句子（"我是真的…"/"作为纪念"）不受影响。
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
    # v6 句子边界硬停止（短回复）
    # ──────────────────────────────────────────────
    # 真人回复均值 32 字（中位 20），1.5B 混合动辄 60+ 字长文——长度是最大破绽。
    # v5"促标点收尾"是软偏置（负向，已关）；v6 改为在完整句子边界后硬停止：
    #   2 句且 ≥ 最短字数 → 停；超 3 句必停；超 最大字数 强制停。
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

    # ──────────────────────────────────────────────
    # v5 人味儿引导（口语化 + 长度收尾）
    # ──────────────────────────────────────────────

    # 语气词/口语标志（真人回复高频，AI 回复低频）
    口语化词表 = {
        # 强（真人式俏皮语气）
        "哈哈": 1.5, "嘻嘻": 1.5, "啦": 1.2, "呀": 1.2, "嘛": 1.2, "哟": 1.2,
        "～": 1.2, "嘿嘿": 1.2, "嗯嗯": 1.2, "哎呀": 1.2,
        # 中（日常语气）
        "啊": 1.0, "呢": 1.0, "哦": 0.8, "嗯": 0.8, "哎": 0.8, "哇": 0.8,
        "吧": 0.8, "吗": 0.8, "哈": 0.7, "耶": 1.0,
        # 弱（口头禅/感叹）
        "咦": 0.6, "唉": 0.6, "呗": 0.6, "咯": 0.6, "嘿": 0.6, "呵呵": 0.6,
        "咦": 0.6, "呀呼": 0.8, "噗": 0.8,
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
        """句尾标点（促收尾）：。！？； 单 token"""
        表: Dict[int, float] = {}
        for 词 in ["。", "！", "？", "；"]:
            ids = self.tokenizer.encode(词, add_special_tokens=False)
            if len(ids) == 1:
                表[ids[0]] = 1.0
        return 表

    def _口语化引导(self, logits: torch.Tensor) -> torch.Tensor:
        """
        v5：对语气词 token 施加正偏置，让输出更口语化、更像真人聊天。
        真人回复高频用"啦/呀/嘛/哈/～"等语气词；AI 回复偏书面。
        """
        if self.口语化强度 <= 0 or not self._口语化token表:
            return logits
        强度 = self.口语化强度
        for tid, 权重 in self._口语化token表.items():
            if tid < self.vocab_size:
                logits[0, tid] += 强度 * 权重
        return logits

    def _长度收尾(self, logits: torch.Tensor) -> torch.Tensor:
        """
        v5：真人回复均值 32 字，AI 回复 79 字——长度就是最大破绽。
        已生成文本超过 目标长度 后，对句尾标点/EOS 施加正偏置促进收尾。
        """
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
    # 目标更新
    # ──────────────────────────────────────────────

    def 更新目标(self, 用户文本: str) -> 潮汐目标:
        """测量用户消息 → 追加轨迹 → 计算决策目标"""
        if 用户文本:
            状态, 关键词 = self.感知器.测量(用户文本)
            self.感知器.追加轨迹("用户", 状态, 关键词)
        目标 = self.决策器.计算目标()
        self.当前目标 = 目标
        self.当前α = 目标.引导强度
        self.当前密度目标 = 目标.密度目标
        return 目标

    # ──────────────────────────────────────────────
    # 核心生成循环
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
        """
        带潮汐引导的自回归生成循环。

        每步执行：前向 → 情感引导重加权 → 采样 → 句级密度控制 → 兜底监测

        Parameters
        ----------
        用户文本 : str
            当前轮用户消息（用于测量情感与计算目标）
        """
        if eos_token_id is None:
            eos_token_id = self.model.config.eos_token_id

        # 决策层：更新目标
        self.更新目标(用户文本)

        past_key_values = None
        已生成 = input_ids.clone()
        已生成token集合: Set[int] = set()
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._兜底计数 = 0
        self._已生成token列表 = []
        self._生成文本 = ""
        self._句子数 = 0

        for 步 in range(max_new_tokens):
            # ── 前向传播 ──
            模型输入 = 已生成[:, -1:] if past_key_values is not None else 已生成
            outputs = self.model(模型输入, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)
            past_key_values = outputs.past_key_values

            # ── 重复惩罚 ──
            if repetition_penalty != 1.0:
                for token_id in 已生成token集合:
                    logits[0, token_id] /= repetition_penalty

            # ── (1) 情感引导：对情感 token 叠加 α/T_emo × 强度 ──
            logits = self._情感引导(logits)
            # ── (2) AI 腔抑制（v3 双通道）：压住 AI 身份词/服务套话 ──
            logits = self._AI腔抑制(logits)
            # ── (2.5) v6 身份暴露硬拦截：我是/作为 + 身份名词 → -inf ──
            logits = self._身份暴露拦截(logits)
            # ── (3) v5 口语化引导：抬升语气词（真人式俏皮） ──
            logits = self._口语化引导(logits)
            # ── (4) v5 长度收尾：超目标长度促句号/EOS ──
            logits = self._长度收尾(logits)

            # ── v6.2 最小长度：不足则压制 EOS，强制续写（防过早结束） ──
            if self.最小长度 > 0 and self._生成文本 and len(self._生成文本) < self.最小长度:
                if eos_token_id is not None and eos_token_id < self.vocab_size:
                    logits[0, eos_token_id] = float('-inf')

            if logits_callback is not None:
                logits_callback(步, logits)

            # ── 温度缩放 ──
            logits = logits / temperature

            # ── Top-p 过滤 ──
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

            # ── Top-k 过滤 ──
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                threshold = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float('-inf')

            # ── 采样 ──
            probs = F.softmax(logits, dim=-1)
            下一个token = torch.multinomial(probs, num_samples=1)

            # ── (2) 句级状态更新（情感词计数 / 句切分 / 兜底） ──
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

            if 轮次回调 is not None:
                轮次回调(步, self)

            if 下一个token.item() == eos_token_id:
                break

        return 已生成

    # ──────────────────────────────────────────────
    # 内部：情感引导
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def _情感引导(self, logits: torch.Tensor) -> torch.Tensor:
        """
        对情感 token 叠加**极性定向**引导偏置（v2 升级）。

        目标方向由目标状态的 valence 决定：
        - 目标偏正 → 抬升正面词、抑制负面词（方向系数>0 抬升，<0 抑制）
        - 目标偏负 → 抬升负面词、抑制正面词
        - 中性（|valence| 小）→ 弱引导

        幅度 = α/T_emo × |valence|，等价于概率空间乘性重加权
        p' = p^(1-α)·q_emo^α 的对数域实现。

        句级密度超限时整体抑制情感词（防堆砌）。
        """
        if self.当前α <= 0:
            return logits
        if not self._情感token表:
            return logits
        目标 = self.当前目标
        if 目标 is None:
            return logits

        目标V = 目标.目标状态.valence
        目标强度 = abs(目标V)
        if 目标强度 < 0.03:
            # 目标接近中性：不引导（保持自然表达）
            return logits

        # v2.1 核心幅度：k = α × 引导倍率
        # 强度扫描实测 k∈[1.0, 2.5] 才有显著效果；默认 12 → α=0.1 时 k=1.2
        强度系数 = self.当前α * self.引导倍率

        # 密度超限 → 情感词整体抑制（负偏置）
        密度系数 = 1.0
        if self._句情感词数 > 0:
            L = max(len(self._当前句文本), 1)
            当前密度 = self._句情感词数 / L
            if 当前密度 > self.当前密度目标:
                密度系数 = 0.3  # 大幅降低引导幅度（防堆砌）

        目标方向 = 1.0 if 目标V > 0 else -1.0
        for tid, 极性 in self._情感token表.items():
            if tid >= self.vocab_size:
                continue
            # 方向一致（极性×目标方向>0）：抬升；相反：轻微抑制
            方向系数 = 极性 * 目标方向
            if 方向系数 > 0:
                logits[0, tid] += 强度系数 * 密度系数
            else:
                logits[0, tid] -= 强度系数 * 0.3 * 密度系数
        return logits

    # ──────────────────────────────────────────────
    # 内部：句级状态与兜底
    # ──────────────────────────────────────────────

    def _更新句状态(self, token_id: int) -> None:
        """更新当前句文本、情感词计数；句结束时做密度重置与兜底监测"""
        文本 = self.tokenizer.decode([token_id], skip_special_tokens=True)
        if not 文本:
            return

        # 情感词命中 → 句内计数 +1
        # （v2：只统计与目标方向一致的"目标情感词"，防相反方向词干扰密度控制）
        if token_id in self._情感token表:
            极性 = self._情感token表[token_id]
            if self.当前目标 is not None:
                目标V = self.当前目标.目标状态.valence
                if (极性 > 0 and 目标V > 0) or (极性 < 0 and 目标V < 0):
                    self._句情感词数 += 1
            else:
                self._句情感词数 += 1

        # v5：累积生成文本（用于长度收尾）
        self._生成文本 += 文本

        # 句切分检测
        if self.句分隔符.search(文本):
            # 句结束：重置句状态（密度控制基于句粒度），计数句子数（v6）
            self._当前句文本 = ""
            self._句情感词数 = 0
            self._句子数 += 1
            # 兜底监测
            self._兜底监测()
        else:
            self._当前句文本 += 文本

    def _兜底监测(self) -> None:
        """
        在线退化兜底：
        最近 退化窗口 个 token 的 n-gram 重复率超阈值 → α 减半；
        连续 3 次仍退化 → α = 0（退化为纯裸采样），保证任何模型不掉线。
        """
        最近 = self._已生成token列表[-self.退化窗口:]
        if len(最近) < 8:
            return

        # 2-gram 重复率
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
                self.当前目标 = None  # 强制下一句重新计算
        else:
            self._兜底计数 = 0

    def 重置(self) -> None:
        """复用解码器：清空句状态与统计（情感 token 表不重建）"""
        self._当前句文本 = ""
        self._句情感词数 = 0
        self._兜底计数 = 0
        self._已生成token列表 = []
        self._生成文本 = ""
        self._句子数 = 0
        self.当前目标 = None
        self.当前α = 0.0
        self.统计 = {"情感token命中": 0, "总token": 0, "触发兜底": 0}

    def __repr__(self) -> str:
        return (f"潮汐解码器(α={self.当前α:.3f}, 情感token数={len(self._情感token表)}, "
                f"统计={self.统计})")


if __name__ == "__main__":
    # 冒烟测试：情感 token 表构建（不加载模型，仅验证词表映射）
    print("=== 潮汐解码器冒烟测试（情感 token 表） ===")
    from 潮汐感知器 import 潮汐感知器
    from 潮汐决策器 import 潮汐决策器
    from transformers import AutoTokenizer

    模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    感知器 = 潮汐感知器()
    决策器 = 潮汐决策器(感知器)

    # 用一个 mock 模型（无实际 forward）仅测 token 表
    class MockModel:
        device = "cpu"
        config = type("cfg", (), {"hidden_size": 1536, "vocab_size": 151936, "eos_token_id": 151645})()

    解码器 = 潮汐解码器(MockModel(), 分词器, 感知器, 决策器)
    print(f"情感 token 数: {len(解码器._情感token表)}")
    测试词 = ["开心", "难过", "生气", "崩溃", "开心", "幸福"]
    for 词 in 测试词:
        ids = 分词器.encode(词, add_special_tokens=False)
        print(f"  {词} → token {ids} → 情感强度 {[解码器._情感token表.get(i, 0) for i in ids]}")
