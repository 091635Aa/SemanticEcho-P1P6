# -*- coding: utf-8 -*-
"""
感知层：情感潮汐测量器
======================
第三套架构（情感潮汐解码 ETD）的感知模块。

把文本对话转成显式、连续、可追踪的情感状态：
- VAD 三维投影：V = (valence 愉悦度, arousal 唤醒度, dominance 支配度) ∈ [-1,1]³
- 会话情感轨迹：按时间记录对话情感变化——"潮汐"的涨落
- 关键词提取：供裁判审计与解释

与架构一（语义回响）的本质区别：
  架构一读模型【内部】hidden_state（每 token 一次钩子，必须本地权重）；
  本模块读【文本层】情感（每句一次，成本可忽略，闭源 API 同样适用）。

灵感来源：语义回响的"情感先于表达"思想，但情感载体换成外部连续情感状态。
"""
import re
import math
import dataclasses
from typing import List, Optional, Tuple

from cnsenti import Sentiment


# 英文情感词库（NRC 高频情感词精简版，pos/neg 各 60+ 常用词）
# 用于英文场景（HEART-BENCH/EmoCharacter 等英文数据）的情感测量兜底
英文正面词 = {
    "happy", "glad", "joy", "joyful", "cheerful", "delighted", "pleased", "wonderful",
    "great", "excellent", "amazing", "awesome", "fantastic", "good", "nice", "love",
    "loved", "loving", "grateful", "thankful", "excited", "excitedly", "thrilled",
    "proud", "hopeful", "hopeful", "optimistic", "relieved", "calm", "peaceful",
    "warm", "comforted", "comforting", "confident", "satisfied", "content", "free",
    "blessed", "lucky", "success", "successful", "win", "won", "victory", "joyous",
    "smile", "smiling", "laugh", "laughter", "fun", "enjoy", "enjoying", "beautiful",
    "caring", "kind", "sweet", "gentle", "tender", "supportive", "encouraging",
    "inspiring", "motivating", "healing", "safety", "safe", "secure", "strong",
}
英文负面词 = {
    "sad", "sadness", "unhappy", "upset", "depressed", "depressing", "miserable",
    "grief", "grieving", "heartbroken", "heartbreaking", "hurt", "hurting", "pain",
    "painful", "suffering", "suffer", "cry", "crying", "tears", "lonely", "loneliness",
    "alone", "isolated", "abandoned", "rejected", "rejection", "disappointed",
    "disappointment", "frustrated", "frustrating", "angry", "anger", "furious", "rage",
    "annoyed", "irritated", "tired", "exhausted", "drained", "stressed", "anxious",
    "anxiety", "worried", "worry", "nervous", "scared", "afraid", "fear", "fearful",
    "terrified", "horrified", "hopeless", "helpless", "desperate", "despair", "shame",
    "ashamed", "guilt", "guilty", "regret", "regretful", "jealous", "envious",
    "hate", "hatred", "disgust", "disgusted", "awful", "terrible", "horrible",
    "worse", "worst", "broken", "failure", "failed", "lose", "lost", "weak",
}


@dataclasses.dataclass
class 情感状态:
    """三维连续情感状态 V = (valence, arousal, dominance) ∈ [-1,1]³"""
    valence: float   # 愉悦度：正 = 积极，负 = 消极
    arousal: float   # 唤醒度：高 = 激动/紧张，低 = 平静
    dominance: float # 支配度：高 = 果断/强势，低 = 犹豫/退缩

    def 到向量(self) -> Tuple[float, float, float]:
        return (self.valence, self.arousal, self.dominance)

    def 强度(self) -> float:
        """情感强度 = 唤醒度（情绪有多激烈）"""
        return abs(self.arousal)

    def 极性(self) -> float:
        """情感极性 = 愉悦度（正=积极 负=消极）"""
        return self.valence

    def __repr__(self) -> str:
        return (f"情感状态(V={self.valence:.3f}, A={self.arousal:.3f}, "
                f"D={self.dominance:.3f})")


@dataclasses.dataclass
class 轨迹点:
    """会话情感轨迹的一个点"""
    说话人: str            # "用户" | "模型"
    轮次: int
    状态: 情感状态
    关键词: List[str]


def _平滑(x: float) -> float:
    """tanh 平滑到 [-1,1]，保留符号与小值"""
    return math.tanh(x)


class 潮汐感知器:
    """
    情感潮汐测量器

    用法
    ----
    >>> 感知器 = 潮汐感知器()
    >>> 状态, 关键词 = 感知器.测量("我今天好开心啊！")
    >>> 感知器.追加轨迹("用户", 状态)
    >>> 潮位 = 感知器.当前潮位()
    """

    def __init__(
        self,
        词库: str = "cnsenti",
        窗口: int = 32,
        位置衰减: float = 0.9,
        句衰减: float = 0.95,
    ) -> None:
        """
        Parameters
        ----------
        词库 : str
            情感词库来源，当前仅支持 cnsenti（与现有情感过滤器同源）
        窗口 : int
            轨迹最大容量（轮数），超出后按指数遗忘最旧项
        位置衰减 : float
            位置权重指数衰减系数（越靠近当前句权重越高），每次追加轨迹后旧状态 × 该系数
        句衰减 : float
            单条文本内按句子位置衰减系数（越靠后的句子权重越高，用于流式追加）
        """
        if 词库 != "cnsenti":
            raise ValueError(f"当前仅支持词库 'cnsenti'，收到 {词库!r}")
        if 窗口 <= 0:
            raise ValueError(f"窗口必须为正整数，收到 {窗口}")
        if not (0 < 位置衰减 <= 1.0):
            raise ValueError(f"位置衰减必须在 (0,1] 内，收到 {位置衰减}")
        if not (0 < 句衰减 <= 1.0):
            raise ValueError(f"句衰减必须在 (0,1] 内，收到 {句衰减}")

        self.窗口 = 窗口
        self.位置衰减 = 位置衰减
        self.句衰减 = 句衰减

        # 情感分析器（cnsenti 内置知网 Hownet 词典：pos/neg/程度词/否定词）
        self._分析器 = Sentiment()

        # 会话轨迹
        self._轨迹: List[轨迹点] = []
        self._轮次 = 0

        # 静态词表缓存：把词典转 set 加速命中判断
        self._正面词 = set(self._分析器.Poss) | 英文正面词
        self._负面词 = set(self._分析器.Negs) | 英文负面词

        # 词表来源标记（中文来自 cnsenti，英文为内置 NRC 精简版）
        self._中文正面 = set(self._分析器.Poss)
        self._中文负面 = set(self._分析器.Negs)

        # 程度副词权重（cnsenti 内置分级）
        self._程度词 = {}
        for w in self._分析器.Extremes:
            self._程度词[w] = 4.0
        for w in self._分析器.Verys:
            self._程度词[w] = 3.0
        for w in self._分析器.Mores:
            self._程度词[w] = 2.0
        for w in self._分析器.Ishs:
            self._程度词[w] = 0.5

        # 否定词表（用于极性反转）
        self._否定词 = set(self._分析器.Denys)

    # ──────────────────────────────────────────────
    # 单文本 VAD 测量
    # ──────────────────────────────────────────────

    def 测量(self, 文本: str) -> Tuple[情感状态, List[str]]:
        """
        对单段文本做 VAD 情感投影。

        自实现词典扫描（不依赖 cnsenti sentiment_calculate，其含累积 bug）：
        1. jieba 分词 + 子串匹配（解决"好开心/太高兴"复合词命中失败）
        2. 否定词反转极性（奇数个否定 → 反转）
        3. 程度副词加权（极=×4, 很=×3, 较=×2, 稍=×0.5）

        - valence = tanh((pos得分 - neg得分) / max(words,1))
        - arousal = min(1, (|pos得分| + |neg得分|) / max(words,1))
        - dominance = min(1, |valence| × (1 + 强度加权))

        Returns
        -------
        (情感状态, 关键词列表)
        """
        if not 文本 or not 文本.strip():
            return 情感状态(0.0, 0.0, 0.0), []

        pos, neg, 命中词, 强度加权 = self._扫描情感得分(文本)
        词数 = max(len(list(self._分词(文本))), 1)

        # 极性：正负抵消后平滑到 [-1,1]
        valence = _平滑((pos - neg) / 词数)

        # 唤醒度：情感词密度（绝对值之和 / 词数），截断到 [0,1]
        arousal = min(1.0, (abs(pos) + abs(neg)) / 词数)

        # 支配度：情感越明确（|valence| 高）越果断；强度副词进一步放大
        dominance = min(1.0, abs(valence) * (1.0 + 强度加权))

        关键词 = self._提取关键词(命中词, top_k=5)

        return 情感状态(valence, arousal, dominance), 关键词

    # ──────────────────────────────────────────────
    # 自实现词典扫描
    # ──────────────────────────────────────────────

    def _分词(self, 文本: str) -> List[str]:
        """jieba 中文分词 + 英文按词切分（统一返回词序列）"""
        import jieba
        import re
        词序列: List[str] = []
        for 段 in re.split(r"([A-Za-z']+)", 文本):
            if 段 and 段[0].isalpha() and 段.isascii():
                # 英文段：按空格拆成单词
                词序列.extend(段.split())
            elif 段.strip():
                # 中文段：jieba 分词
                词序列.extend(jieba.lcut(段))
        return 词序列

    def _子串命中(self, 词: str) -> Tuple[str, float]:
        """
        对分词后的词做情感词典匹配。

        优先整词命中（中英文都查整词）；中文整词未命中时尝试子串（最长优先），
        解决 jieba 把"副词+情感词"切成复合词（如"好开心"）导致
        词典整词匹配失败的问题。
        """
        if 词 in self._正面词:
            return 词, 1.0
        if 词 in self._负面词:
            return 词, -1.0
        # 英文词不做子串匹配（避免"happy"误中"happily"等派生词碎片）
        if 词.isascii():
            return "", 0.0
        # 中文子串匹配：从长到短，找到第一个命中词库的子串
        for 长度 in range(len(词) - 1, 1, -1):
            for 起始 in range(len(词) - 长度 + 1):
                子 = 词[起始:起始 + 长度]
                if 子 in self._正面词:
                    return 子, 1.0
                if 子 in self._负面词:
                    return 子, -1.0
        return "", 0.0

    def _扫描情感得分(self, 文本: str) -> Tuple[float, float, List[Tuple[str, float]], float]:
        """
        扫描文本，返回 (pos得分, neg得分, 命中词列表[(词,极性)], 强度加权系数)。

        算法：
        - 对每个分词结果做子串匹配，命中情感词
        - 向前回看最多 3 个词：否定词计数（奇数 → 极性反转），程度副词取最大权重
        - 单个情感词得分 = 极性 × 程度权重；被否定时极性取反、权重减半
        """
        词序列 = self._分词(文本)
        pos, neg = 0.0, 0.0
        命中词: List[Tuple[str, float]] = []
        强度总和 = 0.0

        for i, 词 in enumerate(词序列):
            情感词, 极性 = self._子串命中(词)
            if 极性 == 0.0:
                continue

            # 回看前 3 个词：程度副词 + 否定词
            程度 = 1.0
            否定数 = 0
            for j in range(max(0, i - 3), i):
                前词 = 词序列[j]
                if 前词 in self._程度词:
                    程度 = max(程度, self._程度词[前词])
                if 前词 in self._否定词:
                    否定数 += 1

            # 奇数个否定 → 极性反转，权重减半
            if 否定数 % 2 == 1:
                极性 = -极性
                程度 *= 0.5

            得分 = 极性 * 程度
            if 得分 > 0:
                pos += 得分
            else:
                neg += -得分
            命中词.append((情感词, 极性))
            强度总和 += 程度

        # 强度加权系数：平均程度权重增量，封顶 0.5
        强度加权 = min(0.5, (强度总和 / max(len(命中词), 1) - 1.0) * 0.3) if 命中词 else 0.0
        return pos, neg, 命中词, 强度加权

    def _强度副词系数(self, 文本: str) -> float:
        """估算文本中程度副词的加权系数（0~1 增量）"""
        _, _, _, 强度加权 = self._扫描情感得分(文本)
        return 强度加权

    def _提取关键词(self, 命中词: List[Tuple[str, float]], top_k: int = 5) -> List[str]:
        """从命中词列表提取高强度情感词（按频次 + 极性强度排序）"""
        词频: dict = {}
        for w, 极 in 命中词:
            if not w or w.strip() == "":
                continue
            if w not in 词频:
                词频[w] = [0, abs(极)]
            词频[w][0] += 1
        排序 = sorted(词频.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))
        return [w for w, _ in 排序[:top_k]]

    # ──────────────────────────────────────────────
    # 会话轨迹维护
    # ──────────────────────────────────────────────

    def 追加轨迹(self, 说话人: str, 状态: 情感状态, 关键词: Optional[List[str]] = None) -> None:
        """
        追加一个轨迹点（一轮对话的情感状态）。

        旧状态每次追加后按 位置衰减 指数遗忘（模拟"潮汐随时间退去"）。
        """
        self._轮次 += 1
        self._轨迹.append(轨迹点(说话人, self._轮次, 状态, 关键词 or []))
        # 超窗口：从最旧开始淘汰（保留最近的 窗口 个）
        if len(self._轨迹) > self.窗口:
            self._轨迹 = self._轨迹[-self.窗口:]

    def 当前潮位(self, 忽略说话人: Optional[str] = None) -> 情感状态:
        """
        计算当前会话潮位（加权最新状态）。

        Parameters
        ----------
        忽略说话人 : str
            为 "模型" 时只看用户轨迹（用于"共情目标"只追踪对方情绪）

        Returns
        -------
        情感状态
            按位置指数衰减加权的综合情感状态
        """
        if not self._轨迹:
            return 情感状态(0.0, 0.0, 0.0)

        可用 = [p for p in self._轨迹 if p.说话人 != 忽略说话人]
        if not 可用:
            return 情感状态(0.0, 0.0, 0.0)

        # 权重：离当前越近权重越高
        权重 = []
        for i, p in enumerate(可用):
            权重.append(self.位置衰减 ** (len(可用) - 1 - i))
        总重 = sum(权重)
        if 总重 <= 0:
            return 情感状态(0.0, 0.0, 0.0)

        v = sum(p.状态.valence * w for p, w in zip(可用, 权重)) / 总重
        a = sum(p.状态.arousal * w for p, w in zip(可用, 权重)) / 总重
        d = sum(p.状态.dominance * w for p, w in zip(可用, 权重)) / 总重
        return 情感状态(v, a, d)

    def 轨迹活跃度(self, 最近轮数: int = 8) -> float:
        """最近 N 轮 VAD 位移之和，衡量对话情感活跃程度（0~1 量级）。

        新会话（轨迹不足 2 条）时：以最近状态的情感强度作为启动基础，
        保证首条消息的情感就能触发引导（否则冷启动 α=0 引导失效）。
        """
        if len(self._轨迹) == 0:
            return 0.0
        if len(self._轨迹) == 1:
            # 首条消息：用户情感越强，会话越活跃
            return min(1.0, abs(self._轨迹[-1].状态.arousal) * 1.5)
        最近 = self._轨迹[-最近轮数:]
        位移 = 0.0
        for i in range(1, len(最近)):
            p0, p1 = 最近[i - 1].状态, 最近[i].状态
            位移 += abs(p1.valence - p0.valence) + abs(p1.arousal - p0.arousal)
        return min(1.0, 位移 / max(len(最近) - 1, 1))

    def 最近关键词(self, 说话人: Optional[str] = None, top_k: int = 8) -> List[str]:
        """汇总最近轨迹点的关键词，供裁判审计"""
        词频: dict = {}
        可用 = [p for p in self._轨迹 if 说话人 is None or p.说话人 == 说话人]
        for p in 可用[-8:]:
            for w in p.关键词:
                词频[w] = 词频.get(w, 0) + 1
        排序 = sorted(词频.items(), key=lambda kv: -kv[1])
        return [w for w, _ in 排序[:top_k]]

    def 重置轨迹(self) -> None:
        """清空轨迹（新会话）"""
        self._轨迹 = []
        self._轮次 = 0

    def 轨迹长度(self) -> int:
        return len(self._轨迹)

    def __repr__(self) -> str:
        return (f"潮汐感知器(轨迹={self.轨迹长度()}/{self.窗口}, "
                f"轮次={self._轮次}, 位置衰减={self.位置衰减})")


if __name__ == "__main__":
    # 冒烟测试：人工核对 VAD 方向性
    感知器 = 潮汐感知器()
    测试集 = [
        ("我好开心啊，今天太高兴了！", "正 · 高唤醒"),
        ("我崩溃了，整个人都垮掉了……", "负 · 高唤醒"),
        ("嗯，就这样吧，随便。", "中性 · 低唤醒"),
        ("我非常非常生气！", "负 · 高唤醒 · 强程度"),
    ]
    print("=== 潮汐感知器冒烟测试 ===")
    for 文本, 期望 in 测试集:
        状态, 关键词 = 感知器.测量(文本)
        print(f"[{期望}] {文本!r}\n     → {状态} 关键词={关键词}")

    # 轨迹与潮位
    print("\n=== 轨迹与潮位 ===")
    for 说话人, 文本 in [("用户", "我最近好累"), ("模型", "听起来你很疲惫"),
                        ("用户", "而且特别委屈！"), ("模型", "辛苦了，抱抱你")]:
        状态, 关键词 = 感知器.测量(文本)
        感知器.追加轨迹(说话人, 状态, 关键词)
        print(f"{说话人}「{文本}」→ {状态}")
    print(f"\n当前潮位（全部）: {感知器.当前潮位()}")
    print(f"当前潮位（只看用户）: {感知器.当前潮位(忽略说话人='模型')}")
    print(f"轨迹活跃度: {感知器.轨迹活跃度():.3f}")
    print(f"最近关键词: {感知器.最近关键词()}")
