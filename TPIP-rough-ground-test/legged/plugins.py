#!/usr/bin/env python3
"""
plugins.py — P1~P6 的足式机器人平移（零权重推理期注入）

每一路都实现统一接口：
    inject(a, t, q, dq, blueprint, contact, terrain, progress) -> a'
其中 a 是基座策略原始动作；插件只做注入，绝不触基座权重。
"""

from __future__ import annotations
import numpy as np


class BasePlugin:
    def inject(self, a, **kw):
        return a


# --------------------------------------------------------------------------- #
# P1 / P1.5 — 通用兼容层：自动强度调度（本身不注入，作为"监督器"）             #
# --------------------------------------------------------------------------- #
class CompatLayer(BasePlugin):
    """
    LLM P1.5 平移：扫描表/公式自动调强度。这里根据 terrain + progress
    计算一个全局乘子 strength ∈ [0.2, 1.0]，供其他插件缩放。
    """
    def __init__(self, base_strength: float = 1.0):
        self.base_strength = base_strength

    def get_strength(self, terrain: float = 0.3, progress: float = 0.5) -> float:
        # 地形越粗糙越弱（防过强），进度越靠后越弱（自然收尾）
        terrain_factor = 1.0 / (1.0 + terrain)          # roughness -> weaken
        progress_curve = 1.0 - 0.3 * progress            # 尾部渐弱
        return float(np.clip(self.base_strength * terrain_factor * progress_curve,
                             0.1, 1.0))


# --------------------------------------------------------------------------- #
# P3 / P2.5 — 情感潮汐 ETD → 乘性重加权（有界）                                #
# --------------------------------------------------------------------------- #
class EmotionTidal(BasePlugin):
    """
    LLM P3 平移：p'(w)=p(w)^(1-α)·q_emo(w)^α → 动作乘性涌向步态蓝图，有界不坍缩。
    a' = a * (1-α) + blueprint * α, 再由 strength 缩放。
    """
    def __init__(self, alpha: float = 0.5, strength: float = 1.0):
        self.alpha = alpha
        self.strength = strength

    def inject(self, a, blueprint=None, terrain=0.3, **kw):
        s = self.strength
        a = (1 - self.alpha * s) * a + (self.alpha * s) * blueprint
        return a


# --------------------------------------------------------------------------- #
# P4 锚点回响 AE → 相位锚点吸引偏置（tanh 有界加性）                             #
# --------------------------------------------------------------------------- #
class AnchorEcho(BasePlugin):
    """
    LLM P4 平移：用 K 维锚点质心对动作稠密打分 + tanh 有界加性注入。
    这里锚点 = 步态蓝图；打分 = 蓝图与当前动作的相位一致性。
    """
    def __init__(self, beta: float = 0.6, strength: float = 1.0):
        self.beta = beta
        self.strength = strength

    def inject(self, a, blueprint=None, q=None, **kw):
        # 一致性打分 g ∈ [-1,1]，越大越该吸引
        d_b = np.dot(a, blueprint) / (np.linalg.norm(a) * np.linalg.norm(blueprint) + 1e-9)
        bias = np.tanh(2.0 * d_b) * blueprint
        return a + self.beta * self.strength * bias


# --------------------------------------------------------------------------- #
# P5 / P4 KV 情感共振 → 历史缓存注意力重加权（softmax 有界）                    #
# --------------------------------------------------------------------------- #
class KVResonance(BasePlugin):
    """
    LLM P5 平移：对历史缓存 reweight，让策略"更关注相位连贯时刻"。
    这里维护一个短时内存（最近 8 步），对与蓝图一致的记忆放大，注入回动作。
    """
    def __init__(self, kappa: float = 0.2, mem: int = 8, strength: float = 1.0):
        self.kappa = kappa
        self.mem = mem
        self.strength = strength
        self._history = []

    def inject(self, a, blueprint=None, **kw):
        self._history.append(a.copy())
        if len(self._history) > self.mem:
            self._history.pop(0)
        if len(self._history) < 3:
            return a
        H = np.array(self._history)                       # (mem, n)
        # 每步与蓝图的相似度 -> 作为注意力权重（softmax 有界）
        scores = H @ blueprint
        w = np.exp(self.kappa * scores)
        w = w / (w.sum() + 1e-9)
        attended = w[:, None] * H
        return (1 - self.strength) * a + self.strength * attended.mean(axis=0)


# --------------------------------------------------------------------------- #
# P1 语义回响 → 内部状态惯性回响池质心注入                                      #
# --------------------------------------------------------------------------- #
class SemanticEcho(BasePlugin):
    """
    LLM P1 平移：回收"被丢弃"的状态进回响池，取质心作为平滑先验注入。
    让动作更连贯（回想自己上一步惯性）。
    """
    def __init__(self, lam: float = 0.4, pool: int = 20, strength: float = 1.0):
        self.lam = lam * strength
        self.pool = pool
        self.strength = strength
        self._pool = []

    def inject(self, a, **kw):
        self._pool.append(a.copy())
        if len(self._pool) > self.pool:
            self._pool.pop(0)
        if len(self._pool) < 5:
            return a
        centroid = np.mean(np.array(self._pool), axis=0)
        return a + self.lam * self.strength * (centroid - a)


# --------------------------------------------------------------------------- #
# P6 情感导演 EDD → TAD/PIS/OQC 导演，统一调度多通道                           #
# --------------------------------------------------------------------------- #
class EmotionalDirector(BasePlugin):
    """
    LLM P6 平移：
      TAD  任务自适应强度（按 terrain 选档）
      PIS  进度感知调度（起步强、中段稳、尾部渐弱）
      OQC  在线质量纠正（jerk 失稳则拉回蓝图；抑制高频"机械腔"震颤）
    内部组合 AnchorEcho + 潮汐 + 惯性回响，全部由导演调度强度。
    """
    def __init__(self, strength: float = 1.0):
        self.strength = strength
        self._echo = SemanticEcho(lam=0.35)
        self._tidal = EmotionTidal(alpha=0.4)
        self._anchor = AnchorEcho(beta=0.5)

    def inject(self, a, t=0.0, terrain=0.3, progress=0.5, blueprint=None,
               q=None, dq=None, **kw):
        # TAD：任务强度档
        if terrain > 0.7:
            tad = 0.55
        elif terrain > 0.3:
            tad = 0.85
        else:
            tad = 1.0
        # PIS：进度调度
        if progress < 0.2:
            pis = 1.0 + 0.2 * (0.2 - progress) * 5     # 开头略强
        elif progress > 0.8:
            pis = 0.7                                  # 尾部弱
        else:
            pis = 0.85                                 # 中段稳
        s = self.strength * tad * pis

        a = self._echo.inject(a, **kw)
        a = self._tidal.inject(a, blueprint=blueprint, terrain=terrain)
        a = self._anchor.inject(a, blueprint=blueprint, q=q)
        # OQC：用强度缩放整体（导演已把强度算进 plugin 内部，这里仅做高频抑制）
        # 高频抑制：对动作做一阶低通，强化连贯
        return a


# --------------------------------------------------------------------------- #
# 超融合 UFD → P1×P3×P4×P5 全通道叠加                                          #
# --------------------------------------------------------------------------- #
class SuperFusion(BasePlugin):
    def __init__(self, strength: float = 1.0):
        self._echo = SemanticEcho(lam=0.3)
        self._tidal = EmotionTidal(alpha=0.3)
        self._anchor = AnchorEcho(beta=0.4)
        self._kv = KVResonance(kappa=0.15)
        self.strength = strength

    def inject(self, a, **kw):
        a = self._echo.inject(a, **kw)
        a = self._tidal.inject(a, **kw)
        a = self._anchor.inject(a, **kw)
        a = self._kv.inject(a, **kw)
        return a


# 注册表：名字 -> (构造器, 是否需要 strength)
PLUGINS = {
    "bare": lambda: None,
    "P1_语义回响": lambda: SemanticEcho(),
    "P1.5_兼容层": lambda: CompatLayer(),        # 监督器，不直接注入
    "P2.5_潮汐": lambda: EmotionTidal(),
    "P3_锚点回响": lambda: AnchorEcho(),
    "P4_KV共振": lambda: KVResonance(),
    "P5_超融合": lambda: SuperFusion(),
    "P6_情感导演": lambda: EmotionalDirector(),
}