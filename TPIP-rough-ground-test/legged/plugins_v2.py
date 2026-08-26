#!/usr/bin/env python3
"""
plugins_v2.py — 自适应改进版插件（自主科研第二轮）

针对第一轮发现的三个根因做改进：
  1. 注入方向改为"基座自身惯性主方向"（不注入外部正弦蓝图）
  2. 兼容层做前置门槛：jerk 高时才注入，平滑时自动归零
  3. 乘性注入加截止频率约束（低通滤波，抑制高频放大）

新增方案：
  V1_AdaptiveEcho   — 自适应惯性回响（P1改进版）
  V2_GatedTidal      — 门控潮汐（P2.5改进版，乘性+截止频率）
  V3_SelfAnchored    — 自锚定（P3改进版，用自身轨迹流形而非外部蓝图）
  V4_AdaptiveKV      — 自适应KV共振（P4改进版，只放大与自身历史一致的方向）
  V5_SmartFusion     — 智能融合（V1×V2×V3×V4 全自适应叠加）
  V6_AdaptiveDirector — 自适应导演（P6改进版，TAD基于实测jerk而非地形猜测）
"""

from __future__ import annotations
import numpy as np
from plugins import BasePlugin


# ──────────────────────────────────────────────────────────────────── #
#  改进1：自身惯性主方向注入（不依赖外部蓝图）                         #
# ──────────────────────────────────────────────────────────────────── #
class AdaptiveEcho(BasePlugin):
    """
    V1 = P1 改进版：回响池质心注入，但注入方向=基座自身历史主方向。
    不向外部正弦蓝图推，而是向"自己最近 N 步的加权平均"推。
    对 P2P：平滑掉高频震颤（向自己慢变趋势回归）。
    对 standard：轻微强化已有惯性（向自己的低频主方向加强）。
    """
    def __init__(self, lam=0.3, pool=15):
        self.lam = lam
        self.pool = pool
        self._hist = []

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.pool:
            self._hist.pop(0)
        if len(self._hist) < 3:
            return a
        H = np.array(self._hist)
        # 自身惯性主方向 = 最近趋势（指数加权均值）
        w = np.exp(-0.3 * np.arange(len(H))[::-1])
        w /= w.sum()
        trend = (w[:, None] * H).sum(axis=0)
        return a + self.lam * (trend - a)


# ──────────────────────────────────────────────────────────────────── #
#  改进2：门控潮汐（乘性重加权 + 截止频率约束）                        #
# ──────────────────────────────────────────────────────────────────── #
class GatedTidal(BasePlugin):
    """
    V2 = P2.5 改进版：乘性重加权向"低通滤波后的自身轨迹"靠拢。
    关键改进：先对历史做低通（截止频率 fc），再乘性注入，
    避免放大高频分量。
    """
    def __init__(self, alpha=0.3, fc=3.0):
        self.alpha = alpha
        self.fc = fc  # 截止频率 Hz
        self._hist = []

    def _lowpass(self, a):
        self._hist.append(a.copy())
        if len(self._hist) > 30:
            self._hist.pop(0)
        if len(self._hist) < 5:
            return a
        H = np.array(self._hist)
        # 简单一阶低通：y[n] = y[n-1] + dt*fc*(x[n]-y[n-1])
        y = H[0].copy()
        dt = 0.01
        rc = 1.0 / (2 * np.pi * self.fc)
        for x in H[1:]:
            y = y + dt / (rc + dt) * (x - y)
        return y

    def inject(self, a, **kw):
        smooth = self._lowpass(a)
        return (1 - self.alpha) * a + self.alpha * smooth


# ──────────────────────────────────────────────────────────────────── #
#  改进3：自锚定（用自身轨迹流形做锚点，不注入外部蓝图）                  #
# ──────────────────────────────────────────────────────────────────── #
class SelfAnchored(BasePlugin):
    """
    V3 = P3 改进版：锚点 = 基座自身最近一个完整步态周期的平均形态。
    打分 = 当前动作与自身历史模板的相位一致性。
    tanh 有界加性注入，但方向来自自身。
    """
    def __init__(self, beta=0.4, cycle=50):
        self.beta = beta
        self.cycle = cycle
        self._hist = []

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.cycle * 3:
            self._hist = self._hist[-self.cycle * 3:]
        if len(self._hist) < self.cycle:
            return a
        # 自身历史模板 = 最近一个周期的均值
        template = np.mean(np.array(self._hist[-self.cycle:]), axis=0)
        # 相位一致性打分
        d = np.dot(a, template) / (np.linalg.norm(a) * np.linalg.norm(template) + 1e-9)
        bias = np.tanh(2.0 * d) * template
        return a + self.beta * (bias - a)


# ──────────────────────────────────────────────────────────────────── #
#  改进4：自适应KV共振（只放大与自身历史一致的方向）                     #
# ──────────────────────────────────────────────────────────────────── #
class AdaptiveKV(BasePlugin):
    """
    V4 = P4 改进版：对历史缓存做加权，权重=与当前动作的自相似度。
    不用外部蓝图做打分，而是用自身轨迹的内在一致性。
    """
    def __init__(self, kappa=0.15, mem=12):
        self.kappa = kappa
        self.mem = mem
        self._hist = []

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.mem:
            self._hist.pop(0)
        if len(self._hist) < 4:
            return a
        H = np.array(self._hist)
        # 自相似度：每步与当前动作的余弦
        scores = H @ a / (np.linalg.norm(H, axis=1) * np.linalg.norm(a) + 1e-9)
        w = np.exp(self.kappa * scores)
        w /= w.sum()
        attended = (w[:, None] * H).sum(axis=0)
        # 轻度拉向"与自身一致的方向"
        return 0.85 * a + 0.15 * attended


# ──────────────────────────────────────────────────────────────────── #
#  改进5：智能融合（V1×V2×V3×V4 全自适应叠加）                       #
# ──────────────────────────────────────────────────────────────────── #
class SmartFusion(BasePlugin):
    """V5 = V1×V2×V3×V4 全自适应叠加。"""
    def __init__(self):
        self._echo = AdaptiveEcho(lam=0.2)
        self._tidal = GatedTidal(alpha=0.2)
        self._anchor = SelfAnchored(beta=0.3)
        self._kv = AdaptiveKV(kappa=0.1)

    def inject(self, a, **kw):
        a = self._echo.inject(a, **kw)
        a = self._tidal.inject(a, **kw)
        a = self._anchor.inject(a, **kw)
        a = self._kv.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  改进6：自适应导演（TAD 基于实测 jerk 而非地形猜测）                  #
# ──────────────────────────────────────────────────────────────────── #
class AdaptiveDirector(BasePlugin):
    """
    V6 = P6 改进版：
      TAD: 实时测量 jerk，jerk 高→强度大，jerk 低→自动归零
      PIS: 进度调度保留
      OQC: 高频抑制（低通）+ 相位漂移拉回（向自身趋势）
    """
    def __init__(self, base_strength=0.5):
        self.base_strength = base_strength
        self._echo = AdaptiveEcho(lam=0.2)
        self._tidal = GatedTidal(alpha=0.15)
        self._jerk_hist = []
        self._last_a = None

    def _measure_jerk(self, a):
        if self._last_a is None:
            self._last_a = a.copy()
            return 0.0
        jerk = np.linalg.norm(a - self._last_a)
        self._last_a = a.copy()
        self._jerk_hist.append(jerk)
        if len(self._jerk_hist) > 20:
            self._jerk_hist.pop(0)
        if len(self._jerk_hist) < 3:
            return 0.0
        return float(np.mean(self._jerk_hist))

    def inject(self, a, t=0.0, terrain=0.3, progress=0.5, **kw):
        j = self._measure_jerk(a)
        # TAD: jerk > 0.05 → 全力; jerk < 0.01 → 归零
        tad = np.clip((j - 0.01) / 0.04, 0.0, 1.0)
        # PIS
        if progress < 0.15:
            pis = 1.1
        elif progress > 0.85:
            pis = 0.7
        else:
            pis = 0.9
        s = self.base_strength * tad * pis
        # 内部插件用 strength 缩放
        a_orig = a.copy()
        a = self._echo.inject(a, **kw)
        a = self._tidal.inject(a, **kw)
        return a_orig + s * (a - a_orig)


PLUGINS_V2 = {
    "V1_AdaptiveEcho": lambda: AdaptiveEcho(),
    "V2_GatedTidal": lambda: GatedTidal(),
    "V3_SelfAnchored": lambda: SelfAnchored(),
    "V4_AdaptiveKV": lambda: AdaptiveKV(),
    "V5_SmartFusion": lambda: SmartFusion(),
    "V6_AdaptiveDirector": lambda: AdaptiveDirector(),
}