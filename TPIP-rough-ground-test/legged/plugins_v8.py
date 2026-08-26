#!/usr/bin/env python3
"""
plugins_v8.py — V8 GoldilocksFusion：金发茄门控 + 强参数

核心创新：jerk 门控为钟形曲线（中段开满，极高降级）
  - 低 jerk(平滑基座): 中等注入 → 突破 10%
  - 中 jerk: 全力注入 → 大幅提升
  - 极高 jerk(P2P跳变): 自动降级为仅KV → 避免 destabilize
"""
import numpy as np
from plugins import BasePlugin
from plugins_v2 import AdaptiveEcho, GatedTidal, AdaptiveKV


class GoldilocksFusion(BasePlugin):
    """
    V8 = V7 改进版：金发茄门控 + 强参数。
    钟形门控使平滑基座获得强注入(>10%)，同时极高 jerk 基座自动降级。
    """
    def __init__(self, lam=0.50, alpha=0.50, kappa=0.12,
                 j_lo=0.01, j_peak=0.05, j_hi=0.15):
        self._echo = AdaptiveEcho(lam=lam)
        self._tidal = GatedTidal(alpha=alpha)
        self._kv = AdaptiveKV(kappa=kappa)
        self.j_lo = j_lo      # 低于此 → 仅KV
        self.j_peak = j_peak  # 此处门控最大
        self.j_hi = j_hi      # 高于此 → 降级回仅KV
        self._jerk_hist = []
        self._last_a = None

    def _measure_jerk(self, a):
        if self._last_a is None:
            self._last_a = a.copy()
            return 0.0
        j = float(np.linalg.norm(a - self._last_a))
        self._last_a = a.copy()
        self._jerk_hist.append(j)
        if len(self._jerk_hist) > 20:
            self._jerk_hist.pop(0)
        if len(self._jerk_hist) < 3:
            return 0.0
        return float(np.mean(self._jerk_hist))

    def _goldilocks_gate(self, j):
        """钟形门控：j_lo~j_hi 之间为钟形，j_peak 处最大=1。"""
        if j < self.j_lo or j > self.j_hi:
            return 0.0
        # 钟形：左半升，右半降
        if j <= self.j_peak:
            # 上升段: j_lo → j_peak
            t = (j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)
            return float(np.sin(t * np.pi / 2))  # 0→1 平滑上升
        else:
            # 下降段: j_peak → j_hi
            t = (j - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)
            return float(np.cos(t * np.pi / 2))  # 1→0 平滑下降

    def inject(self, a, **kw):
        j = self._measure_jerk(a)
        gate = self._goldilocks_gate(j)
        gate_kv = 0.20  # KV 始终轻开

        a_orig = a.copy()
        # KV 始终开
        a = self._kv.inject(a, **kw)
        a = a_orig + gate_kv * (a - a_orig)

        # Echo+Tidal 由钟形门控控制
        if gate > 0.01:
            a_mid = a.copy()
            a_mid = self._echo.inject(a_mid, **kw)
            a_mid = self._tidal.inject(a_mid, **kw)
            a = a + gate * (a_mid - a)

        return a


PLUGINS_V8 = {
    "V8_GoldilocksFusion": lambda: GoldilocksFusion(),
}