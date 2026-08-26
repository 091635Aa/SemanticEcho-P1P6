#!/usr/bin/env python3
"""
stress_test9.py — 第九轮：相位锁定共振注入 + 频域净化 + 长池回响

基于 stress_test8 的发现：纯 GoldilocksFusion 级联7宽门已达 ceiling (~0.3258 avg)，
深度级联(9/11/13)反而下降；多项式在线优化会破坏周期步态。

本轮回顾根因：基座噪声+步态周期是耦合的，简单平滑会同时削弱两者。
新机制：
  V14 ResonantPhaseLocker (RPL) - 在线 FFT 找主频，注入相位对齐的纯净基波
  V15 SpectralPeakInjection   - 频域减噪：保留主频带，抑制其他
  V16 LongPoolEcho            - 长池回响(pool=100, 覆盖完整步态周期)
  V17 FrequencyMatchedTidal    - 低通截止频率=检测到的步态主频×2
  V18 CascadeStrong5          - 浅级联5趟但参数翻倍(lam=2,3,5)
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion
from plugins_v2 import AdaptiveEcho, GatedTidal, AdaptiveKV

SEEDS = [42, 137, 2024, 7777, 314159]


def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)


def eval_plugin(make_fn, family, seeds=SEEDS):
    opts = []
    for seed in seeds:
        base = BasePolicy(n_joints=6, family=family, seed=seed)
        sim_b = LeggedMicroSim(base, plugins=[], seed=seed)
        t = sim_b.run(goal=3.0, terrain=0.3)
        ci_b = compute_coherence(t['q'], t['dq'],
                                 central_diff(t['dq'], sim_b.dt),
                                 dt=sim_b.dt)['coherence_index']
        plug = make_fn()
        sim_p = LeggedMicroSim(base, plugins=[plug], seed=seed)
        t2 = sim_p.run(goal=3.0, terrain=0.3)
        ci_p = compute_coherence(t2['q'], t2['dq'],
                                 central_diff(t2['dq'], sim_p.dt),
                                 dt=sim_p.dt)['coherence_index']
        opts.append(opt(ci_p, ci_b))
    return float(np.mean(opts)), float(np.std(opts))


def run_round(round_num, name, make_fn):
    results = {}
    for fam in ['p2p', 'standard', 'transformer']:
        m, s = eval_plugin(make_fn, fam)
        results[fam] = {'mean': m, 'std': s}
    uni = all(results[f]['mean'] > 0 for f in ['p2p', 'standard', 'transformer'])
    results['universal'] = uni
    tag = "OK" if uni else "--"
    avg = (results['standard']['mean'] + results['transformer']['mean']) / 2
    print(f"  R{round_num}: {name:50s}  std={results['standard']['mean']:+.4f}  trans={results['transformer']['mean']:+.4f}  avg={avg:+.4f}  p2p={results['p2p']['mean']:+.4f}  {tag}")
    return results


# ──────────────────────────────────────────────────────────────────── #
#  V14: 相位锁定共振注入 (Resonant Phase Locker)                       #
# ──────────────────────────────────────────────────────────────────── #
class ResonantPhaseLocker(BasePlugin):
    """
    在线 FFT 找主频，生成相位对齐的纯净基波作为注入方向。
    关键：注入的是"周期成分"，不是"平滑均值"——保留步态，去噪。
    """
    def __init__(self, strength=0.5, window=128, dt=0.01,
                 j_lo=0.01, j_peak=0.1, j_hi=0.3):
        self.strength = strength
        self.window = window
        self.dt = dt
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []  # 每个关节一个时间序列
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

    def _gate(self, j):
        if j < self.j_lo or j > self.j_hi:
            return 0.0
        if j <= self.j_peak:
            t = (j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)
            return float(np.sin(t * np.pi / 2))
        else:
            t = (j - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)
            return float(np.cos(t * np.pi / 2))

    def _extract_periodic(self):
        """从历史中提取主频成分，预测下一步。"""
        if len(self._hist) < 16:
            return None
        H = np.array(self._hist)  # (T, n_joints)
        T = len(H)
        # FFT 每个关节
        # 取 0 频(直流)和最大幅度频率
        F = np.fft.rfft(H, axis=0)
        freqs = np.fft.rfftfreq(T, d=self.dt)
        # 忽略直流和极低频(<0.1Hz)
        mask = freqs > 0.1
        if not mask.any():
            return None
        # 找每关节最强频
        out = np.zeros(H.shape[1])
        for j in range(H.shape[1]):
            mag = np.abs(F[:, j]) * mask
            peak_idx = int(np.argmax(mag))
            if mag[peak_idx] < 1e-6:
                out[j] = H[-1, j]  # fallback: 用最近值
                continue
            # 主频成分 = 直流 + 主频
            dc = F[0, j] / T
            main = F[peak_idx, j] / T
            # 预测下一步相位：当前相位 = 2π f t_next
            t_next = T * self.dt
            phase = 2 * np.pi * freqs[peak_idx] * t_next
            recon = dc + 2 * (main.real * np.cos(phase) - main.imag * np.sin(phase))
            # 但 FFT 假设信号周期化，预测相位可能漂；用相位跟踪更稳
            # 简化：用主频重建当前时刻
            out[j] = float(recon)
        return out

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)
        j = self._measure_jerk(a)
        gate = self._gate(j)
        if gate < 0.01:
            return a
        periodic = self._extract_periodic()
        if periodic is None:
            return a
        return a + self.strength * gate * (periodic - a)


# ──────────────────────────────────────────────────────────────────── #
#  V15: 频域减噪 (Spectral Peak Injection)                            #
#  在 FFT 后只保留主频附近的窄带，反 FFT 得到净化轨迹                  #
# ──────────────────────────────────────────────────────────────────── #
class SpectralPeakInjection(BasePlugin):
    def __init__(self, strength=0.5, window=128, dt=0.01, band_hz=0.5,
                 j_lo=0.01, j_peak=0.1, j_hi=0.3):
        self.strength = strength
        self.window = window
        self.dt = dt
        self.band_hz = band_hz
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
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

    def _gate(self, j):
        if j < self.j_lo or j > self.j_hi:
            return 0.0
        if j <= self.j_peak:
            t = (j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)
            return float(np.sin(t * np.pi / 2))
        else:
            t = (j - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)
            return float(np.cos(t * np.pi / 2))

    def _spectral_denoise(self):
        if len(self._hist) < 16:
            return None
        H = np.array(self._hist)
        T = len(H)
        F = np.fft.rfft(H, axis=0)
        freqs = np.fft.rfftfreq(T, d=self.dt)
        # 找全局主频（所有关节平均幅度最大）
        avg_mag = np.mean(np.abs(F), axis=1)
        mask_dc = freqs > 0.1
        avg_mag_masked = avg_mag * mask_dc
        peak_idx = int(np.argmax(avg_mag_masked))
        peak_freq = freqs[peak_idx]
        # 保留 |f - peak_freq| < band_hz 的窄带
        mask_band = (np.abs(freqs - peak_freq) < self.band_hz) | (freqs < 0.05)
        F_filtered = F * mask_band[:, None]
        recon = np.fft.irfft(F_filtered, n=T, axis=0)
        return recon[-1]  # 取最近一步作为注入方向

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)
        j = self._measure_jerk(a)
        gate = self._gate(j)
        if gate < 0.01:
            return a
        periodic = self._spectral_denoise()
        if periodic is None:
            return a
        return a + self.strength * gate * (periodic - a)


# ──────────────────────────────────────────────────────────────────── #
#  V16: 长池回响 (Long Pool Echo)                                      #
#  pool=100 覆盖完整步态周期，捕捉真实步态均值                          #
# ──────────────────────────────────────────────────────────────────── #
class LongPoolEchoFusion(BasePlugin):
    def __init__(self, lam=1.0, alpha=1.0, kappa=0.12, pool=100,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        self._echo = AdaptiveEcho(lam=lam, pool=pool)
        self._tidal = GatedTidal(alpha=alpha)
        self._kv = AdaptiveKV(kappa=kappa)
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
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

    def _gate(self, j):
        if j < self.j_lo or j > self.j_hi:
            return 0.0
        if j <= self.j_peak:
            t = (j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)
            return float(np.sin(t * np.pi / 2))
        else:
            t = (j - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)
            return float(np.cos(t * np.pi / 2))

    def inject(self, a, **kw):
        j = self._measure_jerk(a)
        gate = self._gate(j)
        gate_kv = 0.20
        a_orig = a.copy()
        a = self._kv.inject(a, **kw)
        a = a_orig + gate_kv * (a - a_orig)
        if gate > 0.01:
            a_mid = a.copy()
            a_mid = self._echo.inject(a_mid, **kw)
            a_mid = self._tidal.inject(a_mid, **kw)
            a = a + gate * (a_mid - a)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  V18: 浅级联 + 强参数                                                 #
# ──────────────────────────────────────────────────────────────────── #
class Cascade(BasePlugin):
    def __init__(self, configs):
        self.passes = [GoldilocksFusion(**c) for c in configs]
    def inject(self, a, **kw):
        for p in self.passes:
            a = p.inject(a, **kw)
        return a


class LongPoolCascade(BasePlugin):
    def __init__(self, configs):
        self.passes = [LongPoolEchoFusion(**c) for c in configs]
    def inject(self, a, **kw):
        for p in self.passes:
            a = p.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  组合：Cascade × RPL                                                 #
# ──────────────────────────────────────────────────────────────────── #
class CascadePlusRPL(BasePlugin):
    def __init__(self, configs, rpl_cfg=None):
        self.cascade = Cascade(configs)
        rpl_cfg = rpl_cfg or {}
        self.rpl = ResonantPhaseLocker(**rpl_cfg)
    def inject(self, a, **kw):
        a = self.cascade.inject(a, **kw)
        a = self.rpl.inject(a, **kw)
        return a


def main():
    print("=" * 120)
    print("第九轮：相位锁定共振注入 + 频域净化 + 长池回响")
    print("=" * 120)
    all_r = {}
    wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}

    # ── 对照：上轮最优 ──
    all_r['R0_cascade7_wide_ref'] = run_round(0, "级联7宽门(对照)",
        lambda: Cascade([wide]*7))

    # ── V14 RPL 单独 ──
    all_r['R1_rpl_s05'] = run_round(1, "RPL 单独 strength=0.5",
        lambda: ResonantPhaseLocker(strength=0.5, window=128,
                                     j_lo=0.01, j_peak=0.1, j_hi=0.3))
    all_r['R2_rpl_s08_wide'] = run_round(2, "RPL 单独 宽门 strength=0.8",
        lambda: ResonantPhaseLocker(strength=0.8, window=128,
                                     j_lo=0.01, j_peak=0.15, j_hi=0.5))
    all_r['R3_rpl_s10_xwide'] = run_round(3, "RPL 单独 极宽门 strength=1.0",
        lambda: ResonantPhaseLocker(strength=1.0, window=256,
                                     j_lo=0.01, j_peak=0.20, j_hi=0.8))

    # ── V15 频域减噪 单独 ──
    all_r['R4_spectral_s05_wide'] = run_round(4, "频域减噪 宽门 strength=0.5",
        lambda: SpectralPeakInjection(strength=0.5, window=128, band_hz=0.5,
                                       j_lo=0.01, j_peak=0.15, j_hi=0.5))
    all_r['R5_spectral_s08_wide'] = run_round(5, "频域减噪 宽门 strength=0.8",
        lambda: SpectralPeakInjection(strength=0.8, window=128, band_hz=0.5,
                                       j_lo=0.01, j_peak=0.15, j_hi=0.5))

    # ── V16 长池回响 单独 ──
    all_r['R6_longpool_5_wide'] = run_round(6, "长池回响 宽门 pool=100",
        lambda: LongPoolEchoFusion(lam=1.0, alpha=1.0, kappa=0.12, pool=100,
                                  j_lo=0.01, j_peak=0.15, j_hi=0.5))

    # ── V16 长池级联 ──
    all_r['R7_longpool_cascade5_wide'] = run_round(7, "长池级联5 宽门 pool=100",
        lambda: LongPoolCascade([{'lam': 1.0, 'alpha': 1.0, 'kappa': 0.12, 'pool': 100,
                                  'j_lo': 0.01, 'j_peak': 0.15, 'j_hi': 0.5}]*5))
    all_r['R8_longpool_cascade7_wide'] = run_round(8, "长池级联7 宽门 pool=100",
        lambda: LongPoolCascade([{'lam': 1.0, 'alpha': 1.0, 'kappa': 0.12, 'pool': 100,
                                  'j_lo': 0.01, 'j_peak': 0.15, 'j_hi': 0.5}]*7))

    # ── V18 浅级联强参数 ──
    all_r['R9_cascade5_lam2'] = run_round(9, "级联5 宽门 lam=2.0",
        lambda: Cascade([{'lam': 2.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}]*5))
    all_r['R10_cascade5_lam3'] = run_round(10, "级联5 宽门 lam=3.0",
        lambda: Cascade([{'lam': 3.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}]*5))
    all_r['R11_cascade5_lam2_alpha2'] = run_round(11, "级联5 宽门 lam=2 alpha=2",
        lambda: Cascade([{'lam': 2.0, 'alpha': 2.0, 'j_peak': 0.15, 'j_hi': 0.5}]*5))
    all_r['R12_cascade7_lam2_alpha2'] = run_round(12, "级联7 宽门 lam=2 alpha=2",
        lambda: Cascade([{'lam': 2.0, 'alpha': 2.0, 'j_peak': 0.15, 'j_hi': 0.5}]*7))

    # ── Cascade + RPL 组合 ──
    all_r['R13_cascade7_rpl_s08'] = run_round(13, "级联7宽门 + RPL(0.8宽门)",
        lambda: CascadePlusRPL(
            [wide]*7,
            rpl_cfg={'strength': 0.8, 'window': 128,
                     'j_lo': 0.01, 'j_peak': 0.15, 'j_hi': 0.5}))
    all_r['R14_cascade7_rpl_s10_xwide'] = run_round(14, "级联7宽门 + RPL(1.0极宽门)",
        lambda: CascadePlusRPL(
            [wide]*7,
            rpl_cfg={'strength': 1.0, 'window': 256,
                     'j_lo': 0.01, 'j_peak': 0.20, 'j_hi': 0.8}))

    # ── 长池级联 + RPL ──
    class LongPoolCascadePlusRPL(BasePlugin):
        def __init__(self):
            self.cascade = LongPoolCascade([{'lam': 1.0, 'alpha': 1.0, 'kappa': 0.12, 'pool': 100,
                                             'j_lo': 0.01, 'j_peak': 0.15, 'j_hi': 0.5}]*7)
            self.rpl = ResonantPhaseLocker(strength=1.0, window=128,
                                            j_lo=0.01, j_peak=0.15, j_hi=0.5)
        def inject(self, a, **kw):
            a = self.cascade.inject(a, **kw)
            a = self.rpl.inject(a, **kw)
            return a
    all_r['R15_longpool_cascade7_rpl'] = run_round(15, "长池级联7宽门 + RPL(1.0宽门)",
        lambda: LongPoolCascadePlusRPL())

    # ── RPL + Cascade 顺序反转 ──
    class RPLPlusCascade(BasePlugin):
        def __init__(self):
            self.rpl = ResonantPhaseLocker(strength=0.8, window=128,
                                            j_lo=0.01, j_peak=0.15, j_hi=0.5)
            self.cascade = Cascade([wide]*7)
        def inject(self, a, **kw):
            a = self.rpl.inject(a, **kw)
            a = self.cascade.inject(a, **kw)
            return a
    all_r['R16_rpl_then_cascade7'] = run_round(16, "RPL(0.8宽门) + 级联7宽门",
        lambda: RPLPlusCascade())

    # ── 汇总 ──
    print(f"\n{'='*120}")
    print("汇总（按 std+trans 均值排序）")
    print(f"{'='*120}")
    sorted_r = sorted(all_r.items(),
                      key=lambda x: (x[1]['standard']['mean'] + x[1]['transformer']['mean']) / 2,
                      reverse=True)
    print(f"  {'配置':35s}  {'std':>8s}  {'trans':>8s}  {'avg':>8s}  {'p2p':>8s}  通用?")
    print("  " + "-" * 85)
    for k, v in sorted_r:
        uni = v.get('universal', False)
        avg = (v['standard']['mean'] + v['transformer']['mean']) / 2
        print(f"  {k:35s}  {v['standard']['mean']:+.4f}  {v['transformer']['mean']:+.4f}  {avg:+.4f}  {v['p2p']['mean']:+.4f}  {'YES' if uni else 'NO'}")

    best = sorted_r[0]
    print(f"\n  最优: {best[0]} → avg={(best[1]['standard']['mean']+best[1]['transformer']['mean'])/2:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "stress_test9.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
