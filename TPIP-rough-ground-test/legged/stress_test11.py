#!/usr/bin/env python3
"""
stress_test11.py — 第十一轮：纯潮汐级联 + 自适应低通 + 残差噪声抑制

发现（stress_test10）：
- 蓝图锚定反而下降（蓝图幅度 0.25 ≠ 实际步态 0.35/0.40）
- 级联7宽门仍是 ceiling (~0.3258)

新机制（本轮）：
  V24 PureTidalCascade   - 仅 Tidal 级联，多阶低通
  V25 AdaptiveLowpass   - fc 在线跟踪检测到的主频
  V26 ResidualNoiseCancel - 残差(=a-trend) 中的低频部分减掉
  V27 PhaseMatchedBlueprint - 用蓝图相位 + 在线估计幅度
  V28 EchoOnlyCascade   - 仅 Echo 级联（保中位趋势）
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


def measure_jerk(hist_obj, a):
    """通用 jerk 测量。hist_obj 需有 _last_a 和 _jerk_hist 属性。"""
    if hist_obj._last_a is None:
        hist_obj._last_a = a.copy()
        return 0.0
    j = float(np.linalg.norm(a - hist_obj._last_a))
    hist_obj._last_a = a.copy()
    hist_obj._jerk_hist.append(j)
    if len(hist_obj._jerk_hist) > 20:
        hist_obj._jerk_hist.pop(0)
    if len(hist_obj._jerk_hist) < 3:
        return 0.0
    return float(np.mean(hist_obj._jerk_hist))


def goldilocks_gate(j, j_lo, j_peak, j_hi):
    if j < j_lo or j > j_hi:
        return 0.0
    if j <= j_peak:
        t = (j - j_lo) / (j_peak - j_lo + 1e-9)
        return float(np.sin(t * np.pi / 2))
    else:
        t = (j - j_peak) / (j_hi - j_peak + 1e-9)
        return float(np.cos(t * np.pi / 2))


# ──────────────────────────────────────────────────────────────────── #
#  V24: 纯 Tidal 级联（每 pass 一个一阶低通，N pass = N 阶）            #
# ──────────────────────────────────────────────────────────────────── #
class PureTidalCascade(BasePlugin):
    """
    仅用 GatedTidal 级联，跳过 Echo/KV。
    N 趟级联 = N 阶低通，滚降更陡。
    """
    def __init__(self, n_passes=7, alpha=1.0, fc=3.0,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        self.passes = [GatedTidal(alpha=alpha, fc=fc) for _ in range(n_passes)]
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._jerk_hist = []
        self._last_a = None

    def inject(self, a, **kw):
        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a
        a_orig = a.copy()
        for p in self.passes:
            a = p.inject(a, **kw)
        # 门控融合：gate=1 时完全用 tidal 输出
        return a_orig + gate * (a - a_orig)


# ──────────────────────────────────────────────────────────────────── #
#  V25: 自适应低通（fc 在线跟踪主频×2）                                #
# ──────────────────────────────────────────────────────────────────── #
class AdaptiveLowpass(BasePlugin):
    """
    在线 FFT 检测主频，设 fc = main_freq * 2。
    保证 gait 主频通过，更高频被滤掉。
    """
    def __init__(self, alpha=1.0, window=128, dt=0.01, fc_mult=2.0,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        self.alpha = alpha
        self.window = window
        self.dt = dt
        self.fc_mult = fc_mult
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None
        self._fc = 3.0  # 初始
        self._y = None  # 低通状态
        self._update_counter = 0

    def _detect_main_freq(self):
        if len(self._hist) < 32:
            return None
        H = np.array(self._hist)
        T = len(H)
        # 取幅度最大的关节
        sig = H[:, 0]
        F = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(T, d=self.dt)
        mask = freqs > 0.1
        if not mask.any():
            return None
        mag = np.abs(F) * mask
        peak_idx = int(np.argmax(mag))
        if mag[peak_idx] < 1e-6:
            return None
        return float(freqs[peak_idx])

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)

        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a

        # 定期更新 fc
        self._update_counter += 1
        if self._update_counter % 16 == 0:
            f_main = self._detect_main_freq()
            if f_main is not None:
                self._fc = max(1.0, f_main * self.fc_mult)

        # 一阶低通
        if self._y is None:
            self._y = a.copy()
        else:
            rc = 1.0 / (2 * np.pi * self._fc)
            alpha_lp = self.dt / (rc + self.dt)
            self._y = self._y + alpha_lp * (a - self._y)

        # 门控融合
        return a + self.alpha * gate * (self._y - a)


# ──────────────────────────────────────────────────────────────────── #
#  V26: 残差噪声抑制                                                    #
#  residual = a - trend (trend=长池加权中位)                            #
#  residual 中"慢变"部分 = drift noise，减掉                            #
# ──────────────────────────────────────────────────────────────────── #
class ResidualNoiseCancel(BasePlugin):
    def __init__(self, pool=50, drift_fc=0.5, dt=0.01, cancel_strength=1.0,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        self.pool = pool
        self.drift_fc = drift_fc
        self.dt = dt
        self.cancel_strength = cancel_strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None
        self._drift_y = None

    def _weighted_median(self, H):
        n = len(H)
        w = np.exp(-0.3 * np.arange(n)[::-1])
        w /= w.sum()
        trend = np.zeros(H.shape[1])
        for j in range(H.shape[1]):
            col = H[:, j]
            order = np.argsort(col)
            cum = np.cumsum(w[order])
            idx = int(np.searchsorted(cum, 0.5))
            trend[j] = col[order[min(idx, n-1)]]
        return trend

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.pool:
            self._hist.pop(0)

        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a

        if len(self._hist) < 5:
            return a

        H = np.array(self._hist)
        trend = self._weighted_median(H)
        residual = a - trend

        # 残差低通 = 慢变 drift 噪声
        if self._drift_y is None:
            self._drift_y = residual.copy()
        else:
            rc = 1.0 / (2 * np.pi * self.drift_fc)
            alpha_lp = self.dt / (rc + self.dt)
            self._drift_y = self._drift_y + alpha_lp * (residual - self._drift_y)

        # 减掉 drift 噪声
        return a - self.cancel_strength * gate * self._drift_y


# ──────────────────────────────────────────────────────────────────── #
#  V27: 相位匹配蓝图                                                    #
#  用蓝图相位 + 在线估计的幅度                                          #
# ──────────────────────────────────────────────────────────────────── #
class PhaseMatchedBlueprint(BasePlugin):
    def __init__(self, strength=1.0, amp_window=100, dt=0.01,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        self.strength = strength
        self.amp_window = amp_window
        self.dt = dt
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None
        self._amp_est = 0.3  # 初始估计
        self._phase_offset = np.zeros(6)  # 初始相位偏移
        self._update_counter = 0

    def _estimate_amp_phase(self, blueprint):
        """用 blueprint 相位作为参考，估计实际幅度和相位偏移。"""
        if len(self._hist) < 32:
            return
        H = np.array(self._hist)
        # 用蓝图相位作为参考信号
        # 实际信号 = amp * sin(blueprint_phase + phase_offset)
        # 用最小二乘拟合 amp 和 phase_offset
        # 对每个关节，residual = a - mean(a) ≈ amp * sin(phi + offset)
        # a * cos(phi) 和 a * sin(phi) 的均值可解出 (amp*cos(offset), amp*sin(offset))
        b = np.asarray(blueprint, dtype=float) if blueprint is not None else None
        if b is None:
            return
        # 历史蓝图（重新生成）
        T = len(H)
        t_arr = np.arange(T) * self.dt
        # 假设 blueprint 主频 = 1.0
        for j in range(H.shape[1]):
            phi = 2 * np.pi * 1.0 * t_arr + np.linspace(0, np.pi, H.shape[1])[j]
            x = H[:, j] - np.mean(H[:, j])
            I = np.mean(x * np.sin(phi))
            Q = np.mean(x * np.cos(phi))
            amp = float(np.sqrt(I*I + Q*Q))
            if amp > 0.05:
                self._amp_est = 0.9 * self._amp_est + 0.1 * amp
                self._phase_offset[j] = float(np.arctan2(Q, I))

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.amp_window:
            self._hist.pop(0)

        bp = kw.get('blueprint', None)
        if bp is None:
            return a

        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a

        # 定期更新幅度估计
        self._update_counter += 1
        if self._update_counter % 32 == 0:
            self._estimate_amp_phase(bp)

        bp = np.asarray(bp, dtype=float)
        # 蓝图归一化（蓝图幅度=0.25），缩放到估计幅度
        bp_norm = np.linalg.norm(bp)
        if bp_norm > 1e-6:
            bp_normalized = bp / bp_norm
        else:
            bp_normalized = bp
        # 目标 = estimated_amp * (bp / 0.25)  但需保持方向
        # 实际上：bp 已经是 0.25*sin(phi+offset_bp)，我们要 amp*sin(phi+offset_actual)
        # 简化：用 bp 方向 × 估计幅度
        target = bp_normalized * self._amp_est
        return a + self.strength * gate * (target - a)


# ──────────────────────────────────────────────────────────────────── #
#  V28: 仅 Echo 级联（保中位趋势）                                      #
# ──────────────────────────────────────────────────────────────────── #
class EchoOnlyCascade(BasePlugin):
    def __init__(self, n_passes=7, lam=1.0, pool=15,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        self.passes = [AdaptiveEcho(lam=lam, pool=pool) for _ in range(n_passes)]
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._jerk_hist = []
        self._last_a = None

    def inject(self, a, **kw):
        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a
        a_orig = a.copy()
        for p in self.passes:
            a = p.inject(a, **kw)
        return a_orig + gate * (a - a_orig)


# ──────────────────────────────────────────────────────────────────── #
#  组合：Adaptive Lowpass + Echo Cascade                               #
# ──────────────────────────────────────────────────────────────────── #
class ALPlusEchoCascade(BasePlugin):
    def __init__(self, n_echo=5, fc_mult=2.0, lam=1.0):
        self.al = AdaptiveLowpass(alpha=1.0, fc_mult=fc_mult)
        self.echo = EchoOnlyCascade(n_passes=n_echo, lam=lam, pool=15)
    def inject(self, a, **kw):
        a = self.al.inject(a, **kw)
        a = self.echo.inject(a, **kw)
        return a


def main():
    print("=" * 120)
    print("第十一轮：纯潮汐级联 + 自适应低通 + 残差噪声抑制 + 相位匹配蓝图")
    print("=" * 120)
    all_r = {}
    wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}

    # ── 对照 ──
    class RefCascade(BasePlugin):
        def __init__(self):
            self.passes = [GoldilocksFusion(**wide) for _ in range(7)]
        def inject(self, a, **kw):
            for p in self.passes:
                a = p.inject(a, **kw)
            return a
    all_r['R0_cascade7_wide_ref'] = run_round(0, "级联7宽门(对照)",
        lambda: RefCascade())

    # ── V24 纯 Tidal 级联 ──
    all_r['R1_pure_tidal_cascade5_fc3'] = run_round(1, "纯Tidal级联5 fc=3",
        lambda: PureTidalCascade(n_passes=5, alpha=1.0, fc=3.0))
    all_r['R2_pure_tidal_cascade7_fc3'] = run_round(2, "纯Tidal级联7 fc=3",
        lambda: PureTidalCascade(n_passes=7, alpha=1.0, fc=3.0))
    all_r['R3_pure_tidal_cascade7_fc5'] = run_round(3, "纯Tidal级联7 fc=5",
        lambda: PureTidalCascade(n_passes=7, alpha=1.0, fc=5.0))
    all_r['R4_pure_tidal_cascade7_fc10'] = run_round(4, "纯Tidal级联7 fc=10",
        lambda: PureTidalCascade(n_passes=7, alpha=1.0, fc=10.0))
    all_r['R5_pure_tidal_cascade7_fc15'] = run_round(5, "纯Tidal级联7 fc=15",
        lambda: PureTidalCascade(n_passes=7, alpha=1.0, fc=15.0))

    # ── V25 自适应低通 ──
    all_r['R6_adaptive_lp_x2'] = run_round(6, "自适应低通 fc=主频×2",
        lambda: AdaptiveLowpass(alpha=1.0, fc_mult=2.0))
    all_r['R7_adaptive_lp_x3'] = run_round(7, "自适应低通 fc=主频×3",
        lambda: AdaptiveLowpass(alpha=1.0, fc_mult=3.0))
    all_r['R8_adaptive_lp_x1_5'] = run_round(8, "自适应低通 fc=主频×1.5",
        lambda: AdaptiveLowpass(alpha=1.0, fc_mult=1.5))

    # ── V26 残差噪声抑制 ──
    all_r['R9_residual_cancel_s05'] = run_round(9, "残差噪声抑制 strength=0.5",
        lambda: ResidualNoiseCancel(pool=50, drift_fc=0.5, cancel_strength=0.5))
    all_r['R10_residual_cancel_s10'] = run_round(10, "残差噪声抑制 strength=1.0",
        lambda: ResidualNoiseCancel(pool=50, drift_fc=0.5, cancel_strength=1.0))
    all_r['R11_residual_cancel_s20'] = run_round(11, "残差噪声抑制 strength=2.0",
        lambda: ResidualNoiseCancel(pool=50, drift_fc=0.5, cancel_strength=2.0))
    all_r['R12_residual_cancel_s10_fc02'] = run_round(12, "残差噪声抑制 s=1.0 fc=0.2",
        lambda: ResidualNoiseCancel(pool=80, drift_fc=0.2, cancel_strength=1.0))

    # ── V27 相位匹配蓝图 ──
    all_r['R13_phase_matched_s10'] = run_round(13, "相位匹配蓝图 strength=1.0",
        lambda: PhaseMatchedBlueprint(strength=1.0))
    all_r['R14_phase_matched_s15'] = run_round(14, "相位匹配蓝图 strength=1.5",
        lambda: PhaseMatchedBlueprint(strength=1.5))

    # ── V28 仅 Echo 级联 ──
    all_r['R15_echo_only_cascade5'] = run_round(15, "仅Echo级联5",
        lambda: EchoOnlyCascade(n_passes=5, lam=1.0))
    all_r['R16_echo_only_cascade7'] = run_round(16, "仅Echo级联7",
        lambda: EchoOnlyCascade(n_passes=7, lam=1.0))

    # ── 组合：自适应低通 + Echo 级联 ──
    all_r['R17_al_x2_echo5'] = run_round(17, "自适应低通×2 + Echo级联5",
        lambda: ALPlusEchoCascade(n_echo=5, fc_mult=2.0, lam=1.0))
    all_r['R18_al_x2_echo7'] = run_round(18, "自适应低通×2 + Echo级联7",
        lambda: ALPlusEchoCascade(n_echo=7, fc_mult=2.0, lam=1.0))

    # ── 组合：自适应低通 + 残差噪声抑制 ──
    class ALPlusResidualCancel(BasePlugin):
        def __init__(self, fc_mult=2.0, cancel_strength=1.0):
            self.al = AdaptiveLowpass(alpha=1.0, fc_mult=fc_mult)
            self.rc = ResidualNoiseCancel(pool=50, drift_fc=0.5,
                                          cancel_strength=cancel_strength)
        def inject(self, a, **kw):
            a = self.al.inject(a, **kw)
            a = self.rc.inject(a, **kw)
            return a
    all_r['R19_al_x2_residual_s10'] = run_round(19, "自适应低通×2 + 残差抑制s=1.0",
        lambda: ALPlusResidualCancel(fc_mult=2.0, cancel_strength=1.0))

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

    out = os.path.join(os.path.dirname(__file__), "stress_test11.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
