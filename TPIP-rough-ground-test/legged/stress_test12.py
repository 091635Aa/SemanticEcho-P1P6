#!/usr/bin/env python3
"""
stress_test12.py — 第十二轮：谱减法 + 维纳滤波 + 中值滤波

根因分析：
  - cascade7 已将 a 的噪声从 0.08→0.006 (13x reduction)
  - 但 rms_jerk ∝ noise/dt^1.5 仍很高 (1323)，s_smooth=0.17
  - 理论 floor：noise≈0.0005 时 s_smooth≈0.85, CI≈0.65
  - 一阶低通已到极限，需更先进降噪

新机制：
  V29 SpectralSubtraction - 滑窗 FFT，估计噪声底，软减
  V30 WienerFilter        - 频域最优滤波
  V31 MedianFilter        - 中值滤波，杀脉冲噪声
  V32 SpectralSubPlusCascade - 谱减 + 级联
  V33 CascadePlusWiener   - 级联后 Wiener 精修
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


def measure_jerk(obj, a):
    if obj._last_a is None:
        obj._last_a = a.copy()
        return 0.0
    j = float(np.linalg.norm(a - obj._last_a))
    obj._last_a = a.copy()
    obj._jerk_hist.append(j)
    if len(obj._jerk_hist) > 20:
        obj._jerk_hist.pop(0)
    if len(obj._jerk_hist) < 3:
        return 0.0
    return float(np.mean(obj._jerk_hist))


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
#  V29: 谱减法 (Spectral Subtraction)                                  #
#  滑窗 FFT，估计噪声底，软减                                          #
# ──────────────────────────────────────────────────────────────────── #
class SpectralSubtraction(BasePlugin):
    def __init__(self, strength=1.0, window=128, dt=0.01, noise_alpha=2.0,
                 j_lo=0.005, j_peak=0.15, j_hi=0.5):
        self.strength = strength
        self.window = window
        self.dt = dt
        self.noise_alpha = noise_alpha
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None
        self._noise_floor = None

    def _spectral_subtract(self):
        if len(self._hist) < 32:
            return None
        H = np.array(self._hist)
        T = len(H)
        F = np.fft.rfft(H, axis=0)
        mag = np.abs(F)
        # 噪声底估计：每个频率 bin 取时间下分位数
        # 简化：用所有 bin 的中位数 * alpha
        noise_floor = np.median(mag, axis=1) * self.noise_alpha
        noise_floor = noise_floor[:, None]  # broadcast to (n_freq, 1)
        # 软减: |F_clean| = max(|F| - noise_floor, 0)
        mag_clean = np.maximum(mag - noise_floor, 0)
        # 保持相位
        phase = F / (mag + 1e-9)
        F_clean = mag_clean * phase
        recon = np.fft.irfft(F_clean, n=T, axis=0)
        return recon[-1]  # 取最近一步

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)
        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a
        cleaned = self._spectral_subtract()
        if cleaned is None:
            return a
        return a + self.strength * gate * (cleaned - a)


# ──────────────────────────────────────────────────────────────────── #
#  V30: 维纳滤波 (Wiener Filter)                                       #
#  F_clean = F * |F|² / (|F|² + σ²)                                    #
# ──────────────────────────────────────────────────────────────────── #
class WienerFilter(BasePlugin):
    def __init__(self, strength=1.0, window=128, dt=0.01, sigma_mult=1.0,
                 j_lo=0.005, j_peak=0.15, j_hi=0.5):
        self.strength = strength
        self.window = window
        self.dt = dt
        self.sigma_mult = sigma_mult
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None

    def _wiener(self):
        if len(self._hist) < 32:
            return None
        H = np.array(self._hist)
        T = len(H)
        F = np.fft.rfft(H, axis=0)
        mag = np.abs(F)
        # 噪声方差估计：中位数 * sigma_mult
        sigma = np.median(mag, axis=1) * self.sigma_mult
        sigma = sigma[:, None]
        # 维纳增益: |F|² / (|F|² + σ²)
        gain = mag ** 2 / (mag ** 2 + sigma ** 2 + 1e-9)
        F_clean = F * gain
        recon = np.fft.irfft(F_clean, n=T, axis=0)
        return recon[-1]

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)
        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a
        cleaned = self._wiener()
        if cleaned is None:
            return a
        return a + self.strength * gate * (cleaned - a)


# ──────────────────────────────────────────────────────────────────── #
#  V31: 中值滤波                                                       #
#  对最近 N 步取中值，杀脉冲噪声                                       #
# ──────────────────────────────────────────────────────────────────── #
class MedianFilterPlugin(BasePlugin):
    def __init__(self, strength=1.0, window=7,
                 j_lo=0.005, j_peak=0.15, j_hi=0.5):
        self.strength = strength
        self.window = window
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)
        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a
        if len(self._hist) < 3:
            return a
        H = np.array(self._hist)
        med = np.median(H, axis=0)
        return a + self.strength * gate * (med - a)


# ──────────────────────────────────────────────────────────────────── #
#  组合：级联 + 谱减                                                    #
# ──────────────────────────────────────────────────────────────────── #
class CascadePlusSpectral(BasePlugin):
    def __init__(self, n_passes=7, spec_strength=1.0, spec_window=128,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_passes)]
        self.spec = SpectralSubtraction(strength=spec_strength, window=spec_window,
                                         j_lo=0.005, j_peak=j_peak, j_hi=j_hi)
    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.spec.inject(a, **kw)
        return a


class CascadePlusWiener(BasePlugin):
    def __init__(self, n_passes=7, wiener_strength=1.0, wiener_window=128,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_passes)]
        self.wiener = WienerFilter(strength=wiener_strength, window=wiener_window,
                                    j_lo=0.005, j_peak=j_peak, j_hi=j_hi)
    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.wiener.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  组合：谱减先行 + 级联                                                #
# ──────────────────────────────────────────────────────────────────── #
class SpectralPlusCascade(BasePlugin):
    def __init__(self, spec_strength=1.0, spec_window=128, n_passes=7,
                 j_lo=0.005, j_peak=0.15, j_hi=0.5):
        self.spec = SpectralSubtraction(strength=spec_strength, window=spec_window,
                                         j_lo=j_lo, j_peak=j_peak, j_hi=j_hi)
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_passes)]
    def inject(self, a, **kw):
        a = self.spec.inject(a, **kw)
        for p in self.cascade:
            a = p.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  组合：中值 + 级联 + 谱减                                             #
# ──────────────────────────────────────────────────────────────────── #
class TripleDenoise(BasePlugin):
    def __init__(self, med_window=7, n_passes=5, spec_window=128):
        self.med = MedianFilterPlugin(strength=0.5, window=med_window,
                                       j_lo=0.005, j_peak=0.15, j_hi=0.5)
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_passes)]
        self.spec = SpectralSubtraction(strength=1.0, window=spec_window,
                                         j_lo=0.005, j_peak=0.15, j_hi=0.5)
    def inject(self, a, **kw):
        a = self.med.inject(a, **kw)
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.spec.inject(a, **kw)
        return a


def main():
    print("=" * 120)
    print("第十二轮：谱减法 + 维纳滤波 + 中值滤波")
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

    # ── V29 谱减法 单独 ──
    all_r['R1_spec_s05_w128'] = run_round(1, "谱减 strength=0.5 w=128",
        lambda: SpectralSubtraction(strength=0.5, window=128))
    all_r['R2_spec_s10_w128'] = run_round(2, "谱减 strength=1.0 w=128",
        lambda: SpectralSubtraction(strength=1.0, window=128))
    all_r['R3_spec_s15_w256'] = run_round(3, "谱减 strength=1.5 w=256",
        lambda: SpectralSubtraction(strength=1.5, window=256))
    all_r['R4_spec_s20_w256'] = run_round(4, "谱减 strength=2.0 w=256",
        lambda: SpectralSubtraction(strength=2.0, window=256))

    # ── V30 维纳滤波 单独 ──
    all_r['R5_wiener_s10_w128'] = run_round(5, "维纳 strength=1.0 w=128",
        lambda: WienerFilter(strength=1.0, window=128))
    all_r['R6_wiener_s15_w256'] = run_round(6, "维纳 strength=1.5 w=256",
        lambda: WienerFilter(strength=1.5, window=256))

    # ── V31 中值滤波 单独 ──
    all_r['R7_median_s05_w7'] = run_round(7, "中值 strength=0.5 w=7",
        lambda: MedianFilterPlugin(strength=0.5, window=7))
    all_r['R8_median_s10_w9'] = run_round(8, "中值 strength=1.0 w=9",
        lambda: MedianFilterPlugin(strength=1.0, window=9))

    # ── 组合：级联 + 谱减 ──
    all_r['R9_cascade7_spec_s10'] = run_round(9, "级联7 + 谱减(1.0 w=128)",
        lambda: CascadePlusSpectral(n_passes=7, spec_strength=1.0, spec_window=128))
    all_r['R10_cascade7_spec_s15'] = run_round(10, "级联7 + 谱减(1.5 w=256)",
        lambda: CascadePlusSpectral(n_passes=7, spec_strength=1.5, spec_window=256))
    all_r['R11_cascade7_spec_s20'] = run_round(11, "级联7 + 谱减(2.0 w=256)",
        lambda: CascadePlusSpectral(n_passes=7, spec_strength=2.0, spec_window=256))

    # ── 组合：级联 + 维纳 ──
    all_r['R12_cascade7_wiener_s10'] = run_round(12, "级联7 + 维纳(1.0 w=128)",
        lambda: CascadePlusWiener(n_passes=7, wiener_strength=1.0, wiener_window=128))
    all_r['R13_cascade7_wiener_s15'] = run_round(13, "级联7 + 维纳(1.5 w=256)",
        lambda: CascadePlusWiener(n_passes=7, wiener_strength=1.5, wiener_window=256))

    # ── 组合：谱减先行 + 级联 ──
    all_r['R14_spec_then_cascade7'] = run_round(14, "谱减(1.5) + 级联7",
        lambda: SpectralPlusCascade(spec_strength=1.5, spec_window=256, n_passes=7))

    # ── 三件套：中值 + 级联 + 谱减 ──
    all_r['R15_triple_med5_cascade5_spec'] = run_round(15, "中值5 + 级联5 + 谱减",
        lambda: TripleDenoise(med_window=5, n_passes=5, spec_window=128))
    all_r['R16_triple_med7_cascade7_spec'] = run_round(16, "中值7 + 级联7 + 谱减",
        lambda: TripleDenoise(med_window=7, n_passes=7, spec_window=256))

    # ── 谱减 + 维纳 双重 ──
    class SpecPlusWiener(BasePlugin):
        def __init__(self):
            self.spec = SpectralSubtraction(strength=1.5, window=256,
                                             j_lo=0.005, j_peak=0.15, j_hi=0.5)
            self.wiener = WienerFilter(strength=1.0, window=128,
                                        j_lo=0.005, j_peak=0.15, j_hi=0.5)
        def inject(self, a, **kw):
            a = self.spec.inject(a, **kw)
            a = self.wiener.inject(a, **kw)
            return a
    all_r['R17_spec_wiener_double'] = run_round(17, "谱减+维纳 双重",
        lambda: SpecPlusWiener())

    # ── 级联 + 谱减 + 维纳 三重 ──
    class CascadeSpecWiener(BasePlugin):
        def __init__(self):
            self.cascade = [GoldilocksFusion(**wide) for _ in range(7)]
            self.spec = SpectralSubtraction(strength=1.5, window=256,
                                             j_lo=0.005, j_peak=0.15, j_hi=0.5)
            self.wiener = WienerFilter(strength=1.0, window=128,
                                        j_lo=0.005, j_peak=0.15, j_hi=0.5)
        def inject(self, a, **kw):
            for p in self.cascade:
                a = p.inject(a, **kw)
            a = self.spec.inject(a, **kw)
            a = self.wiener.inject(a, **kw)
            return a
    all_r['R18_cascade7_spec_wiener'] = run_round(18, "级联7+谱减+维纳 三重",
        lambda: CascadeSpecWiener())

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

    out = os.path.join(os.path.dirname(__file__), "stress_test12.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
