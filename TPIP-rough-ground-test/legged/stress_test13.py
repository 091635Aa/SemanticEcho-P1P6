#!/usr/bin/env python3
"""
stress_test13.py — 第十三轮：频带选择性降噪

发现（stress_test12）：
- 谱减+级联 对 transformer 有效 (trans +0.39 vs 0.38)
- 但对 standard 有损 (std +0.20 vs 0.27)
- 原因：谱减无差别去除所有"低于噪声底"的 bin，包括 standard 的 0.3Hz 漂移

新机制：
  V34 HighFreqSpectralSub - 只对 f > 2*main_freq 的 bin 做谱减
  V35 BandStopNoise       - 保留主频带 ±1Hz，其他频带维纳滤波
  V36 AdaptiveBandDenoise - 在线检测主频，自适应构造保留带
  V37 CascadePlusHighFreq - 级联7 + 高频谱减
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
#  V34: 高频谱减 (High-Freq Spectral Subtraction)                      #
#  只对 f > cutoff 的 bin 做谱减，保留低频                              #
# ──────────────────────────────────────────────────────────────────── #
class HighFreqSpectralSub(BasePlugin):
    def __init__(self, strength=1.0, window=128, dt=0.01, cutoff_hz=3.0,
                 noise_alpha=2.0, j_lo=0.005, j_peak=0.15, j_hi=0.5):
        self.strength = strength
        self.window = window
        self.dt = dt
        self.cutoff_hz = cutoff_hz
        self.noise_alpha = noise_alpha
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None

    def _high_freq_subtract(self):
        if len(self._hist) < 32:
            return None
        H = np.array(self._hist)
        T = len(H)
        F = np.fft.rfft(H, axis=0)
        mag = np.abs(F)
        freqs = np.fft.rfftfreq(T, d=self.dt)

        # 只对 f > cutoff 的 bin 做谱减
        high_mask = freqs > self.cutoff_hz
        if not high_mask.any():
            return None

        # 高频区域的噪声底估计
        high_mag = mag[high_mask]
        noise_floor = np.median(high_mag, axis=0) * self.noise_alpha
        noise_floor = noise_floor[None, :]  # (1, n_joints)

        # 软减：只对高频 bin
        mag_clean = mag.copy()
        mag_clean[high_mask] = np.maximum(high_mag - noise_floor, 0)

        # 保持相位
        phase = F / (mag + 1e-9)
        F_clean = mag_clean * phase
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
        cleaned = self._high_freq_subtract()
        if cleaned is None:
            return a
        return a + self.strength * gate * (cleaned - a)


# ──────────────────────────────────────────────────────────────────── #
#  V35: 频带维纳滤波 (Band Wiener)                                      #
#  保留主频带 ±band_width，其他频带做维纳                                #
# ──────────────────────────────────────────────────────────────────── #
class BandWiener(BasePlugin):
    def __init__(self, strength=1.0, window=128, dt=0.01,
                 band_width=1.5, sigma_mult=1.5,
                 j_lo=0.005, j_peak=0.15, j_hi=0.5):
        self.strength = strength
        self.window = window
        self.dt = dt
        self.band_width = band_width
        self.sigma_mult = sigma_mult
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None
        self._main_freq = 1.0  # 默认

    def _detect_main_freq(self):
        if len(self._hist) < 32:
            return 1.0
        H = np.array(self._hist)
        T = len(H)
        # 取方差最大的关节
        sig = H[:, int(np.argmax(np.var(H, axis=0)))]
        F = np.fft.rfft(sig - np.mean(sig))
        freqs = np.fft.rfftfreq(T, d=self.dt)
        mask = freqs > 0.1
        mag = np.abs(F) * mask
        peak_idx = int(np.argmax(mag))
        if mag[peak_idx] < 1e-6:
            return 1.0
        return float(freqs[peak_idx])

    def _band_wiener(self):
        if len(self._hist) < 32:
            return None
        H = np.array(self._hist)
        T = len(H)
        F = np.fft.rfft(H, axis=0)
        mag = np.abs(F)
        freqs = np.fft.rfftfreq(T, d=self.dt)

        # 保留带 = |f - main_freq| < band_width
        keep_mask = np.abs(freqs - self._main_freq) < self.band_width
        # 也保留低频（漂移）
        keep_mask = keep_mask | (freqs < 0.5)

        # 保留带：不处理
        # 非保留带：维纳滤波
        sigma = np.median(mag[~keep_mask], axis=0) * self.sigma_mult if (~keep_mask).any() else 1.0
        sigma = np.atleast_1d(sigma)
        if sigma.ndim == 0:
            sigma = np.full(mag.shape[1], float(sigma))
        # 对非保留带做维纳
        gain = np.ones_like(mag)
        not_keep = ~keep_mask
        if not_keep.any():
            mag_nk = mag[not_keep]
            sigma_b = np.broadcast_to(sigma[None, :], mag_nk.shape) if sigma.ndim == 1 else sigma
            gain[not_keep] = mag_nk ** 2 / (mag_nk ** 2 + sigma_b ** 2 + 1e-9)

        F_clean = F * gain
        recon = np.fft.irfft(F_clean, n=T, axis=0)
        return recon[-1]

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)
        # 定期更新主频
        if len(self._hist) % 16 == 0:
            self._main_freq = self._detect_main_freq()
        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a
        cleaned = self._band_wiener()
        if cleaned is None:
            return a
        return a + self.strength * gate * (cleaned - a)


# ──────────────────────────────────────────────────────────────────── #
#  V36: 自适应频带降噪                                                  #
#  在线检测主频，自适应构造保留带，其他频带谱减                           #
# ──────────────────────────────────────────────────────────────────── #
class AdaptiveBandDenoise(BasePlugin):
    def __init__(self, strength=1.0, window=128, dt=0.01,
                 band_width=1.5, noise_alpha=2.0,
                 j_lo=0.005, j_peak=0.15, j_hi=0.5):
        self.strength = strength
        self.window = window
        self.dt = dt
        self.band_width = band_width
        self.noise_alpha = noise_alpha
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None
        self._main_freq = 1.0

    def _detect_main_freq(self):
        if len(self._hist) < 32:
            return 1.0
        H = np.array(self._hist)
        T = len(H)
        sig = H[:, int(np.argmax(np.var(H, axis=0)))]
        F = np.fft.rfft(sig - np.mean(sig))
        freqs = np.fft.rfftfreq(T, d=self.dt)
        mask = freqs > 0.1
        mag = np.abs(F) * mask
        peak_idx = int(np.argmax(mag))
        if mag[peak_idx] < 1e-6:
            return 1.0
        return float(freqs[peak_idx])

    def _adaptive_denoise(self):
        if len(self._hist) < 32:
            return None
        H = np.array(self._hist)
        T = len(H)
        F = np.fft.rfft(H, axis=0)
        mag = np.abs(F)
        freqs = np.fft.rfftfreq(T, d=self.dt)

        # 保留带 = 主频 ± band_width 和 DC ± 0.5Hz
        keep_mask = (np.abs(freqs - self._main_freq) < self.band_width) | (freqs < 0.5)

        # 非保留带的噪声底
        not_keep = ~keep_mask
        if not not_keep.any():
            return None
        high_mag = mag[not_keep]
        noise_floor = np.median(high_mag, axis=0) * self.noise_alpha
        noise_floor = noise_floor[None, :]

        # 软减：只对非保留带
        mag_clean = mag.copy()
        mag_clean[not_keep] = np.maximum(high_mag - noise_floor, 0)

        phase = F / (mag + 1e-9)
        F_clean = mag_clean * phase
        recon = np.fft.irfft(F_clean, n=T, axis=0)
        return recon[-1]

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)
        if len(self._hist) % 16 == 0:
            self._main_freq = self._detect_main_freq()
        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a
        cleaned = self._adaptive_denoise()
        if cleaned is None:
            return a
        return a + self.strength * gate * (cleaned - a)


# ──────────────────────────────────────────────────────────────────── #
#  组合：级联 + 高频谱减                                                 #
# ──────────────────────────────────────────────────────────────────── #
class CascadePlusHighFreq(BasePlugin):
    def __init__(self, n_passes=7, hf_strength=1.0, hf_cutoff=3.0,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_passes)]
        self.hf = HighFreqSpectralSub(strength=hf_strength, cutoff_hz=hf_cutoff,
                                       j_lo=0.005, j_peak=j_peak, j_hi=j_hi)
    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.hf.inject(a, **kw)
        return a


class CascadePlusBandWiener(BasePlugin):
    def __init__(self, n_passes=7, bw_strength=1.0, bw_band=1.5,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_passes)]
        self.bw = BandWiener(strength=bw_strength, band_width=bw_band,
                              j_lo=0.005, j_peak=j_peak, j_hi=j_hi)
    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.bw.inject(a, **kw)
        return a


class CascadePlusAdaptiveBand(BasePlugin):
    def __init__(self, n_passes=7, ab_strength=1.0, ab_band=1.5,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_passes)]
        self.ab = AdaptiveBandDenoise(strength=ab_strength, band_width=ab_band,
                                       j_lo=0.005, j_peak=j_peak, j_hi=j_hi)
    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.ab.inject(a, **kw)
        return a


def main():
    print("=" * 120)
    print("第十三轮：频带选择性降噪 (高频谱减 + 频带维纳 + 自适应频带)")
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

    # ── V34 高频谱减 单独 ──
    all_r['R1_hf_cut2'] = run_round(1, "高频谱减 cutoff=2Hz",
        lambda: HighFreqSpectralSub(strength=1.0, cutoff_hz=2.0))
    all_r['R2_hf_cut3'] = run_round(2, "高频谱减 cutoff=3Hz",
        lambda: HighFreqSpectralSub(strength=1.0, cutoff_hz=3.0))
    all_r['R3_hf_cut5'] = run_round(3, "高频谱减 cutoff=5Hz",
        lambda: HighFreqSpectralSub(strength=1.0, cutoff_hz=5.0))
    all_r['R4_hf_cut10'] = run_round(4, "高频谱减 cutoff=10Hz",
        lambda: HighFreqSpectralSub(strength=1.0, cutoff_hz=10.0))

    # ── V35 频带维纳 单独 ──
    all_r['R5_bw_band1'] = run_round(5, "频带维纳 band=1.0",
        lambda: BandWiener(strength=1.0, band_width=1.0))
    all_r['R6_bw_band15'] = run_round(6, "频带维纳 band=1.5",
        lambda: BandWiener(strength=1.0, band_width=1.5))
    all_r['R7_bw_band2'] = run_round(7, "频带维纳 band=2.0",
        lambda: BandWiener(strength=1.0, band_width=2.0))

    # ── V36 自适应频带 单独 ──
    all_r['R8_ab_band1'] = run_round(8, "自适应频带 band=1.0",
        lambda: AdaptiveBandDenoise(strength=1.0, band_width=1.0))
    all_r['R9_ab_band15'] = run_round(9, "自适应频带 band=1.5",
        lambda: AdaptiveBandDenoise(strength=1.0, band_width=1.5))

    # ── 组合：级联7 + 高频谱减 ──
    all_r['R10_cascade7_hf_cut2'] = run_round(10, "级联7 + 高频谱减(cut=2)",
        lambda: CascadePlusHighFreq(n_passes=7, hf_cutoff=2.0))
    all_r['R11_cascade7_hf_cut3'] = run_round(11, "级联7 + 高频谱减(cut=3)",
        lambda: CascadePlusHighFreq(n_passes=7, hf_cutoff=3.0))
    all_r['R12_cascade7_hf_cut5'] = run_round(12, "级联7 + 高频谱减(cut=5)",
        lambda: CascadePlusHighFreq(n_passes=7, hf_cutoff=5.0))
    all_r['R13_cascade7_hf_cut10'] = run_round(13, "级联7 + 高频谱减(cut=10)",
        lambda: CascadePlusHighFreq(n_passes=7, hf_cutoff=10.0))

    # ── 组合：级联7 + 频带维纳 ──
    all_r['R14_cascade7_bw_band1'] = run_round(14, "级联7 + 频带维纳(band=1)",
        lambda: CascadePlusBandWiener(n_passes=7, bw_band=1.0))
    all_r['R15_cascade7_bw_band15'] = run_round(15, "级联7 + 频带维纳(band=1.5)",
        lambda: CascadePlusBandWiener(n_passes=7, bw_band=1.5))
    all_r['R16_cascade7_bw_band2'] = run_round(16, "级联7 + 频带维纳(band=2)",
        lambda: CascadePlusBandWiener(n_passes=7, bw_band=2.0))

    # ── 组合：级联7 + 自适应频带 ──
    all_r['R17_cascade7_ab_band1'] = run_round(17, "级联7 + 自适应频带(band=1)",
        lambda: CascadePlusAdaptiveBand(n_passes=7, ab_band=1.0))
    all_r['R18_cascade7_ab_band15'] = run_round(18, "级联7 + 自适应频带(band=1.5)",
        lambda: CascadePlusAdaptiveBand(n_passes=7, ab_band=1.5))

    # ── 组合：级联5 + 高频谱减（更快）──
    all_r['R19_cascade5_hf_cut3'] = run_round(19, "级联5 + 高频谱减(cut=3)",
        lambda: CascadePlusHighFreq(n_passes=5, hf_cutoff=3.0))

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

    out = os.path.join(os.path.dirname(__file__), "stress_test13.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
