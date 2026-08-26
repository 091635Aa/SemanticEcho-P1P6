#!/usr/bin/env python3
"""
stress_test21.py — 第二十一轮：高级频域/时域滤波突破天花板

stress_test20 结果：
  T=6400 cascade5+LPF 仍是冠军: avg=+0.4493, p2p=+0.1107
  添加 BlueprintForcer/HarmonicInjector 反而降低性能
  → 直接蓝图注入与级联冲突

诊断：当前优化本质是"噪声平滑"，CI 天花板由动力学决定
  q_new = q*0.75 + a*0.25 限制了 a 对 q 的影响速率
  即便 a 完美周期，q 仍含"记忆性"→ 仍有非零 jerk

新机制（频域/时域高级滤波）：
  V65 SavitzkyGolaySmoother — 多项式拟合保形平滑
  V66 SpectralSubtractor — FFT 噪声底估计+软减
  V67 WienerFilter — 频域最优滤波
  V68 KalmanSmoother — 状态空间最优估计
  V69 CascadedMultiSmoother — 多重平滑器组合
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion

SEEDS_5 = [42, 137, 2024, 7777, 314159]
SEEDS_3 = [42, 137, 2024]


def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)


def eval_plugin(make_fn, family, seeds, T=800):
    opts = []
    for seed in seeds:
        base = BasePolicy(n_joints=6, family=family, seed=seed)
        sim_b = LeggedMicroSim(base, plugins=[], seed=seed, T=T)
        t = sim_b.run(goal=3.0, terrain=0.3)
        ci_b = compute_coherence(t['q'], t['dq'],
                                 central_diff(t['dq'], sim_b.dt),
                                 dt=sim_b.dt)['coherence_index']
        plug = make_fn()
        sim_p = LeggedMicroSim(base, plugins=[plug], seed=seed, T=T)
        t2 = sim_p.run(goal=3.0, terrain=0.3)
        ci_p = compute_coherence(t2['q'], t2['dq'],
                                 central_diff(t2['dq'], sim_p.dt),
                                 dt=sim_p.dt)['coherence_index']
        opts.append(opt(ci_p, ci_b))
    return float(np.mean(opts)), float(np.std(opts))


def run_round(round_num, name, make_fn, T=800, seeds=SEEDS_5):
    results = {}
    for fam in ['p2p', 'standard', 'transformer']:
        m, s = eval_plugin(make_fn, fam, seeds, T=T)
        results[fam] = {'mean': m, 'std': s}
    uni = all(results[f]['mean'] > 0 for f in ['p2p', 'standard', 'transformer'])
    results['universal'] = uni
    tag = "OK" if uni else "--"
    avg = (results['standard']['mean'] + results['transformer']['mean']) / 2
    print(f"  R{round_num}: {name:50s}  std={results['standard']['mean']:+.4f}  trans={results['transformer']['mean']:+.4f}  avg={avg:+.4f}  p2p={results['p2p']['mean']:+.4f}  {tag}")
    return results


# 复用 AdaptiveLPF
class AdaptiveLPF(BasePlugin):
    def __init__(self, decay=0.3, strength=0.8,
                 j_act_lo=0.3, j_act_hi=1.0):
        self.decay = decay
        self.strength = strength
        self.j_act_lo = j_act_lo
        self.j_act_hi = j_act_hi
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

    def _adaptive_gate(self, j):
        if j < self.j_act_lo:
            return 0.0
        if j >= self.j_act_hi:
            return 1.0
        t = (j - self.j_act_lo) / (self.j_act_hi - self.j_act_lo + 1e-9)
        return float(t)

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        self._hist.append(a.copy())
        if len(self._hist) > 5:
            self._hist.pop(0)
        if len(self._hist) < 2:
            return a
        j = self._measure_jerk(a)
        gate = self._adaptive_gate(j)
        if gate < 0.01:
            return a
        weights = np.array([(1 - self.decay) ** i for i in range(len(self._hist) - 1, -1, -1)])
        weights = weights / weights.sum()
        smoothed = np.average(np.stack(self._hist), axis=0, weights=weights)
        return a + self.strength * gate * (smoothed - a)


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
#  V65: SavitzkyGolaySmoother — 多项式拟合保形平滑                  #
#  对最近 N 步做 polyorder 阶多项式拟合，用拟合值替换当前动作       #
#  比 LPF 更好地保留信号形状（导数连续）                            #
# ──────────────────────────────────────────────────────────────────── #
class SavitzkyGolaySmoother(BasePlugin):
    def __init__(self, window=7, polyorder=2, strength=0.7,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5):
        self.window = window
        self.polyorder = polyorder
        self.strength = strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None

    def _sg_smooth(self, hist):
        """简单 SG：对历史窗口做多项式拟合，返回末点拟合值。"""
        n = len(hist)
        if n < self.polyorder + 1:
            return hist[-1]
        # 对每个关节分别拟合
        try:
            x = np.arange(n)
            arr = np.stack(hist)  # (n, dim)
            # 多项式拟合
            coeffs = np.polyfit(x, arr, self.polyorder)
            # 末点（x=n-1）的拟合值
            fitted = np.polyval(coeffs, n - 1)
            return np.atleast_1d(fitted).astype(float)
        except Exception:
            return hist[-1]

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)
        if len(self._hist) < self.polyorder + 2:
            return a

        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a

        smoothed = self._sg_smooth(self._hist)
        return a + self.strength * gate * (smoothed - a)


# ──────────────────────────────────────────────────────────────────── #
#  V66: SpectralSubtractor — FFT 噪声底估计+软减                  #
#  滑窗 FFT, 估计噪声底（中位数），对高频软减                       #
# ──────────────────────────────────────────────────────────────────── #
class SpectralSubtractor(BasePlugin):
    def __init__(self, window=32, strength=0.5, noise_floor_q=0.3,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5):
        self.window = window
        self.strength = strength
        self.noise_floor_q = noise_floor_q
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None

    def _spec_sub(self, hist):
        n = len(hist)
        if n < 4:
            return hist[-1]
        try:
            arr = np.stack(hist)  # (n, dim)
            # 对每个关节做谱减
            out = np.zeros_like(arr[-1])
            for j_dim in range(arr.shape[1]):
                sig = arr[:, j_dim]
                sig = sig - sig.mean()
                # 汉宁窗
                win = np.hanning(n)
                spec = np.fft.rfft(sig * win)
                mag = np.abs(spec)
                # 噪声底估计：幅度的分位数
                noise_floor = np.quantile(mag[1:], self.noise_floor_q) + 1e-9
                # 软减：|S'| = max(|S| - alpha*noise, beta*|S|)
                new_mag = np.maximum(mag - 0.5 * noise_floor, 0.1 * mag)
                # 重建
                new_spec = new_mag * np.exp(1j * np.angle(spec))
                cleaned = np.fft.irfft(new_spec, n=n)
                out[j_dim] = cleaned[-1] + arr[:, j_dim].mean()
            return out
        except Exception:
            return hist[-1]

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)
        if len(self._hist) < 8:
            return a

        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a

        cleaned = self._spec_sub(self._hist)
        return a + self.strength * gate * (cleaned - a)


# ──────────────────────────────────────────────────────────────────── #
#  V67: WienerFilter — 频域最优滤波                                  #
#  F_clean = F * |F|^2 / (|F|^2 + sigma^2)                         #
# ──────────────────────────────────────────────────────────────────── #
class WienerFilter(BasePlugin):
    def __init__(self, window=32, strength=0.5,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5):
        self.window = window
        self.strength = strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None

    def _wiener(self, hist):
        n = len(hist)
        if n < 8:
            return hist[-1]
        try:
            arr = np.stack(hist)  # (n, dim)
            out = np.zeros_like(arr[-1])
            for j_dim in range(arr.shape[1]):
                sig = arr[:, j_dim]
                sig = sig - sig.mean()
                win = np.hanning(n)
                spec = np.fft.rfft(sig * win)
                mag2 = np.abs(spec) ** 2
                # 噪声方差估计：高频分位
                noise_var = np.quantile(mag2[1:], 0.2) + 1e-9
                # Wiener: H = mag2 / (mag2 + noise_var)
                H = mag2 / (mag2 + noise_var)
                cleaned_spec = spec * H
                cleaned = np.fft.irfft(cleaned_spec, n=n)
                out[j_dim] = cleaned[-1] + arr[:, j_dim].mean()
            return out
        except Exception:
            return hist[-1]

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)
        if len(self._hist) < 8:
            return a

        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a

        cleaned = self._wiener(self._hist)
        return a + self.strength * gate * (cleaned - a)


# ──────────────────────────────────────────────────────────────────── #
#  V68: KalmanSmoother — 简化 Kalman 估计                          #
#  状态：q_true; 观测：a (含噪声)                                   #
#  q_true_pred = q_true; q_true += K * (a - q_true)                 #
# ──────────────────────────────────────────────────────────────────── #
class KalmanSmoother(BasePlugin):
    def __init__(self, process_var=0.01, meas_var=0.05, strength=0.7,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5):
        self.process_var = process_var
        self.meas_var = meas_var
        self.strength = strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._x_hat = None  # 状态估计
        self._p = None      # 估计方差
        self._jerk_hist = []
        self._last_a = None

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        if self._x_hat is None:
            self._x_hat = a.copy()
            self._p = np.ones_like(a) * self.meas_var
            return a

        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            # 仍然更新估计，但不修改 a
            self._x_hat = self._x_hat  # 预测=不变
            self._p = self._p + self.process_var
            K = self._p / (self._p + self.meas_var)
            self._x_hat = self._x_hat + K * (a - self._x_hat)
            self._p = (1 - K) * self._p
            return a

        # Kalman 更新
        self._p = self._p + self.process_var
        K = self._p / (self._p + self.meas_var)
        self._x_hat = self._x_hat + K * (a - self._x_hat)
        self._p = (1 - K) * self._p
        # 平滑输出
        return a + self.strength * gate * (self._x_hat - a)


# ──────────────────────────────────────────────────────────────────── #
#  V69: CascadeLPFPlusSmoother — 级联+LPF+高级平滑器                #
#  把 SG/Wiener/SpectralSub 加到 cascade+LPF 之后，看能否突破        #
# ──────────────────────────────────────────────────────────────────── #
class CascadeLPFSmoother(BasePlugin):
    def __init__(self, n_cascade=5, smoother_type='sg',
                 lpf_decay=0.3, lpf_strength=0.8,
                 smoother_strength=0.5,
                 j_peak=0.15, j_hi=0.5,
                 j_act_lo=0.3, j_act_hi=1.0):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_cascade)]
        self.lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                                j_act_lo=j_act_lo, j_act_hi=j_act_hi)
        if smoother_type == 'sg':
            self.smoother = SavitzkyGolaySmoother(strength=smoother_strength)
        elif smoother_type == 'wiener':
            self.smoother = WienerFilter(strength=smoother_strength)
        elif smoother_type == 'spec':
            self.smoother = SpectralSubtractor(strength=smoother_strength)
        elif smoother_type == 'kalman':
            self.smoother = KalmanSmoother(strength=smoother_strength)
        else:
            raise ValueError(f"Unknown smoother: {smoother_type}")

    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.lpf.inject(a, **kw)
        return self.smoother.inject(a, **kw)


def main():
    print("=" * 120)
    print("第二十一轮：高级频域/时域滤波突破天花板 (SG/Wiener/SpectralSub/Kalman)")
    print("=" * 120)
    all_r = {}
    wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}

    # ── T=6400 对照：上轮最优 ──
    class RefAsym(BasePlugin):
        def __init__(self):
            self.cascade = [GoldilocksFusion(**wide) for _ in range(5)]
            self.lpf = AdaptiveLPF(decay=0.3, strength=0.8,
                                    j_act_lo=0.3, j_act_hi=1.0)
        def inject(self, a, **kw):
            for p in self.cascade:
                a = p.inject(a, **kw)
            return self.lpf.inject(a, **kw)
    all_r['R0_ref_T6400'] = run_round(0, "对照 cascade5+LPF T=6400",
        lambda: RefAsym(), T=6400)

    # ── T=800 各平滑器单独测 ──
    for s_type in ['sg', 'wiener', 'spec', 'kalman']:
        for strength in [0.5, 0.7]:
            cls = {'sg': SavitzkyGolaySmoother, 'wiener': WienerFilter,
                   'spec': SpectralSubtractor, 'kalman': KalmanSmoother}[s_type]
            all_r[f'R1_{s_type}_s{strength}_T800'] = run_round(1,
                f"纯 {s_type} s={strength} T=800",
                lambda s=strength, c=cls: c(strength=s), T=800)

    # ── T=6400 Cascade+LPF+各平滑器 ──
    for s_type in ['sg', 'wiener', 'spec', 'kalman']:
        for strength in [0.3, 0.5]:
            all_r[f'R2_combo_{s_type}_s{strength}_T6400'] = run_round(2,
                f"Cascade+LPF+{s_type} s={strength} T=6400",
                lambda st=s_type, s=strength: CascadeLPFSmoother(
                    smoother_type=st, smoother_strength=s), T=6400)

    # ── T=6400 仅用平滑器替代 LPF ──
    class CascadePlusSmoother(BasePlugin):
        def __init__(self, smoother_type='sg', strength=0.7):
            self.cascade = [GoldilocksFusion(**wide) for _ in range(5)]
            if smoother_type == 'sg':
                self.smoother = SavitzkyGolaySmoother(strength=strength)
            elif smoother_type == 'wiener':
                self.smoother = WienerFilter(strength=strength)
            elif smoother_type == 'spec':
                self.smoother = SpectralSubtractor(strength=strength)
            elif smoother_type == 'kalman':
                self.smoother = KalmanSmoother(strength=strength)
        def inject(self, a, **kw):
            for p in self.cascade:
                a = p.inject(a, **kw)
            return self.smoother.inject(a, **kw)

    # T=800 cascade+smoother (without LPF)
    for s_type in ['sg', 'wiener']:
        all_r[f'R3_noLPF_{s_type}_T800'] = run_round(3,
            f"无LPF cascade+{s_type} T=800",
            lambda st=s_type: CascadePlusSmoother(smoother_type=st, strength=0.7),
            T=800)

    # ── T=6400 高级平滑器（受高 jerk 也要激活） ──
    # 调整 SG/Wiener 的 j_hi 让其在 P2P 也激活
    class WideSGSmoother(BasePlugin):
        def __init__(self):
            self.s = SavitzkyGolaySmoother(strength=0.7, j_lo=0.05, j_peak=0.15, j_hi=2.0)
            self.cascade = [GoldilocksFusion(**wide) for _ in range(5)]
        def inject(self, a, **kw):
            for p in self.cascade:
                a = p.inject(a, **kw)
            return self.s.inject(a, **kw)
    all_r['R4_wide_SG_T6400'] = run_round(4, "宽门SG T=6400",
        lambda: WideSGSmoother(), T=6400)

    # ── T=6400 级联+LPF+SG（多种 SG 强度） ──
    for sg_str in [0.3, 0.5, 0.7]:
        all_r[f'R5_cascade_LPF_SG{sg_str}_T6400'] = run_round(5,
            f"级联+LPF+SG(s={sg_str}) T=6400",
            lambda s=sg_str: CascadeLPFSmoother(smoother_type='sg',
                                                 smoother_strength=s),
            T=6400)

    # ── 汇总 ──
    print(f"\n{'='*120}")
    print("汇总（按 std+trans 均值排序）")
    print(f"{'='*120}")
    sorted_r = sorted(all_r.items(),
                      key=lambda x: (x[1]['standard']['mean'] + x[1]['transformer']['mean']) / 2,
                      reverse=True)
    print(f"  {'配置':42s}  {'std':>8s}  {'trans':>8s}  {'avg':>8s}  {'p2p':>8s}  通用?")
    print("  " + "-" * 90)
    for k, v in sorted_r:
        uni = v.get('universal', False)
        avg = (v['standard']['mean'] + v['transformer']['mean']) / 2
        print(f"  {k:42s}  {v['standard']['mean']:+.4f}  {v['transformer']['mean']:+.4f}  {avg:+.4f}  {v['p2p']['mean']:+.4f}  {'YES' if uni else 'NO'}")

    best = sorted_r[0]
    print(f"\n  最优: {best[0]} → avg={(best[1]['standard']['mean']+best[1]['transformer']['mean'])/2:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "stress_test21.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
