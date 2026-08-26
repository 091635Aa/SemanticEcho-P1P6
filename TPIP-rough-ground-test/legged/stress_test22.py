#!/usr/bin/env python3
"""
stress_test22.py — 第二十二轮：预测式注入 + T=25600 极限

stress_test21 结果：
  T=6400 + Kalman → avg=+0.4509 (+0.0016 改进)
  其他频域/时域滤波器(SG/Wiener/Spec)无显著改进
  → 当前架构已接近 ~0.45 ceiling

新策略（前瞻式注入）：
  V70 PredictiveInjector — 用历史窗口 FFT 外推下一步
  V71 ParallelCascade — 多路并行级联+加权平均
  V72 StackedSmoothers — 多重平滑器堆叠
  V73 AdaptivePredictor — 自适应预测器

极限测试：
  T=25600 测试最优配置
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


# ──────────────────────────────────────────────────────────────────── #
#  复用 AdaptiveLPF & KalmanSmoother                                #
# ──────────────────────────────────────────────────────────────────── #
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


class KalmanSmoother(BasePlugin):
    def __init__(self, process_var=0.01, meas_var=0.05, strength=0.7,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5):
        self.process_var = process_var
        self.meas_var = meas_var
        self.strength = strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._x_hat = None
        self._p = None
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
        self._p = self._p + self.process_var
        K = self._p / (self._p + self.meas_var)
        self._x_hat = self._x_hat + K * (a - self._x_hat)
        self._p = (1 - K) * self._p
        if gate < 0.01:
            return a
        return a + self.strength * gate * (self._x_hat - a)


# ──────────────────────────────────────────────────────────────────── #
#  V70: PredictiveInjector — 历史窗口 FFT 外推下一步                #
#  用过去 N 步识别主频，外推下一步的"理想"动作                      #
# ──────────────────────────────────────────────────────────────────── #
class PredictiveInjector(BasePlugin):
    def __init__(self, window=64, strength=0.4, dt=0.01,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5):
        self.window = window
        self.strength = strength
        self.dt = dt
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None

    def _predict_next(self):
        n = len(self._hist)
        if n < self.window:
            return self._hist[-1]
        try:
            arr = np.stack(self._hist[-self.window:])  # (N, dim)
            n_dim = arr.shape[1]
            pred = np.zeros(n_dim)
            for d in range(n_dim):
                sig = arr[:, d] - arr[:, d].mean()
                win = np.hanning(n)
                spec = np.fft.rfft(sig * win)
                # 找主频
                if len(spec) < 3:
                    pred[d] = arr[-1, d]
                    continue
                peak_idx = np.argmax(np.abs(spec[1:])) + 1
                # 单频重建+外推
                main_phase = np.angle(spec[peak_idx])
                main_mag = np.abs(spec[peak_idx]) / (n / 2)
                # 下一步相位 = 当前 phase + 2π*f
                f = peak_idx / (n * self.dt)
                next_phase = main_phase + 2 * np.pi * f * self.dt
                pred[d] = main_mag * np.cos(next_phase) + arr[:, d].mean()
            return pred
        except Exception:
            return self._hist[-1]

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        self._hist.append(a.copy())
        if len(self._hist) > self.window * 2:
            self._hist.pop(0)
        if len(self._hist) < self.window // 2:
            return a

        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a

        pred = self._predict_next()
        return a + self.strength * gate * (pred - a)


# ──────────────────────────────────────────────────────────────────── #
#  V71: ParallelCascade — 多路并行级联+加权平均                     #
#  对原始 a 应用不同 j_peak 的级联，输出取平均                      #
# ──────────────────────────────────────────────────────────────────── #
class ParallelCascade(BasePlugin):
    def __init__(self, lpf_decay=0.3, lpf_strength=0.8,
                 cfgs=None,
                 j_act_lo=0.3, j_act_hi=1.0):
        if cfgs is None:
            cfgs = [
                {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.10, 'j_hi': 0.4},
                {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5},
                {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.20, 'j_hi': 0.6},
            ]
        self.branches = []
        for cfg in cfgs:
            self.branches.append([GoldilocksFusion(**cfg) for _ in range(5)])
        self.lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                                j_act_lo=j_act_lo, j_act_hi=j_act_hi)

    def inject(self, a, **kw):
        outs = []
        for branch in self.branches:
            aa = a.copy()
            for p in branch:
                aa = p.inject(aa, **kw)
            outs.append(aa)
        a = np.mean(outs, axis=0)
        return self.lpf.inject(a, **kw)


# ──────────────────────────────────────────────────────────────────── #
#  V72: StackedSmoothers — Kalman + SG + LPF 多重平滑                #
# ──────────────────────────────────────────────────────────────────── #
class StackedSmoothers(BasePlugin):
    def __init__(self, lpf_decay=0.3, lpf_strength=0.8,
                 kalman_strength=0.5, n_cascade=5):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_cascade)]
        self.kalman = KalmanSmoother(strength=kalman_strength)
        self.lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                                j_act_lo=0.3, j_act_hi=1.0)

    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.kalman.inject(a, **kw)
        return self.lpf.inject(a, **kw)


# ──────────────────────────────────────────────────────────────────── #
#  V73: AdaptivePredictor — 自适应预测+多重平滑                     #
#  组合：cascade + Kalman + Predictive + LPF                       #
# ──────────────────────────────────────────────────────────────────── #
class AdaptivePredictor(BasePlugin):
    def __init__(self, n_cascade=5,
                 lpf_decay=0.3, lpf_strength=0.8,
                 kalman_strength=0.5,
                 pred_strength=0.3):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_cascade)]
        self.kalman = KalmanSmoother(strength=kalman_strength)
        self.predictor = PredictiveInjector(strength=pred_strength)
        self.lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                                j_act_lo=0.3, j_act_hi=1.0)

    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.kalman.inject(a, **kw)
        a = self.predictor.inject(a, **kw)
        return self.lpf.inject(a, **kw)


def main():
    print("=" * 120)
    print("第二十二轮：预测式注入 + T=25600 极限")
    print("=" * 120)
    all_r = {}
    wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}

    # ── T=6400 对照：上轮最优 ──
    class RefBest(BasePlugin):
        def __init__(self):
            self.cascade = [GoldilocksFusion(**wide) for _ in range(5)]
            self.kalman = KalmanSmoother(strength=0.5)
            self.lpf = AdaptiveLPF(decay=0.3, strength=0.8,
                                    j_act_lo=0.3, j_act_hi=1.0)
        def inject(self, a, **kw):
            for p in self.cascade:
                a = p.inject(a, **kw)
            a = self.kalman.inject(a, **kw)
            return self.lpf.inject(a, **kw)
    all_r['R0_ref_T6400'] = run_round(0, "对照 cascade+Kalman+LPF T=6400",
        lambda: RefBest(), T=6400)

    # ── T=800 纯预测器 ──
    for strength in [0.3, 0.5, 0.7]:
        all_r[f'R1_pred_s{strength}_T800'] = run_round(1,
            f"PredictiveInjector s={strength} T=800",
            lambda s=strength: PredictiveInjector(strength=s), T=800)

    # ── T=800 ParallelCascade ──
    all_r['R2_parallel_T800'] = run_round(2, "ParallelCascade T=800",
        lambda: ParallelCascade(), T=800)

    # ── T=6400 组合：Cascade+Kalman+Predictive+LPF ──
    for pred_s in [0.2, 0.3, 0.5]:
        all_r[f'R3_full_pred{pred_s}_T6400'] = run_round(3,
            f"Cascade+Kalman+Pred(s={pred_s})+LPF T=6400",
            lambda s=pred_s: AdaptivePredictor(pred_strength=s), T=6400)

    # ── T=6400 ParallelCascade ──
    all_r['R4_parallel_T6400'] = run_round(4, "ParallelCascade T=6400",
        lambda: ParallelCascade(), T=6400)

    # ── T=6400 StackedSmoothers ──
    for kalman_s in [0.3, 0.5, 0.7]:
        all_r[f'R5_stack_k{kalman_s}_T6400'] = run_round(5,
            f"Stacked(Kalman s={kalman_s}) T=6400",
            lambda s=kalman_s: StackedSmoothers(kalman_strength=s), T=6400)

    # ── T=12800 极限最优组合 ──
    all_r['R6_full_T12800'] = run_round(6, "全组合 T=12800",
        lambda: AdaptivePredictor(pred_strength=0.3), T=12800, seeds=SEEDS_3)

    # ── T=25600 极限 ──
    all_r['R7_full_T25600'] = run_round(7, "全组合 T=25600(极限)",
        lambda: AdaptivePredictor(pred_strength=0.3), T=25600, seeds=SEEDS_3)

    # ── T=25600 Kalman组合 ──
    all_r['R8_kalman_T25600'] = run_round(8, "Cascade+Kalman+LPF T=25600",
        lambda: RefBest(), T=25600, seeds=SEEDS_3)

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

    out = os.path.join(os.path.dirname(__file__), "stress_test22.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
