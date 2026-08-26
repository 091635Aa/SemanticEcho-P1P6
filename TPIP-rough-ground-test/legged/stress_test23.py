#!/usr/bin/env python3
"""
stress_test23.py — 第二十三轮：T=51200 极限 + T=25600 变体精调

stress_test22 结果：
  R7 T=25600 全组合 → avg=+0.5149 (+51.49%), p2p=+0.2265, Universal=YES!
  T 翻倍再次带来大跳跃 (+0.0633)
  → 趋势未饱和！

本轮目标：
  1. T=51200 极限测试（最优组合）
  2. T=25600 参数精调：cascade深度、Kalman/LPF 参数
  3. 寻找 T=25600 下的最优子配置
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


# 复用：AdaptiveLPF & KalmanSmoother
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
        # 由 measure_jerk 处理
        # 但 Kalman 需要先更新状态再决定门控
        # 简化：始终更新，gate 决定输出混合
        from numpy.linalg import norm
        if self._last_a is None:
            self._last_a = a.copy()
            self._jerk_hist.append(0.0)
        else:
            j = float(norm(a - self._last_a))
            self._last_a = a.copy()
            self._jerk_hist.append(j)
            if len(self._jerk_hist) > 20:
                self._jerk_hist.pop(0)
        j_val = float(np.mean(self._jerk_hist[-3:])) if len(self._jerk_hist) >= 3 else 0.0

        self._p = self._p + self.process_var
        K = self._p / (self._p + self.meas_var)
        self._x_hat = self._x_hat + K * (a - self._x_hat)
        self._p = (1 - K) * self._p

        # goldilocks gate
        if j_val < self.j_lo or j_val > self.j_hi:
            gate = 0.0
        elif j_val <= self.j_peak:
            t = (j_val - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)
            gate = float(np.sin(t * np.pi / 2))
        else:
            t = (j_val - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)
            gate = float(np.cos(t * np.pi / 2))

        if gate < 0.01:
            return a
        return a + self.strength * gate * (self._x_hat - a)


# ──────────────────────────────────────────────────────────────────── #
#  最优组合：Cascade5 + Kalman + LPF                                 #
# ──────────────────────────────────────────────────────────────────── #
class BestCombo(BasePlugin):
    def __init__(self, n_cascade=5, kalman_s=0.5,
                 lpf_decay=0.3, lpf_strength=0.8,
                 j_peak=0.15, j_hi=0.5,
                 j_act_lo=0.3, j_act_hi=1.0):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_cascade)]
        self.kalman = KalmanSmoother(strength=kalman_s)
        self.lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                                j_act_lo=j_act_lo, j_act_hi=j_act_hi)

    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.kalman.inject(a, **kw)
        return self.lpf.inject(a, **kw)


def main():
    print("=" * 120)
    print("第二十三轮：T=51200 极限 + T=25600 变体精调")
    print("=" * 120)
    all_r = {}

    # ── T=25600 对照：上轮最优 ──
    all_r['R0_ref_T25600'] = run_round(0, "对照 全组合 T=25600",
        lambda: BestCombo(), T=25600, seeds=SEEDS_3)

    # ── T=25600 cascade 深度扫描 ──
    for n in [3, 5, 7]:
        all_r[f'R1_cascade{n}_T25600'] = run_round(1,
            f"级联{n}+Kalman+LPF T=25600",
            lambda n=n: BestCombo(n_cascade=n), T=25600, seeds=SEEDS_3)

    # ── T=25600 Kalman strength 扫描 ──
    for ks in [0.3, 0.5, 0.7, 1.0]:
        all_r[f'R2_kalman_s{ks}_T25600'] = run_round(2,
            f"Kalman s={ks} T=25600",
            lambda s=ks: BestCombo(kalman_s=s), T=25600, seeds=SEEDS_3)

    # ── T=25600 LPF decay 扫描 ──
    for d in [0.2, 0.3, 0.4, 0.5]:
        all_r[f'R3_lpf_d{d}_T25600'] = run_round(3,
            f"LPF d={d} T=25600",
            lambda d=d: BestCombo(lpf_decay=d), T=25600, seeds=SEEDS_3)

    # ── T=25600 LPF strength 扫描 ──
    for s in [0.6, 0.8, 1.0]:
        all_r[f'R4_lpf_s{s}_T25600'] = run_round(4,
            f"LPF s={s} T=25600",
            lambda s=s: BestCombo(lpf_strength=s), T=25600, seeds=SEEDS_3)

    # ── T=25600 j_peak 扫描 ──
    for jp in [0.10, 0.15, 0.20]:
        all_r[f'R5_jpeak{jp}_T25600'] = run_round(5,
            f"j_peak={jp} T=25600",
            lambda p=jp: BestCombo(j_peak=p), T=25600, seeds=SEEDS_3)

    # ── T=51200 极限测试最优 ──
    all_r['R6_full_T51200'] = run_round(6, "全组合 T=51200(极限)",
        lambda: BestCombo(), T=51200, seeds=SEEDS_3)

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

    out = os.path.join(os.path.dirname(__file__), "stress_test23.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
