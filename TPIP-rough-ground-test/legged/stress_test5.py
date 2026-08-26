#!/usr/bin/env python3
"""
stress_test5.py — 第五轮：级联注入 + 在线轨迹优化器
突破 20% 天花板的新机制
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from plugins import BasePlugin
from plugins_v2 import AdaptiveEcho, GatedTidal, AdaptiveKV
from plugins_v8 import GoldilocksFusion

SEEDS = [42, 137, 2024, 7777, 314159]

class Scaled:
    def __init__(self, p, s): self.p, self.s = p, s
    def inject(self, a, **kw):
        base = a.copy()
        ap = self.p.inject(base, **kw)
        return base + self.s * (ap - base)

def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)

def eval_plugin(make_fn, family, seeds=SEEDS):
    opts = []
    for seed in seeds:
        base = BasePolicy(n_joints=6, family=family, seed=seed)
        sim_b = LeggedMicroSim(base, plugins=[], seed=seed)
        t = sim_b.run(goal=3.0, terrain=0.3)
        ci_b = compute_coherence(t['q'], t['dq'], central_diff(t['dq'], sim_b.dt), dt=sim_b.dt)['coherence_index']
        plug = make_fn()
        sim_p = LeggedMicroSim(base, plugins=[plug], seed=seed)
        t2 = sim_p.run(goal=3.0, terrain=0.3)
        ci_p = compute_coherence(t2['q'], t2['dq'], central_diff(t2['dq'], sim_p.dt), dt=sim_p.dt)['coherence_index']
        opts.append(opt(ci_p, ci_b))
    return float(np.mean(opts)), float(np.std(opts))

def run_round(round_num, name, make_fn):
    results = {}
    for fam in ['p2p', 'standard', 'transformer']:
        m, s = eval_plugin(make_fn, fam)
        results[fam] = {'mean': m, 'std': s}
    uni = all(results[f]['mean'] > 0 for f in ['p2p', 'standard', 'transformer'])
    results['universal'] = uni
    tag = "✓" if uni else "✗"
    print(f"  R{round_num}: {name:45s}  std={results['standard']['mean']:+.4f}  trans={results['transformer']['mean']:+.4f}  p2p={results['p2p']['mean']:+.4f}  {tag}")
    return results


# ── 级联注入：多趟精炼 ── #
class CascadeInjection(BasePlugin):
    """V10 = V8 级联：输出再过一趟 V8，二次精炼。"""
    def __init__(self, n_passes=2, lam=1.0, alpha=1.0, j_peak=0.1, j_hi=0.3):
        self.passes = [GoldilocksFusion(lam=lam, alpha=alpha, j_peak=j_peak, j_hi=j_hi)
                       for _ in range(n_passes)]
    def inject(self, a, **kw):
        for p in self.passes:
            a = p.inject(a, **kw)
        return a


# ── 在线轨迹优化器：Savitzky-Golay 风格平滑 ── #
class OnlineTrajectoryOptimizer(BasePlugin):
    """
    V11 = 在线轨迹优化：对最近 N 步做多项式拟合，
    用拟合值替换当前动作。金发茄门控控制强度。
    """
    def __init__(self, window=7, polyorder=2, strength=0.5,
                 j_lo=0.01, j_peak=0.1, j_hi=0.3):
        self.window = window
        self.polyorder = polyorder
        self.strength = strength
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

    def _polyfit_smooth(self, a):
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)
        if len(self._hist) < self.polyorder + 1:
            return a
        H = np.array(self._hist)
        n = len(H)
        x = np.arange(n, dtype=float)
        # 对每个关节做多项式拟合
        try:
            coeffs = np.polyfit(x, H, self.polyorder)
            predicted = np.polyval(coeffs, n)  # 预测下一步
            return predicted
        except:
            return a

    def inject(self, a, **kw):
        j = self._measure_jerk(a)
        gate = self._gate(j)
        if gate < 0.01:
            return a
        smoothed = self._polyfit_smooth(a)
        return a + self.strength * gate * (smoothed - a)


# ── V8 + 在线轨迹优化器 组合 ── #
class V8PlusOptimizer(BasePlugin):
    """V12 = V8(双极强) + V11(在线轨迹优化) 组合。"""
    def __init__(self):
        self._v8 = GoldilocksFusion(lam=1.0, alpha=1.0, j_peak=0.1, j_hi=0.3)
        self._opt = OnlineTrajectoryOptimizer(window=7, polyorder=2, strength=0.5)

    def inject(self, a, **kw):
        a = self._v8.inject(a, **kw)
        a = self._opt.inject(a, **kw)
        return a


def main():
    print("=" * 105)
    print("第五轮极限压测：级联注入 + 在线轨迹优化器")
    print("=" * 105)
    all_r = {}

    # 基线
    all_r['R0_V8_baseline'] = run_round(0, "V8 双极强(基线)",
        lambda: GoldilocksFusion(lam=1.0, alpha=1.0, j_peak=0.1, j_hi=0.3))

    # 级联 2 趟
    all_r['R1_cascade2'] = run_round(1, "V10 级联2趟",
        lambda: CascadeInjection(n_passes=2, lam=1.0, alpha=1.0))

    # 级联 3 趟
    all_r['R2_cascade3'] = run_round(2, "V10 级联3趟",
        lambda: CascadeInjection(n_passes=3, lam=1.0, alpha=1.0))

    # 在线轨迹优化器
    all_r['R3_opt_weak'] = run_round(3, "V11 在线优化 strength=0.3",
        lambda: OnlineTrajectoryOptimizer(strength=0.3))
    all_r['R4_opt_medium'] = run_round(4, "V11 在线优化 strength=0.5",
        lambda: OnlineTrajectoryOptimizer(strength=0.5))
    all_r['R5_opt_strong'] = run_round(5, "V11 在线优化 strength=0.8",
        lambda: OnlineTrajectoryOptimizer(strength=0.8))

    # V8 + 优化器组合
    all_r['R6_v8plus_weak'] = run_round(6, "V12 V8+优化(弱) strength=0.3",
        lambda: V8PlusOptimizer())

    # V8 + 优化器（强优化器）
    class V8PlusStrong(BasePlugin):
        def __init__(self):
            self._v8 = GoldilocksFusion(lam=1.0, alpha=1.0, j_peak=0.1, j_hi=0.3)
            self._opt = OnlineTrajectoryOptimizer(window=9, polyorder=3, strength=0.8)
        def inject(self, a, **kw):
            a = self._v8.inject(a, **kw)
            a = self._opt.inject(a, **kw)
            return a
    all_r['R7_v8plus_strong'] = run_round(7, "V12 V8+优化(强) window=9 poly=3 str=0.8",
        lambda: V8PlusStrong())

    # 级联2 + 优化器
    class Cascade2PlusOpt(BasePlugin):
        def __init__(self):
            self._c = CascadeInjection(n_passes=2, lam=1.0, alpha=1.0)
            self._opt = OnlineTrajectoryOptimizer(strength=0.5)
        def inject(self, a, **kw):
            a = self._c.inject(a, **kw)
            a = self._opt.inject(a, **kw)
            return a
    all_r['R8_cascade2_plus_opt'] = run_round(8, "V10级联2 + V11优化器",
        lambda: Cascade2PlusOpt())

    # 汇总
    print(f"\n{'='*105}")
    print("汇总（按 standard+transformer 均值排序）")
    print(f"{'='*105}")
    sorted_r = sorted(all_r.items(),
                      key=lambda x: (x[1]['standard']['mean'] + x[1]['transformer']['mean']) / 2,
                      reverse=True)
    print(f"  {'配置':35s}  {'std':>8s}  {'trans':>8s}  {'avg':>8s}  {'p2p':>8s}  通用?")
    print("  " + "-" * 80)
    for k, v in sorted_r:
        uni = v.get('universal', False)
        avg = (v['standard']['mean'] + v['transformer']['mean']) / 2
        print(f"  {k:35s}  {v['standard']['mean']:+.4f}  {v['transformer']['mean']:+.4f}  {avg:+.4f}  {v['p2p']['mean']:+.4f}  {'YES' if uni else 'NO'}")

    best = sorted_r[0]
    print(f"\n  最优: {best[0]} → std={best[1]['standard']['mean']:+.4f} trans={best[1]['transformer']['mean']:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "stress_test5.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")

if __name__ == "__main__":
    main()