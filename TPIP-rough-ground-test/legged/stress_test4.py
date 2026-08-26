#!/usr/bin/env python3
"""
stress_test4.py — 第四轮：R3最优区微调 + 多尺度双回响新机制
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from plugins import BasePlugin
from plugins_v2 import AdaptiveEcho, GatedTidal, AdaptiveKV

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
    print(f"  R{round_num}: {name:42s}  std={m:+.4f}  trans={results['transformer']['mean']:+.4f}  p2p={results['p2p']['mean']:+.4f}  {tag}")
    return results


# ── 多尺度双回响 V9 ── #
class MultiScaleFusion(BasePlugin):
    """
    V9 = V8 改进：双尺度回响（短时+长时）+ 金发茄门控。
    短时回响(pool=8)捕获快速动态，长时回响(pool=30)捕获慢趋势。
    """
    def __init__(self, lam_s=0.5, lam_l=0.5, alpha=0.5, kappa=0.12,
                 j_lo=0.01, j_peak=0.1, j_hi=0.3):
        self._echo_s = AdaptiveEcho(lam=lam_s, pool=8)
        self._echo_l = AdaptiveEcho(lam=lam_l, pool=30)
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
            a_mid = self._echo_s.inject(a_mid, **kw)
            a_mid = self._echo_l.inject(a_mid, **kw)
            a_mid = self._tidal.inject(a_mid, **kw)
            a = a + gate * (a_mid - a)
        return a


def main():
    print("=" * 100)
    print("第四轮压测：R3微调 + V9 多尺度双回响")
    print("=" * 100)
    from plugins_v8 import GoldilocksFusion
    all_r = {}

    # R3 基线
    all_r['R0_R3_baseline'] = run_round(0, "R3基线 lam=1.0 alpha=1.0",
        lambda: GoldilocksFusion(lam=1.0, alpha=1.0, j_peak=0.1, j_hi=0.3))

    # 微调
    all_r['R1_lam_0.9'] = run_round(1, "lam=0.9 alpha=1.0",
        lambda: GoldilocksFusion(lam=0.9, alpha=1.0, j_peak=0.1, j_hi=0.3))
    all_r['R2_lam_1.1'] = run_round(2, "lam=1.1 alpha=1.0",
        lambda: GoldilocksFusion(lam=1.1, alpha=1.0, j_peak=0.1, j_hi=0.3))
    all_r['R3_alpha_0.9'] = run_round(3, "lam=1.0 alpha=0.9",
        lambda: GoldilocksFusion(lam=1.0, alpha=0.9, j_peak=0.1, j_hi=0.3))
    all_r['R4_alpha_1.1'] = run_round(4, "lam=1.0 alpha=1.1",
        lambda: GoldilocksFusion(lam=1.0, alpha=1.1, j_peak=0.1, j_hi=0.3))
    all_r['R5_both_1.1'] = run_round(5, "lam=1.1 alpha=1.1",
        lambda: GoldilocksFusion(lam=1.1, alpha=1.1, j_peak=0.1, j_hi=0.3))
    all_r['R6_jpeak_0.05'] = run_round(6, "lam=1.0 alpha=1.0 j_peak=0.05",
        lambda: GoldilocksFusion(lam=1.0, alpha=1.0, j_peak=0.05, j_hi=0.3))
    all_r['R7_jpeak_0.15'] = run_round(7, "lam=1.0 alpha=1.0 j_peak=0.15",
        lambda: GoldilocksFusion(lam=1.0, alpha=1.0, j_peak=0.15, j_hi=0.3))

    # V9 多尺度
    all_r['R8_V9_default'] = run_round(8, "V9 多尺度默认 lam_s=0.5 lam_l=0.5 alpha=1.0",
        lambda: MultiScaleFusion(lam_s=0.5, lam_l=0.5, alpha=1.0))
    all_r['R9_V9_strong'] = run_round(9, "V9 多尺度强 lam_s=0.8 lam_l=0.8 alpha=1.0",
        lambda: MultiScaleFusion(lam_s=0.8, lam_l=0.8, alpha=1.0))
    all_r['R10_V9_xstrong'] = run_round(10, "V9 多尺度极强 lam_s=1.0 lam_l=1.0 alpha=1.0",
        lambda: MultiScaleFusion(lam_s=1.0, lam_l=1.0, alpha=1.0))
    all_r['R11_V9_asym'] = run_round(11, "V9 非对称 lam_s=1.5 lam_l=0.5 alpha=1.0",
        lambda: MultiScaleFusion(lam_s=1.5, lam_l=0.5, alpha=1.0))

    # 汇总
    print(f"\n{'='*100}")
    print("汇总（按 standard 排序）")
    print(f"{'='*100}")
    sorted_r = sorted(all_r.items(), key=lambda x: x[1]['standard']['mean'], reverse=True)
    print(f"  {'配置':35s}  {'std':>8s}  {'trans':>8s}  {'p2p':>8s}  通用?")
    print("  " + "-" * 75)
    for k, v in sorted_r:
        uni = v.get('universal', False)
        print(f"  {k:35s}  {v['standard']['mean']:+.4f}  {v['transformer']['mean']:+.4f}  {v['p2p']['mean']:+.4f}  {'YES' if uni else 'NO'}")

    best = sorted_r[0]
    print(f"\n  最优: {best[0]} → std={best[1]['standard']['mean']:+.4f} trans={best[1]['transformer']['mean']:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "stress_test4.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")

if __name__ == "__main__":
    main()