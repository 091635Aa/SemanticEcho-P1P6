#!/usr/bin/env python3
"""
stress_test6.py — 第六轮：级联深度优化 + 异构级联
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from plugins import BasePlugin
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
    print(f"  R{round_num}: {name:50s}  std={results['standard']['mean']:+.4f}  trans={results['transformer']['mean']:+.4f}  p2p={results['p2p']['mean']:+.4f}  {tag}")
    return results

class Cascade(BasePlugin):
    def __init__(self, configs):
        self.passes = [GoldilocksFusion(**c) for c in configs]
    def inject(self, a, **kw):
        for p in self.passes:
            a = p.inject(a, **kw)
        return a

def main():
    print("=" * 115)
    print("第六轮极限压测：级联深度优化 + 异构级联")
    print("=" * 115)
    all_r = {}

    base_cfg = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.1, 'j_hi': 0.3}

    # 级联 3 基线
    all_r['R0_cascade3'] = run_round(0, "级联3(基线)",
        lambda: Cascade([base_cfg]*3))

    # 级联 4
    all_r['R1_cascade4'] = run_round(1, "级联4",
        lambda: Cascade([base_cfg]*4))

    # 级联 5
    all_r['R2_cascade5'] = run_round(2, "级联5",
        lambda: Cascade([base_cfg]*5))

    # 级联 6
    all_r['R3_cascade6'] = run_round(3, "级联6",
        lambda: Cascade([base_cfg]*6))

    # 级联 8
    all_r['R4_cascade8'] = run_round(4, "级联8",
        lambda: Cascade([base_cfg]*8))

    # 级联 10
    all_r['R5_cascade10'] = run_round(5, "级联10",
        lambda: Cascade([base_cfg]*10))

    # 异构：前强后弱（第1趟强 echo，后面弱 tidal）
    all_r['R6_hetero_sw'] = run_round(6, "异构 前强后弱",
        lambda: Cascade([
            {'lam': 1.5, 'alpha': 1.0, 'j_peak': 0.1, 'j_hi': 0.3},
            {'lam': 0.5, 'alpha': 0.5, 'j_peak': 0.1, 'j_hi': 0.3},
            {'lam': 0.3, 'alpha': 0.3, 'j_peak': 0.1, 'j_hi': 0.3},
        ]))

    # 异构：递增（弱→强）
    all_r['R7_hetero_inc'] = run_round(7, "异构 递增 弱→强",
        lambda: Cascade([
            {'lam': 0.3, 'alpha': 0.3, 'j_peak': 0.1, 'j_hi': 0.3},
            {'lam': 0.7, 'alpha': 0.7, 'j_peak': 0.1, 'j_hi': 0.3},
            {'lam': 1.5, 'alpha': 1.5, 'j_peak': 0.1, 'j_hi': 0.3},
        ]))

    # 异构：宽窄门交替
    all_r['R8_hetero_altgate'] = run_round(8, "异构 宽窄门交替",
        lambda: Cascade([
            {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.05, 'j_hi': 0.2},
            {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.4},
            {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.05, 'j_hi': 0.2},
        ]))

    # 级联3 + Scaled 1.5
    all_r['R9_cascade3_scaled'] = run_round(9, "级联3 Scaled×1.5",
        lambda: Scaled(Cascade([base_cfg]*3), 1.5))

    # 汇总
    print(f"\n{'='*115}")
    print("汇总（按 std+trans 均值排序）")
    print(f"{'='*115}")
    sorted_r = sorted(all_r.items(),
                      key=lambda x: (x[1]['standard']['mean'] + x[1]['transformer']['mean']) / 2,
                      reverse=True)
    print(f"  {'配置':30s}  {'std':>8s}  {'trans':>8s}  {'avg':>8s}  {'p2p':>8s}  通用?")
    print("  " + "-" * 80)
    for k, v in sorted_r:
        uni = v.get('universal', False)
        avg = (v['standard']['mean'] + v['transformer']['mean']) / 2
        print(f"  {k:30s}  {v['standard']['mean']:+.4f}  {v['transformer']['mean']:+.4f}  {avg:+.4f}  {v['p2p']['mean']:+.4f}  {'YES' if uni else 'NO'}")

    best = sorted_r[0]
    print(f"\n  最优: {best[0]} → std={best[1]['standard']['mean']:+.4f} trans={best[1]['transformer']['mean']:+.4f} avg={(best[1]['standard']['mean']+best[1]['transformer']['mean'])/2:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "stress_test6.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")

if __name__ == "__main__":
    main()