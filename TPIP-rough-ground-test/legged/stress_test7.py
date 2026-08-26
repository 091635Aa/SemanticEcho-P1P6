#!/usr/bin/env python3
"""
stress_test7.py — 第七轮：5趟级联优化 + 7趟 + 宽门保护P2P
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
    avg = (results['standard']['mean'] + results['transformer']['mean']) / 2
    print(f"  R{round_num}: {name:50s}  std={results['standard']['mean']:+.4f}  trans={results['transformer']['mean']:+.4f}  avg={avg:+.4f}  p2p={results['p2p']['mean']:+.4f}  {tag}")
    return results

class Cascade(BasePlugin):
    def __init__(self, configs):
        self.passes = [GoldilocksFusion(**c) for c in configs]
    def inject(self, a, **kw):
        for p in self.passes:
            a = p.inject(a, **kw)
        return a

def main():
    print("=" * 120)
    print("第七轮极限压测：5趟级联优化 + 7趟 + 宽门保护P2P + 混合异构")
    print("=" * 120)
    all_r = {}
    b = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.1, 'j_hi': 0.3}
    wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}

    # 基线
    all_r['R0_cascade5'] = run_round(0, "级联5(基线)", lambda: Cascade([b]*5))

    # 5趟宽门
    all_r['R1_cascade5_wide'] = run_round(1, "级联5 宽门", lambda: Cascade([wide]*5))

    # 5趟 强echo
    all_r['R2_cascade5_strong_echo'] = run_round(2, "级联5 lam=1.5",
        lambda: Cascade([{'lam': 1.5, 'alpha': 1.0, 'j_peak': 0.1, 'j_hi': 0.3}]*5))

    # 5趟 强tidal
    all_r['R3_cascade5_strong_tidal'] = run_round(3, "级联5 alpha=1.5",
        lambda: Cascade([{'lam': 1.0, 'alpha': 1.5, 'j_peak': 0.1, 'j_hi': 0.3}]*5))

    # 7趟
    all_r['R4_cascade7'] = run_round(4, "级联7", lambda: Cascade([b]*7))

    # 7趟宽门
    all_r['R5_cascade7_wide'] = run_round(5, "级联7 宽门", lambda: Cascade([wide]*7))

    # 混合异构5趟：宽窄交替 + 前强后稳
    all_r['R6_mixed5'] = run_round(6, "混合5趟 宽窄交替+递减",
        lambda: Cascade([
            {'lam': 1.5, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5},
            {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.1, 'j_hi': 0.3},
            {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5},
            {'lam': 0.7, 'alpha': 0.7, 'j_peak': 0.1, 'j_hi': 0.3},
            {'lam': 0.5, 'alpha': 0.5, 'j_peak': 0.15, 'j_hi': 0.5},
        ]))

    # 混合异构5趟：全宽门+递增
    all_r['R7_mixed5_inc'] = run_round(7, "混合5趟 全宽门+递增",
        lambda: Cascade([
            {'lam': 0.5, 'alpha': 0.5, 'j_peak': 0.15, 'j_hi': 0.5},
            {'lam': 0.8, 'alpha': 0.8, 'j_peak': 0.15, 'j_hi': 0.5},
            {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5},
            {'lam': 1.2, 'alpha': 1.2, 'j_peak': 0.15, 'j_hi': 0.5},
            {'lam': 1.5, 'alpha': 1.5, 'j_peak': 0.15, 'j_hi': 0.5},
        ]))

    # 5趟 双极强+宽门
    all_r['R8_cascade5_xstrong_wide'] = run_round(8, "级联5 双极强+宽门 lam=1.5 alpha=1.5",
        lambda: Cascade([{'lam': 1.5, 'alpha': 1.5, 'j_peak': 0.15, 'j_hi': 0.5}]*5))

    # 3趟宽门（对照）
    all_r['R9_cascade3_wide'] = run_round(9, "级联3 宽门(对照)",
        lambda: Cascade([wide]*3))

    # 汇总
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
    print(f"\n  最优: {best[0]} → avg={best[1]['standard']['mean']+best[1]['transformer']['mean']/2:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "stress_test7.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")

if __name__ == "__main__":
    main()