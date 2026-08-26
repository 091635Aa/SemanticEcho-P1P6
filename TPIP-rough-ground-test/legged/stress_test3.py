#!/usr/bin/env python3
"""
stress_test3.py — 第三轮压测：极限推高 + 多尺度新机制
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
import plugins_v8 as PV8

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
    tag = "✓通用" if uni else "✗"
    print(f"  R{round_num}: {name:45s}  std={results['standard']['mean']:+.4f}  "
          f"trans={results['transformer']['mean']:+.4f}  p2p={results['p2p']['mean']:+.4f}  {tag}")
    return results

def main():
    print("=" * 95)
    print("第三轮极限压测：V8 Goldilocks 极限配置 + 多尺度变体")
    print("=" * 95)
    all_r = {}

    # 基线（上轮最优）
    all_r['R0_V8_wide'] = run_round(0, "V8 宽门控(上轮最优)",
        lambda: PV8.GoldilocksFusion(lam=0.6, alpha=0.6, j_peak=0.1, j_hi=0.3))

    # 更强 echo
    all_r['R1_echo_xstrong'] = run_round(1, "V8 echo极强 lam=1.0 alpha=0.6",
        lambda: PV8.GoldilocksFusion(lam=1.0, alpha=0.6, j_peak=0.1, j_hi=0.3))

    # 更强 tidal
    all_r['R2_tidal_xstrong'] = run_round(2, "V8 tidal极强 lam=0.6 alpha=1.0",
        lambda: PV8.GoldilocksFusion(lam=0.6, alpha=1.0, j_peak=0.1, j_hi=0.3))

    # 双强
    all_r['R3_both_xstrong'] = run_round(3, "V8 双极强 lam=1.0 alpha=1.0",
        lambda: PV8.GoldilocksFusion(lam=1.0, alpha=1.0, j_peak=0.1, j_hi=0.3))

    # 双强+更宽门控
    all_r['R4_xstrong_xwide'] = run_round(4, "V8 双极强+超宽门 j_hi=0.5",
        lambda: PV8.GoldilocksFusion(lam=1.0, alpha=1.0, j_peak=0.15, j_hi=0.5))

    # 双强+kappa增强
    all_r['R5_xstrong_kv'] = run_round(5, "V8 双极强+强KV kappa=0.3",
        lambda: PV8.GoldilocksFusion(lam=1.0, alpha=1.0, kappa=0.3, j_peak=0.1, j_hi=0.3))

    # Scaled 双极强
    all_r['R6_scaled_x2'] = run_round(6, "V8 双极强 Scaled×2",
        lambda: Scaled(PV8.GoldilocksFusion(lam=1.0, alpha=1.0, j_peak=0.1, j_hi=0.3), 2.0))

    # Scaled 双极强+超宽
    all_r['R7_scaled_x3_xwide'] = run_round(7, "V8 双极强 Scaled×3+超宽",
        lambda: Scaled(PV8.GoldilocksFusion(lam=1.0, alpha=1.0, j_peak=0.15, j_hi=0.5), 3.0))

    # 极限: lam=2 alpha=2
    all_r['R8_absurd'] = run_round(8, "V8 荒谬级 lam=2.0 alpha=2.0",
        lambda: PV8.GoldilocksFusion(lam=2.0, alpha=2.0, j_peak=0.1, j_hi=0.3))

    # 汇总
    print(f"\n{'='*95}")
    print("汇总（按 standard 优化率排序）")
    print(f"{'='*95}")
    sorted_r = sorted(all_r.items(), key=lambda x: x[1]['standard']['mean'], reverse=True)
    print(f"  {'配置':35s}  {'std':>8s}  {'trans':>8s}  {'p2p':>8s}  通用?")
    print("  " + "-" * 70)
    for k, v in sorted_r:
        uni = v.get('universal', False)
        print(f"  {k:35s}  {v['standard']['mean']:+.4f}  {v['transformer']['mean']:+.4f}  {v['p2p']['mean']:+.4f}  {'YES' if uni else 'NO'}")

    best = sorted_r[0]
    print(f"\n  最优: {best[0]} → standard={best[1]['standard']['mean']:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "stress_test3.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")

if __name__ == "__main__":
    main()