#!/usr/bin/env python3
"""
stress_test2.py — V8 GoldilocksFusion 多轮压测 + 对比 V7
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
import plugins_v2 as PV2
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
    print(f"\n{'='*80}")
    print(f"Round {round_num}: {name}")
    print(f"{'='*80}")
    results = {}
    for fam in ['p2p', 'standard', 'transformer']:
        m, s = eval_plugin(make_fn, fam)
        results[fam] = {'mean': m, 'std': s}
        print(f"  {fam:12s}  opt={m:+.4f}±{s:.4f}  {'OK' if m > 0 else 'FAIL'}")
    uni = all(results[f]['mean'] > 0 for f in ['p2p', 'standard', 'transformer'])
    print(f"  → 通用: {'YES' if uni else 'NO'}")
    results['universal'] = uni
    return results

def main():
    all_r = {}

    # V7 基线（默认参数）
    all_r['R0_V7_default'] = run_round(0, "V7 默认参数（基线）",
        lambda: Scaled(PV2.JerkAdaptiveFusion(), 1.0))

    # V8 Goldilocks 默认
    all_r['R1_V8_default'] = run_round(1, "V8 Goldilocks 默认 lam=0.5 alpha=0.5",
        lambda: PV8.GoldilocksFusion(lam=0.5, alpha=0.5))

    # V8 强参数
    all_r['R2_V8_strong'] = run_round(2, "V8 强参数 lam=0.6 alpha=0.6 kappa=0.15",
        lambda: PV8.GoldilocksFusion(lam=0.6, alpha=0.6, kappa=0.15))

    # V8 极强
    all_r['R3_V8_xstrong'] = run_round(3, "V8 极强 lam=0.8 alpha=0.7 kappa=0.20",
        lambda: PV8.GoldilocksFusion(lam=0.8, alpha=0.7, kappa=0.20))

    # V8 调门控范围（更窄的金发茄区）
    all_r['R4_V8_narrow_gate'] = run_round(4, "V8 窄门控 j_peak=0.03 j_hi=0.08",
        lambda: PV8.GoldilocksFusion(lam=0.6, alpha=0.6, j_peak=0.03, j_hi=0.08))

    # V8 宽门控
    all_r['R5_V8_wide_gate'] = run_round(5, "V8 宽门控 j_peak=0.1 j_hi=0.3",
        lambda: PV8.GoldilocksFusion(lam=0.6, alpha=0.6, j_peak=0.1, j_hi=0.3))

    # V8 极强+宽门控
    all_r['R6_V8_xstrong_wide'] = run_round(6, "V8 极强+宽门控 lam=0.8 alpha=0.7 j_hi=0.4",
        lambda: PV8.GoldilocksFusion(lam=0.8, alpha=0.7, j_peak=0.15, j_hi=0.4))

    # 汇总
    print(f"\n{'='*80}")
    print("压测汇总")
    print(f"{'='*80}")
    print(f"\n  {'配置':35s}  {'std':>8s}  {'trans':>8s}  {'p2p':>8s}  通用?")
    print("  " + "-" * 75)
    for k, v in all_r.items():
        uni = v.get('universal', False)
        print(f"  {k:35s}  {v['standard']['mean']:+.4f}  {v['transformer']['mean']:+.4f}  {v['p2p']['mean']:+.4f}  {'YES' if uni else 'NO'}")

    # 最优
    best_std = max(all_r.items(), key=lambda x: x[1]['standard']['mean'])
    best_uni = max((k for k, v in all_r.items() if v.get('universal')),
                   key=lambda k: all_r[k]['standard']['mean'], default=None)
    print(f"\n  最高 standard: {best_std[0]} = {best_std[1]['standard']['mean']:+.4f}")
    if best_uni:
        print(f"  最高通用 standard: {best_uni} = {all_r[best_uni]['standard']['mean']:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "stress_test2.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")

if __name__ == "__main__":
    main()