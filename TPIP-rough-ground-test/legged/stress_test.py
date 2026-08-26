#!/usr/bin/env python3
"""
stress_test.py — 多轮压测：持续推高足式机器人优化率
每轮尝试更强参数 + 新机制，输出标准RL/Transformer/P2P 优化率
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
import plugins_v2 as PV2
from plugins import BasePlugin

SEEDS = [42, 137, 2024, 7777, 314159]

class Scaled:
    def __init__(self, p, s): self.p, self.s = p, s
    def inject(self, a, **kw):
        base = a.copy()
        ap = self.p.inject(base, **kw)
        return base + self.s * (ap - base)

def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)

def eval_v7(params, family, seeds=SEEDS):
    opts = []
    for seed in seeds:
        base = BasePolicy(n_joints=6, family=family, seed=seed)
        sim_b = LeggedMicroSim(base, plugins=[], seed=seed)
        t = sim_b.run(goal=3.0, terrain=0.3)
        ci_b = compute_coherence(t['q'], t['dq'], central_diff(t['dq'], sim_b.dt), dt=sim_b.dt)['coherence_index']
        v7 = PV2.JerkAdaptiveFusion()
        v7._echo.lam = params.get('lam', 0.25)
        v7._tidal.alpha = params.get('alpha', 0.20)
        v7._kv.kappa = params.get('kappa', 0.12)
        plug = Scaled(v7, params.get('strength', 1.0))
        sim_p = LeggedMicroSim(base, plugins=[plug], seed=seed)
        t2 = sim_p.run(goal=3.0, terrain=0.3)
        ci_p = compute_coherence(t2['q'], t2['dq'], central_diff(t2['dq'], sim_p.dt), dt=sim_p.dt)['coherence_index']
        opts.append(opt(ci_p, ci_b))
    return float(np.mean(opts)), float(np.std(opts))

def run_round(round_num, params, extra_plugins=None):
    """跑一轮压测，返回各基座优化率。"""
    print(f"\n{'='*80}")
    print(f"压测 Round {round_num}: {params.get('desc', '')}")
    print(f"{'='*80}")
    results = {}
    for fam in ['p2p', 'standard', 'transformer']:
        opts = []
        for seed in SEEDS:
            base = BasePolicy(n_joints=6, family=fam, seed=seed)
            sim_b = LeggedMicroSim(base, plugins=[], seed=seed)
            t = sim_b.run(goal=3.0, terrain=0.3)
            ci_b = compute_coherence(t['q'], t['dq'], central_diff(t['dq'], sim_b.dt), dt=sim_b.dt)['coherence_index']

            # 构建 V7 with params
            v7 = PV2.JerkAdaptiveFusion()
            v7._echo.lam = params.get('lam', 0.25)
            v7._tidal.alpha = params.get('alpha', 0.20)
            v7._kv.kappa = params.get('kappa', 0.12)
            plug = Scaled(v7, params.get('strength', 1.0))

            sim_p = LeggedMicroSim(base, plugins=[plug], seed=seed)
            t2 = sim_p.run(goal=3.0, terrain=0.3)
            ci_p = compute_coherence(t2['q'], t2['dq'], central_diff(t2['dq'], sim_p.dt), dt=sim_p.dt)['coherence_index']
            opts.append(opt(ci_p, ci_b))
        m, s = np.mean(opts), np.std(opts)
        results[fam] = {'mean': float(m), 'std': float(s)}
        print(f"  {fam:12s}  opt={m:+.4f}±{s:.4f}  ({'OK' if m > 0 else 'FAIL'})")
    return results

def main():
    all_rounds = {}

    # ── Round 1: 最优参数基线 ──
    r1 = run_round(1, {'lam': 0.35, 'alpha': 0.40, 'kappa': 0.12, 'strength': 1.2, 'desc': '参数扫描最优'})
    all_rounds['round1'] = r1

    # ── Round 2: 更强注入 ──
    r2 = run_round(2, {'lam': 0.50, 'alpha': 0.50, 'kappa': 0.15, 'strength': 1.5, 'desc': '更强注入 lam=0.5 alpha=0.5 str=1.5'})
    all_rounds['round2'] = r2

    # ── Round 3: 极限注入 ──
    r3 = run_round(3, {'lam': 0.70, 'alpha': 0.60, 'kappa': 0.20, 'strength': 2.0, 'desc': '极限注入 lam=0.7 alpha=0.6 str=2.0'})
    all_rounds['round3'] = r3

    # ── Round 4: 极限+降kappa（减少KV干扰） ──
    r4 = run_round(4, {'lam': 0.80, 'alpha': 0.70, 'kappa': 0.05, 'strength': 2.5, 'desc': '极限+弱KV lam=0.8 alpha=0.7 kappa=0.05 str=2.5'})
    all_rounds['round4'] = r4

    # ── Round 5: 纯惯性回响极限（只靠V1） ──
    r5 = run_round(5, {'lam': 1.0, 'alpha': 0.0, 'kappa': 0.0, 'strength': 3.0, 'desc': '纯惯性极限 lam=1.0 str=3.0'})
    all_rounds['round5'] = r5

    # ── 汇总 ──
    print(f"\n{'='*80}")
    print("多轮压测汇总")
    print(f"{'='*80}")
    print(f"\n  {'轮次':30s}  {'standard':>10s}  {'transformer':>12s}  {'p2p':>10s}")
    print("  " + "-" * 70)
    for k, v in all_rounds.items():
        desc = ""
        if 'round1' in k: desc = "R1: 最优参数"
        elif 'round2' in k: desc = "R2: 更强注入"
        elif 'round3' in k: desc = "R3: 极限注入"
        elif 'round4' in k: desc = "R4: 极限+弱KV"
        elif 'round5' in k: desc = "R5: 纯惯性极限"
        print(f"  {desc:30s}  {v['standard']['mean']:+.4f}     {v['transformer']['mean']:+.4f}     {v['p2p']['mean']:+.4f}")

    # 最优轮
    best_round = max(all_rounds.items(), key=lambda x: x[1]['standard']['mean'])
    print(f"\n  最优轮: {best_round[0]} → standard={best_round[1]['standard']['mean']:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "stress_test.json")
    with open(out, "w") as f:
        json.dump(all_rounds, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")

if __name__ == "__main__":
    main()