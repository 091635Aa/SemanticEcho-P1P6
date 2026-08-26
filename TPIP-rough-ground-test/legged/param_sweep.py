#!/usr/bin/env python3
"""
param_sweep.py — V7 参数扫描，找使平滑基座(standard/transformer)突破 10% 的最优配置
扫描：lam, alpha, kappa, gate_kv, jerk_low, jerk_high
"""
import sys, os, numpy as np, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
import plugins_v2 as PV2

SEEDS = [42, 137, 2024]

class Scaled:
    def __init__(self, p, s): self.p, self.s = p, s
    def inject(self, a, **kw):
        base = a.copy()
        ap = self.p.inject(base, **kw)
        return base + self.s * (ap - base)

def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)

def eval_v7(params, family, seeds=SEEDS):
    """用给定参数构建 V7 并测试。"""
    opts = []
    for seed in seeds:
        base = BasePolicy(n_joints=6, family=family, seed=seed)
        sim_b = LeggedMicroSim(base, plugins=[], seed=seed)
        t = sim_b.run(goal=3.0, terrain=0.3)
        ci_b = compute_coherence(t['q'], t['dq'], central_diff(t['dq'], sim_b.dt), dt=sim_b.dt)['coherence_index']

        # 构建 V7 with custom params
        v7 = PV2.JerkAdaptiveFusion()
        v7._echo.lam = params['lam']
        v7._tidal.alpha = params['alpha']
        v7._kv.kappa = params['kappa']
        # 通过修改内部行为来设置 gate 参数
        plug = Scaled(v7, params['strength'])

        sim_p = LeggedMicroSim(base, plugins=[plug], seed=seed)
        t2 = sim_p.run(goal=3.0, terrain=0.3)
        ci_p = compute_coherence(t2['q'], t2['dq'], central_diff(t2['dq'], sim_p.dt), dt=sim_p.dt)['coherence_index']
        opts.append(opt(ci_p, ci_b))
    return float(np.mean(opts)), float(np.std(opts))

def main():
    # 参数网格
    grid = {
        'lam': [0.15, 0.25, 0.35, 0.45],
        'alpha': [0.15, 0.20, 0.30, 0.40],
        'kappa': [0.08, 0.12, 0.18, 0.25],
        'strength': [0.8, 1.0, 1.2, 1.5],
    }

    print("=" * 80)
    print("V7 参数扫描（目标：平滑基座 standard/transformer > 10%）")
    print("=" * 80)

    best = {'score': -999, 'params': None}
    results = []

    for lam, alpha, kappa, strength in itertools.product(
            grid['lam'], grid['alpha'], grid['kappa'], grid['strength']):
        params = {'lam': lam, 'alpha': alpha, 'kappa': kappa, 'strength': strength}
        std_m, std_s = eval_v7(params, 'standard')
        trn_m, trn_s = eval_v7(params, 'transformer')
        p2p_m, _ = eval_v7(params, 'p2p')

        # 评分：平滑基座均值 + p2p 正向惩罚
        score = (std_m + trn_m) / 2
        if p2p_m < 0:
            score -= 0.5  # p2p 负则重罚

        results.append({
            'params': params, 'std': std_m, 'trn': trn_m, 'p2p': p2p_m, 'score': score
        })

        if score > best['score'] and p2p_m > 0:
            best = {'score': score, 'params': params, 'std': std_m, 'trn': trn_m, 'p2p': p2p_m}

    # 排序输出 top 10
    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"\n  Top 10 配置（按平滑基座均值排序）：")
    print(f"  {'lam':>5s} {'alpha':>6s} {'kappa':>6s} {'str':>5s}  {'std':>8s}  {'trn':>8s}  {'p2p':>8s}  {'score':>8s}")
    print("  " + "-" * 70)
    for r in results[:10]:
        p = r['params']
        print(f"  {p['lam']:.2f}  {p['alpha']:.2f}   {p['kappa']:.2f}   {p['strength']:.1f}   "
              f"{r['std']:+.4f}   {r['trn']:+.4f}   {r['p2p']:+.4f}   {r['score']:+.4f}")

    print(f"\n  最优配置: {best['params']}")
    print(f"  standard={best['std']:+.4f}  transformer={best['trn']:+.4f}  p2p={best['p2p']:+.4f}")

    # 写入
    import json
    out = os.path.join(os.path.dirname(__file__), "param_sweep.json")
    with open(out, "w") as f:
        json.dump({'best': best, 'top10': results[:10]}, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")

if __name__ == "__main__":
    main()