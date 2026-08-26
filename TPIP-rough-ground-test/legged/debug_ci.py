#!/usr/bin/env python3
"""debug_ci.py - 打印各基座的基线CI和优化后CI，理解 ceiling"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from plugins_v8 import GoldilocksFusion

SEEDS = [42, 137, 2024, 7777, 314159]
wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}

class Cascade:
    def __init__(self):
        self.passes = [GoldilocksFusion(**wide) for _ in range(7)]
    def inject(self, a, **kw):
        for p in self.passes:
            a = p.inject(a, **kw)
        return a

print(f"{'family':12s} {'seed':>8s} {'CI_base':>10s} {'CI_opt':>10s} {'opt%':>8s} {'jerk_b':>10s} {'jerk_o':>10s}")
print("-" * 80)
for fam in ['p2p', 'standard', 'transformer']:
    base_cis, opt_cis = [], []
    for seed in SEEDS:
        base = BasePolicy(n_joints=6, family=fam, seed=seed)
        sim_b = LeggedMicroSim(base, plugins=[], seed=seed)
        t = sim_b.run(goal=3.0, terrain=0.3)
        ci_b = compute_coherence(t['q'], t['dq'],
                                 central_diff(t['dq'], sim_b.dt),
                                 dt=sim_b.dt)
        # 测 jerk
        a_arr = np.array(t['a'])
        jerk = np.diff(a_arr, axis=0)
        rms_jerk_b = float(np.sqrt(np.mean(jerk**2)))

        plug = Cascade()
        sim_p = LeggedMicroSim(base, plugins=[plug], seed=seed)
        t2 = sim_p.run(goal=3.0, terrain=0.3)
        ci_p = compute_coherence(t2['q'], t2['dq'],
                                 central_diff(t2['dq'], sim_p.dt),
                                 dt=sim_p.dt)
        a_arr2 = np.array(t2['a'])
        jerk2 = np.diff(a_arr2, axis=0)
        rms_jerk_o = float(np.sqrt(np.mean(jerk2**2)))

        opt_pct = (ci_p['coherence_index'] - ci_b['coherence_index']) / (1 - ci_b['coherence_index'] + 1e-9) * 100
        print(f"{fam:12s} {seed:>8d} {ci_b['coherence_index']:>10.4f} {ci_p['coherence_index']:>10.4f} {opt_pct:>7.2f}% {rms_jerk_b:>10.4f} {rms_jerk_o:>10.4f}")
        base_cis.append(ci_b['coherence_index'])
        opt_cis.append(ci_p['coherence_index'])
    print(f"  → {fam} 平均: base={np.mean(base_cis):.4f} opt={np.mean(opt_cis):.4f} opt%={ (np.mean(opt_cis)-np.mean(base_cis))/(1-np.mean(base_cis))*100:.2f}%")
    print()

# 打印 CI 子指标
print("\n=== 子指标分析（seed=42）===")
for fam in ['p2p', 'standard', 'transformer']:
    base = BasePolicy(n_joints=6, family=fam, seed=42)
    sim_b = LeggedMicroSim(base, plugins=[], seed=42)
    t = sim_b.run(goal=3.0, terrain=0.3)
    ci_b = compute_coherence(t['q'], t['dq'],
                             central_diff(t['dq'], sim_b.dt),
                             dt=sim_b.dt)
    plug = Cascade()
    sim_p = LeggedMicroSim(base, plugins=[plug], seed=42)
    t2 = sim_p.run(goal=3.0, terrain=0.3)
    ci_p = compute_coherence(t2['q'], t2['dq'],
                             central_diff(t2['dq'], sim_p.dt),
                             dt=sim_p.dt)
    print(f"\n{fam}:")
    print(f"  base:  s_smooth={ci_b.get('s_smooth','N/A')}  p_coinc={ci_b.get('p_phase_coincidence','N/A')}  rms_jerk={ci_b.get('rms_jerk','N/A')}  CI={ci_b['coherence_index']:.4f}")
    print(f"  opt:   s_smooth={ci_p.get('s_smooth','N/A')}  p_coinc={ci_p.get('p_phase_coincidence','N/A')}  rms_jerk={ci_p.get('rms_jerk','N/A')}  CI={ci_p['coherence_index']:.4f}")
