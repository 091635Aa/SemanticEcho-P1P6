#!/usr/bin/env python3
"""diagnose.py — 诊断各基座的 CI 子指标，找平滑基座的可提升空间"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from run_v2 import Scaled
import plugins_v2 as PV2

print("CI 子指标诊断（bare vs V5_SmartFusion 最优）")
print("=" * 80)
for fam in ["p2p", "standard", "transformer"]:
    for label, strength in [("bare", 0.0), ("V5", 1.0)]:
        make = PV2.PLUGINS_V2["V5_SmartFusion"]
        plug = Scaled(make(), strength) if strength > 0 else None
        base = BasePolicy(n_joints=6, family=fam, seed=42)
        sim = LeggedMicroSim(base, plugins=[plug] if plug else [], seed=42)
        traj = sim.run(goal=3.0, terrain=0.3)
        q, dq = traj["q"], traj["dq"]
        qd = central_diff(dq, sim.dt)
        r = compute_coherence(q, dq, qd, dt=sim.dt)
        print(f"  {fam:12s} {label:4s}  CI={r['coherence_index']:.4f}  "
              f"S_smooth={r['s_smooth']:.4f}  P_coinc={r['p_phase_coincidence']:.4f}  "
              f"rms_jerk={r['rms_jerk']:.4f}")
    print()