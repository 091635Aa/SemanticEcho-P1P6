#!/usr/bin/env python3
"""v11_diag.py — 分解 transformer 在 PhaseRecomb 下的 CI 子指标，定位 R/CI 小损根因。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from plugins_robust import PhaseRecomb
import cdiag

T = 5000; SEED = 99

def breakdown(label, traj):
    q = traj["q"]; dq = traj["dq"]
    r = compute_coherence(q, dq, central_diff(dq, cdiag.DT), dt=cdiag.DT)
    mb = {"C": cdiag.c_cons(traj), "L": cdiag.l_link(traj), "R": cdiag.r_ref(traj)}
    print(f"  {label:10s} CI={r['coherence_index']:.4f} "
          f"S_smooth={r['s_smooth']:.4f}(rms_jerk={r['rms_jerk']:.5f}) "
          f"P_coinc={r['p_phase_coincidence']:.4f} "
          f"gaitscale={r['gait_jerk_scale']:.5f} "
          f"C={mb['C']:.3f} L={mb['L']:.3f} R={mb['R']:.3f}", flush=True)

for scen in ["clean"]:
    noise = cdiag.SCEN[scen]
    for fam in ["standard", "transformer"]:
        print(f"--- {scen}/{fam} ---", flush=True)
        base = BasePolicy(n_joints=6, family=fam, seed=SEED)
        s = LeggedMicroSim(base, T=T, dt=cdiag.DT, seed=SEED, **noise)
        tb = s.run(); breakdown("base", tb)
        p = PhaseRecomb(lam=0.7, alpha=0.05); p.reset()
        ts = LeggedMicroSim(base, plugins=[p], T=T, dt=cdiag.DT, seed=SEED, **noise)
        tp = ts.run(); breakdown("v11-noU", tp)
        p2 = PhaseRecomb(lam=0.4, alpha=0.05, adapt=True, lam_ref=0.45); p2.reset()
        ts2 = LeggedMicroSim(base, plugins=[p2], T=T, dt=cdiag.DT, seed=SEED, **noise)
        tp2 = ts2.run(); breakdown("v11-ada", tp2)