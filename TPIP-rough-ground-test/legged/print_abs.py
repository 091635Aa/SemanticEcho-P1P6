#!/usr/bin/env python3
"""print_abs.py — 打印指定 族/场景 的 baseline 与 V6 的绝对 三维度+CI 值。"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cdiag
from legged_env import BasePolicy
from verify_architecture import BestCombo
import numpy as np
from plugins_robust import RobustComboV6

T = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
SEED = 99
cdiag.T = T
p0 = cdiag.train_uli()

for fam in ["standard", "transformer", "p2p"]:
    base = BasePolicy(n_joints=6, family=fam, seed=SEED)
    for scen in ["clean", "mild", "strong"]:
        mb, mp = cdiag.run_metrics(base, SEED, cdiag.SCEN[scen],
                                   (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=3.0, thr_hf=0.06, p=1.0, sh_cut=0.5)))
        b = {k: round(float(mb[k]), 4) for k in ["C","L","R","CI"]}
        p = {k: round(float(mp[k]), 4) for k in ["C","L","R","CI"]}
        d = {k: round(float(mp[k]-mb[k]), 4) for k in ["C","L","R","CI"]}
        print(f"[{scen}/{fam:11s}] base C{b['C']:+.3f} L{b['L']:+.3f} R{b['R']:+.3f} CI{b['CI']:+.3f} | "
              f"plug C{p['C']:+.3f} L{p['L']:+.3f} R{p['R']:+.3f} CI{p['CI']:+.3f} | "
              f"Δ dC{d['C']:+.3f} dL{d['L']:+.3f} dR{d['R']:+.3f} dCI{d['CI']:+.3f}", flush=True)