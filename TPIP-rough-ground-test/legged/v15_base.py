#!/usr/bin/env python3
"""v15_base.py — 探基线: 各族/场景下 C/L/R/CI 的绝对值, 判断 L/R 提升空间(天花板)。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy
import cdiag

T = 4000
cdiag.T = T
SEEDS = [99, 7]

def main():
    print(f"Baseline probe [T={T}]", flush=True)
    agg = {}
    for scen in ["clean", "mild", "strong"]:
        noise = cdiag.SCEN[scen]
        for fam in ["standard", "transformer", "p2p"]:
            vals = {k: [] for k in ("C", "L", "R", "CI")}
            for sd in SEEDS:
                base = BasePolicy(n_joints=6, family=fam, seed=sd)
                mb, _ = cdiag.run_metrics(base, sd, noise, (lambda: None))
                for k in vals:
                    vals[k].append(mb[k])
            agg[f"{scen}/{fam}"] = {k: float(np.mean(vals[k])) for k in vals}
            print(f"[{scen}/{fam:11s}] base C{agg[f'{scen}/{fam}']['C']:.3f} "
                  f"L{agg[f'{scen}/{fam}']['L']:.3f} R{agg[f'{scen}/{fam}']['R']:.3f} "
                  f"CI{agg[f'{scen}/{fam}']['CI']:.3f}", flush=True)

if __name__ == "__main__":
    main()