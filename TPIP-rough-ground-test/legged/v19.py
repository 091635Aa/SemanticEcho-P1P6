#!/usr/bin/env python3
"""v19.py — clean/transformer C 是否系统损失? 对照 T(4000/8000) × seed数(2/5)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy
from plugins_robust import PhaseRecomb
import cdiag

def make():
    return PhaseRecomb(lam=1.0, alpha=0.05, blend=True,
                       r2b_lo=0.45, r2b_hi=0.85, min_cyc=2, gbl=0.8)

def cell(T, seeds, fam, scen="clean"):
    cdiag.T = T
    noise = cdiag.SCEN[scen]
    keys = ("C", "L", "R", "CI")
    base = {k: [] for k in keys}
    plug = {k: [] for k in keys}
    for sd in seeds:
        b = BasePolicy(n_joints=6, family=fam, seed=sd)
        mb, _ = cdiag.run_metrics(b, sd, noise, (lambda: None))
        _, mp = cdiag.run_metrics(b, sd, noise, make)
        for k in keys:
            base[k].append(mb[k]); plug[k].append(mp[k])
    b = {k: float(np.mean(base[k])) for k in keys}
    d = {k: float(np.mean(plug[k])) - b[k] for k in keys}
    return b, d

def main():
    S2 = [99, 7]
    S5 = [99, 7, 42, 123, 5]
    print("clean/transformer (饱和族, 基线C≈0.62):", flush=True)
    for T in [4000, 8000]:
        for nm, seeds in [("2seed", S2), ("5seed", S5)]:
            b, d = cell(T, seeds, "transformer")
            print(f"[T={T} {nm:6s}] baseC{b['C']:.3f} | ΔC{d['C']:+.4f} ΔCI{d['CI']:+.4f} "
                  f"ΔR{d['R']:+.4f} ΔL{d['L']:+.4f}", flush=True)
    print("clean/standard (非饱和族 对照):", flush=True)
    for T in [4000, 8000]:
        b, d = cell(T, S5, "standard")
        print(f"[T={T} 5seed] baseC{b['C']:.3f} | ΔC{d['C']:+.4f} ΔCI{d['CI']:+.4f} "
              f"ΔR{d['R']:+.4f} ΔL{d['L']:+.4f}", flush=True)

if __name__ == "__main__":
    main()