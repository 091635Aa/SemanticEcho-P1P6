#!/usr/bin/env python3
"""v16 — 一致性/连贯最大化扫: 在 V14(g0.8) 骨架上扫 lam & alpha,
守住 L≥V14, R≥V14, ΣΔC/ΣΔCI 最大化且无净损失. L/R 已近饱和, C/CI 才是空间."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy
from plugins_robust import PhaseRecomb
import cdiag

T = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
cdiag.T = T
SEEDS = [99, 7]
FAMS = ["standard", "transformer"]
SCENS = ["clean", "mild", "strong"]

def make(lam, alpha, gbl=0.8):
    return PhaseRecomb(lam=lam, alpha=alpha, blend=True,
                       r2b_lo=0.45, r2b_hi=0.85, min_cyc=2, gbl=gbl)

CFGS = {
    "v14 lam1.0 a.05": dict(lam=1.0, alpha=0.05),
    "lam1.1 a.05":     dict(lam=1.1, alpha=0.05),
    "lam1.2 a.05":     dict(lam=1.2, alpha=0.05),
    "lam1.0 a.08":     dict(lam=1.0, alpha=0.08),
    "lam1.1 a.08":     dict(lam=1.1, alpha=0.08),
}

def main():
    print(f"V16 lam/alpha sweep [T={T}]", flush=True)
    out = {"T": T, "cfgs": {}}
    for name, kw in CFGS.items():
        agg = {k: [] for k in ("C", "L", "R", "CI")}
        nneg = {k: 0 for k in ("C", "L", "R", "CI")}
        for scen in SCENS:
            noise = cdiag.SCEN[scen]
            for fam in FAMS:
                for sd in SEEDS:
                    base = BasePolicy(n_joints=6, family=fam, seed=sd)
                    mb, _ = cdiag.run_metrics(base, sd, noise, (lambda: None))
                    _, mp = cdiag.run_metrics(base, sd, noise, (lambda: make(**kw)))
                    for k in agg:
                        d = mp[k] - mb[k]; agg[k].append(d)
                        if d < 0: nneg[k] += 1
        tot = {k: float(np.sum(agg[k])) for k in agg}
        mean = {k: float(np.mean(agg[k])) for k in agg}
        out["cfgs"][name] = {"sum": tot, "mean": mean, "nneg": nneg}
        print(f"[{name:15s}] ΣΔC{tot['C']:+.3f} ΣΔL{tot['L']:+.3f} ΣΔR{tot['R']:+.3f} "
              f"ΣΔCI{tot['CI']:+.3f} | 均ΔC{mean['C']:+.3f} ΔCI{mean['CI']:+.3f} | "
              f"nneg(C/L/R/CI)={nneg['C']}/{nneg['L']}/{nneg['R']}/{nneg['CI']}", flush=True)
    json.dump({"out": out}, open("v16_sweep.json", "w"), ensure_ascii=False, indent=1, default=float)

if __name__ == "__main__":
    main()