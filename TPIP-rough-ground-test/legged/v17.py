#!/usr/bin/env python3
"""v17.py — 阶段1 预言验证: r2门控模板(V17) vs V14.
预言: noisy(transformer&standard) C/CI 抬升、L/R 不降、clean 不变(无净损失)."""
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

def make_v14():
    return PhaseRecomb(lam=1.0, alpha=0.05, blend=True,
                       r2b_lo=0.45, r2b_hi=0.85, min_cyc=2, gbl=0.8)
def make_v17(tpl_r2lo, tpl_r2hi):
    return PhaseRecomb(lam=1.0, alpha=0.05, blend=True,
                       r2b_lo=0.45, r2b_hi=0.85, min_cyc=2, gbl=0.8,
                       tpl_gate=True, tpl_r2lo=tpl_r2lo, tpl_r2hi=tpl_r2hi)

V17 = {
    "gA .30/.60": dict(tpl_r2lo=0.30, tpl_r2hi=0.60),
    "gB .40/.70": dict(tpl_r2lo=0.40, tpl_r2hi=0.70),
    "gC .45/.75": dict(tpl_r2lo=0.45, tpl_r2hi=0.75),
}

def main():
    print(f"V17 r2门控模板 [T={T}]", flush=True)
    out = {"T": T, "cfgs": {}}
    data = {}
    for scen in SCENS:
        noise = cdiag.SCEN[scen]
        for fam in FAMS:
            for sd in SEEDS:
                base = BasePolicy(n_joints=6, family=fam, seed=sd)
                mb, _ = cdiag.run_metrics(base, sd, noise, (lambda: None))
                _, mp14 = cdiag.run_metrics(base, sd, noise, make_v14)
                key = f"{scen}/{fam}/{sd}"
                data[key] = {"base": mb, "v14": {k: mp14[k]-mb[k] for k in mb}}
    for nm, c in V17.items():
        agg = {k: [] for k in ("C", "L", "R", "CI")}
        for scen in SCENS:
            noise = cdiag.SCEN[scen]
            for fam in FAMS:
                for sd in SEEDS:
                    base = BasePolicy(n_joints=6, family=fam, seed=sd)
                    _, mp = cdiag.run_metrics(base, sd, noise, (lambda: make_v17(**c)))
                    key = f"{scen}/{fam}/{sd}"
                    d = {k: mp[k]-data[key]["base"][k] for k in agg}
                    for k in agg: agg[k].append(d[k])
        out["cfgs"][nm] = {"v17": {k: float(np.sum(agg[k])) for k in agg}}
    s14 = {k: 0.0 for k in ("C", "L", "R", "CI")}
    for key in data:
        for k in s14: s14[k] += data[key]["v14"][k]
    print(f"[V14] ΣΔC{s14['C']:+.3f} ΣΔL{s14['L']:+.3f} ΣΔR{s14['R']:+.3f} ΣΔCI{s14['CI']:+.3f}", flush=True)
    for nm in V17:
        g = out["cfgs"][nm]["v17"]
        print(f"[{nm}] ΣΔC{g['C']:+.3f} ΣΔL{g['L']:+.3f} ΣΔR{g['R']:+.3f} ΣΔCI{g['CI']:+.3f}", flush=True)
    print("\n分场景 V14 均Δ(基线对照)", flush=True)
    for scen in SCENS:
        for fam in FAMS:
            ds = [data[f"{scen}/{fam}/{sd}"]["v14"] for sd in SEEDS]
            print(f"[{scen}/{fam:11s}] V14均ΔC{np.mean([c['C'] for c in ds]):+.3f} ΔCI{np.mean([c['CI'] for c in ds]):+.3f}", flush=True)
    json.dump({"out": out}, open("v17.json", "w"), ensure_ascii=False, indent=1, default=float)

if __name__ == "__main__":
    main()