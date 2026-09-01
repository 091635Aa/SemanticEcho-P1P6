#!/usr/bin/env python3
"""v18.py — 阶段1b: 修复 clean/transformer C/CI 净损失(短视界转负).
假设: 对"已自洽"(低rr)关节 stand-down 去抖, 护饱和 transformer 但保住 standard/noise 增益.
扫 rr_stand on/off + rr_lo/rr_hi. 输出分 scen/fam 明细(含 clean/transformer 是否回正)."""
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

BASE = dict(lam=1.0, alpha=0.05, blend=True, r2b_lo=0.45, r2b_hi=0.85, min_cyc=2, gbl=0.8)

def make_extras(**k):
    return PhaseRecomb(**{**BASE, **k})

CFGS = {
    "v14(stand off)": dict(),
    "st.02/.04":      dict(rr_stand=True, rr_lo=0.020, rr_hi=0.040),
    "st.03/.05":      dict(rr_stand=True, rr_lo=0.030, rr_hi=0.050),
    "st.03/.06":      dict(rr_stand=True, rr_lo=0.030, rr_hi=0.060),
    "st.04/.07":      dict(rr_stand=True, rr_lo=0.040, rr_hi=0.070),
}

def main():
    print(f"V18 rr_stand 扫 [T={T}]", flush=True)
    out = {"T": T, "cfgs": {}}
    base_d = {}
    for scen in SCENS:
        noise = cdiag.SCEN[scen]
        for fam in FAMS:
            for sd in SEEDS:
                base = BasePolicy(n_joints=6, family=fam, seed=sd)
                mb, _ = cdiag.run_metrics(base, sd, noise, (lambda: None))
                base_d[f"{scen}/{fam}/{sd}"] = mb
    for nm, kws in CFGS.items():
        cells = {}
        for scen in SCENS:
            noise = cdiag.SCEN[scen]
            for fam in FAMS:
                ag = {k: [] for k in ("C", "L", "R", "CI")}
                for sd in SEEDS:
                    base = BasePolicy(n_joints=6, family=fam, seed=sd)
                    _, mp = cdiag.run_metrics(base, sd, noise, (lambda: make_extras(**kws)))
                    for k in ag:
                        ag[k].append(mp[k] - base_d[f"{scen}/{fam}/{sd}"][k])
                cells[f"{scen}/{fam}"] = {k: float(np.mean(ag[k])) for k in ag}
        out["cfgs"][nm] = cells
        line = f"[{nm:15s}]"
        for scen in SCENS:
            for fam in FAMS:
                c = cells[f"{scen}/{fam}"]
                line += f" {scen[0]}/{fam[:4]}:ΔC{c['C']:+.3f}"
        negs = [f"{scen}/{fam}" for scen in SCENS for fam in FAMS
                if cells[f"{scen}/{fam}"]["C"] < 0]
        line += " | 净损失(ΔC<0):" + str(negs)
        print(line, flush=True)
    json.dump({"out": out}, open("v18.json", "w"), ensure_ascii=False, indent=1, default=float)

if __name__ == "__main__":
    main()