#!/usr/bin/env python3
"""v20.py — r2高值stand-down扫: 修 clean/transformer 净损失, 且不伤 standard/噪声增益.
同时探 r2_ema 实测值以校准阈值. 输出分 scen/fam 明细."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy
from plugins_robust import PhaseRecomb
import cdiag

T = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
cdiag.T = T
SEEDS = [99, 7, 42, 123, 5]
FAMS = ["standard", "transformer"]
SCENS = ["clean", "mild", "strong"]

BASE = dict(lam=1.0, alpha=0.05, blend=True, r2b_lo=0.45, r2b_hi=0.85, min_cyc=2, gbl=0.8)

def make(**k):
    return PhaseRecomb(**{**BASE, **k})

CFGS = {
    "v14 (off)  ": dict(),
    "rs .80/.92": dict(r2_stand=True, r2_stand_lo=0.80, r2_stand_hi=0.92),
    "rs .82/.94": dict(r2_stand=True, r2_stand_lo=0.82, r2_stand_hi=0.94),
    "rs .85/.95": dict(r2_stand=True, r2_stand_lo=0.85, r2_stand_hi=0.95),
    "rs .88/.96": dict(r2_stand=True, r2_stand_lo=0.88, r2_stand_hi=0.96),
}

def main():
    print(f"V20 r2_stand 扫 [T={T}, seed={len(SEEDS)}]", flush=True)
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
                    _, mp = cdiag.run_metrics(base, sd, noise, (lambda: make(**kws)))
                    for k in ag:
                        ag[k].append(mp[k] - base_d[f"{scen}/{fam}/{sd}"][k])
                cells[f"{scen}/{fam}"] = {k: float(np.mean(ag[k])) for k in ag}
        out["cfgs"][nm] = cells
        line = f"[{nm}]"
        for scen in SCENS:
            c0 = cells[f"{scen}/{FAMS[0]}"]; c1 = cells[f"{scen}/{FAMS[1]}"]
            line += (f" {scen[0]}/st:ΔC{c0['C']:+.3f}ΔCI{c0['CI']:+.3f} "
                     f"{scen[0]}/tr:ΔC{c1['C']:+.3f}ΔCI{c1['CI']:+.3f}")
        negs = [f"{s}/{f}" for s in SCENS for f in FAMS if cells[f"{s}/{f}"]["C"] < 0]
        line += " | ΔC<0:" + str(negs)
        print(line, flush=True)
    json.dump({"out": out}, open("v20.json", "w"), ensure_ascii=False, indent=1, default=float)

if __name__ == "__main__":
    main()