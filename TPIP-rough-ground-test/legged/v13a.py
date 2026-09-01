#!/usr/bin/env python3
"""v13a.py — GBL(蓝图相位联动)强度扫描: 看是否抬 L联动 / R参照 且不掉 C/CI。
用法: python3 v13a.py [T] [seed]"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy
from plugins_robust import PhaseRecomb
import cdiag

T = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 99
cdiag.T = T
FAMS = ["standard", "transformer", "p2p"]


def mk(gbl):
    return lambda: PhaseRecomb(lam=0.85, alpha=0.05, blend=True,
                               r2b_lo=0.45, r2b_hi=0.85, min_cyc=2, gbl=gbl)


def main():
    print(f"GBL强度扫描 [T={T}, seed={SEED}]", flush=True)
    cfgs = {"base": (lambda: None)}
    order = []
    for g in [0.0, 0.15, 0.3, 0.5]:
        name = f"g{g:g}"
        cfgs[name] = mk(g); order.append(name)
    for scen in ["clean", "mild", "strong"]:
        noise = cdiag.SCEN[scen]
        for fam in FAMS:
            base = BasePolicy(n_joints=6, family=fam, seed=SEED)
            row = {}
            for name, make in cfgs.items():
                mb, mp = cdiag.run_metrics(base, SEED, noise, make)
                row[name] = {"base": mb, "plug": mp}
            line = f"[{scen}/{fam:11s}]"
            mb = row["base"]["base"]
            line += f" base C{mb['C']:.3f} L{mb['L']:.3f} R{mb['R']:.3f} CI{mb['CI']:.3f}"
            for cfg in order:
                p = row[cfg]["plug"]
                dC = p['C']-mb['C']; dL = p['L']-mb['L']; dR = p['R']-mb['R']; dCI = p['CI']-mb['CI']
                line += f"  {cfg:3s} dC{dC:+.2f} dL{dL:+.2f} dR{dR:+.2f} dCI{dCI:+.2f}"
            print(line, flush=True)
    json.dump({"T": T, "seed": SEED}, open(f"v13a_s{SEED}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()