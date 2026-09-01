#!/usr/bin/env python3
"""v12h.py — 低rr stand-down 门(护 clean/transformer) 跨 3族×3场景验证。
目标：让 clean/transformer 从 C净损(-0.049) 回到 0/正，同时标准族 C 增益(NETE) 不降。
用法: python3 v12h.py [T] [seed]"""
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


def mk(rr_lo, rr_hi, lam=0.85):
    return lambda: PhaseRecomb(lam=lam, alpha=0.05, blend=True,
                               r2b_lo=0.45, r2b_hi=0.85, min_cyc=2,
                               rr_stand=True, rr_lo=rr_lo, rr_hi=rr_hi)


def main():
    print(f"V12h rr-stand 全扫 [T={T}, seed={SEED}]", flush=True)
    cfgs = {
        "base": (lambda: None),
        "u-m2": (lambda: PhaseRecomb(lam=0.85, alpha=0.05, blend=True,
                                     r2b_lo=0.45, r2b_hi=0.85, min_cyc=2)),
        "sd25/40": mk(0.025, 0.040),
        "sd28/42": mk(0.028, 0.042),
        "sd30/45": mk(0.030, 0.045),
        "sd26/44": mk(0.026, 0.044),
    }
    order = ["sd25/40", "sd28/42", "sd30/45", "sd26/44"]
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
                dC = p['C'] - mb['C']; dL = p['L'] - mb['L']
                dR = p['R'] - mb['R']; dCI = p['CI'] - mb['CI']
                line += f"  {cfg:7s} dC{dC:+.3f} dR{dR:+.3f} dCI{dCI:+.3f}"
            print(line, flush=True)
    json.dump({"T": T, "seed": SEED},
              open(f"v12h_s{SEED}_T{T}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()