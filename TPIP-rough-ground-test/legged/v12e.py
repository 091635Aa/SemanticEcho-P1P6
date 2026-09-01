#!/usr/bin/env python3
"""v12e.py — 统一R²混合去抖(V12)跨 3族×3场景，检验无净损失(保C/L/R) + C/CI提升。
目标：Lam=0.85, blend=True(r2b 区内:高R²→1Hz骨干,低R²→模板), 扫 min_cyc 避开瞬态。"""
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


def main():
    print(f"V12 统一R²混合去抖 全扫 [T={T}, seed={SEED}]", flush=True)
    cfgs = {
        "base":   (lambda: None),
        "u-m2":   (lambda: PhaseRecomb(lam=0.85, alpha=0.05, blend=True,
                                       r2b_lo=0.45, r2b_hi=0.85, min_cyc=2)),
        "u-m8":   (lambda: PhaseRecomb(lam=0.85, alpha=0.05, blend=True,
                                       r2b_lo=0.45, r2b_hi=0.85, min_cyc=8)),
        "u-m6l7": (lambda: PhaseRecomb(lam=0.7, alpha=0.05, blend=True,
                                       r2b_lo=0.45, r2b_hi=0.85, min_cyc=6)),
        "u-m10":  (lambda: PhaseRecomb(lam=0.85, alpha=0.05, blend=True,
                                       r2b_lo=0.45, r2b_hi=0.85, min_cyc=10)),
    }
    order = ["u-m2", "u-m8", "u-m6l7", "u-m10"]
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
                line += f"  {cfg:6s} dC{dC:+.3f} dL{dL:+.3f} dR{dR:+.3f} dCI{dCI:+.3f}"
            print(line, flush=True)
    json.dump({"T": T, "seed": SEED},
              open(f"v12e_s{SEED}_T{T}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()