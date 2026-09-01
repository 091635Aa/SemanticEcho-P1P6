#!/usr/bin/env python3
"""v12_screen.py — PhaseRecomb 目标去抖: 全谐波模板(tpl) vs R²-混合(1Hz骨干/tpl)。
对照 v11(blend=False)验证 R²-blend 是否修复 transformer 的 C/R 净损失，同时保住
standard 的 C/CI 增益 与 p2p 保护(Universal)。
用法: python3 v12_screen.py [T] [seed]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy
from plugins_robust import PhaseRecomb
import cdiag

T = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 99
FAMS = ["standard", "transformer", "p2p"]


def make_screen():
    cfgs = {
        "base":      (lambda: None),
        "v11-tplbl": (lambda: PhaseRecomb(lam=0.5, alpha=0.05, blend=False)),
        "v12-bl0.5": (lambda: PhaseRecomb(lam=0.5, alpha=0.05, blend=True, r2b_lo=0.35, r2b_hi=0.75)),
        "v12-bl0.7": (lambda: PhaseRecomb(lam=0.7, alpha=0.05, blend=True, r2b_lo=0.35, r2b_hi=0.75)),
        "v12-bl0.7w":(lambda: PhaseRecomb(lam=0.7, alpha=0.05, blend=True, r2b_lo=0.5, r2b_hi=0.85)),
        "v12-bl0.5b":(lambda: PhaseRecomb(lam=0.5, alpha=0.05, blend=True, r2b_lo=0.2, r2b_hi=0.65)),
    }
    return cfgs


def main():
    print(f"V12(R²-blend 去抖) 扫描 [T={T}, seed={SEED}]", flush=True)
    cfgs = make_screen()
    order = ["base", "v11-tplbl", "v12-bl0.5", "v12-bl0.7", "v12-bl0.7w", "v12-bl0.5b"]
    out = {}
    for scen in ["clean", "mild", "strong"]:
        noise = cdiag.SCEN[scen]
        out[scen] = {}
        print(f"--- {scen} ---", flush=True)
        for fam in FAMS:
            base = BasePolicy(n_joints=6, family=fam, seed=SEED)
            row = {}
            for name, make in cfgs.items():
                mb, mp = cdiag.run_metrics(base, SEED, noise, make)
                row[name] = {"base": mb, "plug": mp}
            out[scen][fam] = row
            line = f"[{scen}/{fam:11s}]"
            mb = row["base"]["base"]
            line += f" base C{mb['C']:.3f} L{mb['L']:.3f} R{mb['R']:.3f} CI{mb['CI']:.3f}"
            for cfg in ["v11-tplbl", "v12-bl0.5", "v12-bl0.7", "v12-bl0.7w", "v12-bl0.5b"]:
                p = row[cfg]["plug"]
                dC = p['C'] - mb['C']; dL = p['L'] - mb['L']
                dR = p['R'] - mb['R']; dCI = p['CI'] - mb['CI']
                line += (f"  | {cfg:9s} dC{dC:+.3f} dL{dL:+.3f} dR{dR:+.3f} dCI{dCI:+.3f}")
            print(line, flush=True)
    import json
    json.dump({"T": T, "seed": SEED, "results": out},
              open(f"v12_s{SEED}_T{T}.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()