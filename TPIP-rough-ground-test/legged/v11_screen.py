#!/usr/bin/env python3
"""v11_screen.py — PhaseRecomb(V11) 参数扫描：lam(去抖强度)×alpha(模板遗忘)。
目标：无净损失(C/L/R 全不降)前提下 C 一致性 / CI 连贯性 抬升。对照 V9-l4g2(旧)。
用法: python3 v11_screen.py [T] [seed]"""
import sys, os, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy
from learnable_bypass import ULI
from plugins_robust import RobustComboV9, RobustComboV11, PhaseRecomb
import cdiag

T = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 99
FAMS = ["standard", "transformer", "p2p"]


def train_uli():
    plug = ULI(strength=1.0, W=40, lam=5e-3)
    data = plug.collect_trajectory(BasePolicy(n_joints=6, family="standard", seed=42), seed=42, t_len=60000)
    plug.fit_from(data); plug.strength = 0.5; plug._amp = min(plug._amp, 0.5)
    return plug


def main():
    print(f"V11(PhaseRecomb) 扫描 [T={T}, seed={SEED}]", flush=True)
    p0 = train_uli()
    cfgs = {
        "base":    (lambda: None),
        "v9-l4g2": (lambda: RobustComboV9(copy.deepcopy(p0), lam=0.4, rrcut=0.35, gammax=2.0, thr_hf=0.06)),
        "v11-l5":  (lambda: RobustComboV11(copy.deepcopy(p0), lam=0.5, alpha=0.05)),
        "v11-l7":  (lambda: RobustComboV11(copy.deepcopy(p0), lam=0.7, alpha=0.05)),
        "v11-ng":  (lambda: RobustComboV11(copy.deepcopy(p0), lam=0.7, alpha=0.05, r2_floor=0.45, r2_ceil=0.7)),
        "v11-noU": (lambda: PhaseRecomb(lam=0.7, alpha=0.05)),
        "v11-ada": (lambda: PhaseRecomb(lam=0.7, alpha=0.05, adapt=True, lam_ref=0.45, lam_min=0.03)),
        "v11-ada2":(lambda: PhaseRecomb(lam=0.9, alpha=0.05, adapt=True, lam_ref=0.30, lam_min=0.03)),
    }
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
            for cfg in ["base","v11-noU","v11-ada","v11-ada2","v11-l7"]:
                p = row[cfg]["plug"]
                line += (f"  | {cfg:7s} C{p['C']:+.3f} L{p['L']:+.3f} R{p['R']:+.3f} CI{p['CI']:+.3f}")
            print(line, flush=True)
    import json
    json.dump({"T": T, "seed": SEED, "results": out},
              open(f"v11_s{SEED}_T{T}.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()