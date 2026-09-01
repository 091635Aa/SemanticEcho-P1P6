#!/usr/bin/env python3
"""brl2_screen.py — BRL-SNR 参数扫描：蓝图锁定强度 lam × 噪声放大 gammax。
目标：R_ref 保住(≈基线0.93,/不要掉到0.3) 的同时，噪声下 CI 有增益，p2p 无损。
用法: python3 brl2_screen.py [T] [seed]"""
import sys, os, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from learnable_bypass import ULI
from plugins_robust import RobustComboV9
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
    print(f"BRL-SNR 扫描 [T={T}, seed={SEED}]", flush=True)
    p0 = train_uli()
    cfgs = {
        "base":      (lambda: None),
        "v9-l3g0":   (lambda: RobustComboV9(copy.deepcopy(p0), lam=0.3, rrcut=0.35)),
        "v9-l4g0":   (lambda: RobustComboV9(copy.deepcopy(p0), lam=0.4, rrcut=0.35)),
        "v9-l4g2":   (lambda: RobustComboV9(copy.deepcopy(p0), lam=0.4, rrcut=0.35, gammax=2.0, thr_hf=0.06)),
        "v9-l4g4":   (lambda: RobustComboV9(copy.deepcopy(p0), lam=0.4, rrcut=0.35, gammax=4.0, thr_hf=0.06)),
        "v9-l5g2":   (lambda: RobustComboV9(copy.deepcopy(p0), lam=0.5, rrcut=0.35, gammax=2.0, thr_hf=0.06)),
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
            for cfg in ["base","v9-l3g0","v9-l4g0","v9-l4g2","v9-l4g4","v9-l5g2"]:
                p = row[cfg]["plug"]
                line += (f"  | {cfg:7s} C{p['C']:+.3f} L{p['L']:+.3f} R{p['R']:+.3f} CI{p['CI']:+.3f}")
            print(line, flush=True)
    json.dump({"T": T, "seed": SEED, "results": out},
              open(f"brl2_s{SEED}_T{T}.json", "w"), ensure_ascii=False, indent=1, default=float)


import json
if __name__ == "__main__":
    main()