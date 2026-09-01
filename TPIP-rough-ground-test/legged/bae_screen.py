#!/usr/bin/env python3
"""bae_screen.py — V7(BlueprintAnchor) λ 扫描：测三维度+CI，检验无净损失。
用法: python3 bae_screen.py [T] [seed]"""
import sys, os, json, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cdiag  # 复用 T/SCEN/FAMS/train_uli/run_metrics 等模块级
from legged_env import BasePolicy
from verify_architecture import BestCombo
from plugins_robust import RobustComboV6, RobustComboV7

T = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 99
cdiag.T = T

SCEN = cdiag.SCEN
FAMS = cdiag.FAMS


def main():
    print(f"BAE λ 扫描 [T={T}, seed={SEED}]", flush=True)
    p0 = cdiag.train_uli()
    cfgs = {
        "v6g3s5":       (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=3.0, thr_hf=0.06, p=1.0, sh_cut=0.5)),
        "v7-lam03":     (lambda: RobustComboV7(BestCombo(), copy.deepcopy(p0), bae_lam=0.3, gammax=3.0, thr_hf=0.06, p=1.0, sh_cut=0.5)),
        "v7-lam05":     (lambda: RobustComboV7(BestCombo(), copy.deepcopy(p0), bae_lam=0.5, gammax=3.0, thr_hf=0.06, p=1.0, sh_cut=0.5)),
        "v7-lam07":     (lambda: RobustComboV7(BestCombo(), copy.deepcopy(p0), bae_lam=0.7, gammax=3.0, thr_hf=0.06, p=1.0, sh_cut=0.5)),
    }
    keys = ["C", "L", "R", "CI"]
    out = {}
    for scen in SCEN:
        out[scen] = {}
        for fam in FAMS:
            base = BasePolicy(n_joints=6, family=fam, seed=SEED)
            r0 = {}
            for name, make in cfgs.items():
                mb, mp = cdiag.run_metrics(base, SEED, SCEN[scen], make)
                r0[name] = {k: round(float(mp[k] - mb[k]), 4) for k in keys}
            out[scen][fam] = r0
            line = f"[{scen}/{fam:11s}]"
            for name in cfgs:
                line += (f"  {name:8s} dC={r0[name]['C']:+.3f} dL={r0[name]['L']:+.3f} "
                         f"dR={r0[name]['R']:+.3f} dCI={r0[name]['CI']:+.3f}")
            print(line, flush=True)
    json.dump({"T": T, "seed": SEED, "out": out}, open(f"bae_screen_s{SEED}_T{T}.json", "w"),
              ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()