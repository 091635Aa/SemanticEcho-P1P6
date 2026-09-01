#!/usr/bin/env python3
"""brl_validate.py — V9(BRL) 跨三族×三场景×多seed 最终验证。
对比 baseline / v6g3s5(旧冠军,CI高但R崩) / v9-l3g0 / v9-l4g2。
输出绝对三维度(C,L,R)+CI，评估"在保R/不损p2p的前提下是否C/L/CI同升"。
用法: python3 brl_validate.py [T] [seed] [seed] ...
"""
import sys, os, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from learnable_bypass import ULI
from verify_architecture import BestCombo
from plugins_robust import RobustComboV6, RobustComboV9, BlueprintRefLock, RobustComboV11
import cdiag

T = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
SEEDS = [int(x) for x in sys.argv[2:]] or [42, 7, 99]
FAMS = ["standard", "transformer", "p2p"]


def train_uli():
    plug = ULI(strength=1.0, W=40, lam=5e-3)
    data = plug.collect_trajectory(BasePolicy(n_joints=6, family="standard", seed=42), seed=42, t_len=60000)
    plug.fit_from(data); plug.strength = 0.5; plug._amp = min(plug._amp, 0.5)
    return plug


CFGS = {
    "base":    (lambda p0: None),
    "v6g3s5":  (lambda p0: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=3.0, thr_hf=0.06, p=1.0, sh_cut=0.5)),
    "v9-l3g0": (lambda p0: RobustComboV9(copy.deepcopy(p0), lam=0.3, rrcut=0.35)),
    "v9-l4g2": (lambda p0: RobustComboV9(copy.deepcopy(p0), lam=0.4, rrcut=0.35, gammax=2.0, thr_hf=0.06)),
}


def main():
    print(f"V9 验证 [T={T}, seeds={SEEDS}]", flush=True)
    p0 = train_uli()
    agg = {cfg: {sc: {f: {m: [] for m in ["C","L","R","CI"]} for f in FAMS} for sc in ["clean","mild","strong"]} for cfg in CFGS}
    for seed in SEEDS:
        for scen in ["clean", "mild", "strong"]:
            noise = cdiag.SCEN[scen]
            for fam in FAMS:
                base = BasePolicy(n_joints=6, family=fam, seed=seed)
                for cfg, make in CFGS.items():
                    mb, mp = cdiag.run_metrics(base, seed, noise, lambda c=cfg: make(p0))
                    if cfg == "base":
                        for m in ["C","L","R","CI"]:
                            agg["base"][scen][fam][m].append(mb[m])
                    else:
                        for m in ["C","L","R","CI"]:
                            agg[cfg][scen][fam][m].append(mp[m])
    print("\n==== 绝对均值(多seed) ====", flush=True)
    for scen in ["clean", "mild", "strong"]:
        print(f"\n--- {scen} ---", flush=True)
        for fam in FAMS:
            line = f"[{scen}/{fam:11s}]"
            for cfg in ["base","v6g3s5","v9-l3g0","v9-l4g2"]:
                v = {m: float(np.mean(agg[cfg][scen][fam][m])) for m in ["C","L","R","CI"]}
                line += (f"  | {cfg:7s} C{v['C']:+.3f} L{v['L']:+.3f} R{v['R']:+.3f} CI{v['CI']:+.3f}")
            print(line, flush=True)
    print("\n==== 相对基线增量(多seed均值) ====", flush=True)
    for scen in ["clean", "mild", "strong"]:
        print(f"\n--- {scen} ---", flush=True)
        for fam in FAMS:
            line = f"[{scen}/{fam:11s}]"
            b = {m: float(np.mean(agg["base"][scen][fam][m])) for m in ["C","L","R","CI"]}
            for cfg in ["v6g3s5","v9-l3g0","v9-l4g2"]:
                v = {m: float(np.mean(agg[cfg][scen][fam][m])) for m in ["C","L","R","CI"]}
                line += (f"  | {cfg:7s} dC={v['C']-b['C']:+.3f} dL={v['L']-b['L']:+.3f} "
                         f"dR={v['R']-b['R']:+.3f} dCI={v['CI']-b['CI']:+.3f}")
            print(line, flush=True)


import json
if __name__ == "__main__":
    main()