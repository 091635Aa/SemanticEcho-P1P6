#!/usr/bin/env python3
"""v12k.py — 多seed测量 transformer 族 聚合rr(逐关节rr均值)分布。
目标: 找"clean/trans 停靠(standdown)"与"mild/strong-trans 继续去抖"的聚合rr阈值。
      若 clean 聚合rr 稳定 < thr 且 mild/strong 稳定 > thr → 插件级硬门可用。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy, LeggedMicroSim
from plugins_robust import PhaseRecomb
import cdiag

T = 5000
DT = cdiag.DT
SEEDS = [99, 7, 42, 123, 5]


def run_agg(base, noise, seed):
    plug = PhaseRecomb(lam=0.85, alpha=0.05, blend=True,
                       r2b_lo=0.45, r2b_hi=0.85, min_cyc=2)
    s = LeggedMicroSim(base, plugins=[plug], T=T, dt=DT, seed=seed, **noise)
    s.run()
    return float(np.mean(plug._rr_ema)), float(np.mean(plug._r2_ema))


def main():
    print("聚合rr /聚合r2 跨seed (transformer):", flush=True)
    header = "  scen\\seed" + "".join(f"{s:7d}" for s in SEEDS)
    print(header, flush=True)
    agg = {}
    for scen in ["clean", "mild", "strong"]:
        noise = cdiag.SCEN[scen]
        agg[scen] = []
        row = f"  {scen:8s}"
        for seed in SEEDS:
            base = BasePolicy(n_joints=6, family="transformer", seed=seed)
            rr, r2 = run_agg(base, noise, seed)
            agg[scen].append(rr)
            row += f"{rr:7.3f}"
        print(row, flush=True)
    # 也可看 standard 中间值作参照
    print("参考 standard(应始终>阈值):", flush=True)
    for scen in ["clean", "mild"]:
        noise = cdiag.SCEN[scen]
        row = f"  {scen:8s}"
        for seed in SEEDS:
            base = BasePolicy(n_joints=6, family="standard", seed=seed)
            rr, r2 = run_agg(base, noise, seed)
            row += f"{rr:7.3f}"
        print(row, flush=True)
    json.dump({"agg": agg}, open("v12k_agg.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    import json
    main()