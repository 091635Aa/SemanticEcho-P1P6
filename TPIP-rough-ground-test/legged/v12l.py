#!/usr/bin/env python3
"""v12l.py — 终定 V12(u-m2) 跨 3族×3场景×3seed 鲁棒性验证 + 可信度。
输出: 每个细胞 ΔC/ΔL/ΔR/ΔCI 及"净增益"(C+CI 相对小量R代价), 判断是否满足
      "无净损失 或 损失与增益成正比"。"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy
from plugins_robust import PhaseRecomb
import cdiag

T = 6000
DT = cdiag.DT
cdiag.T = T


def make_v12():
    return PhaseRecomb(lam=0.85, alpha=0.05, blend=True,
                       r2b_lo=0.45, r2b_hi=0.85, min_cyc=2)


def main():
    SEEDS = [99, 7, 42]
    FAMS = ["standard", "transformer", "p2p"]
    print(f"V12终定(lam.85 blend r2b.45/.85 mc2) 跨 3族×3场景×{len(SEEDS)}seed [T={T}]", flush=True)
    allout = {}
    for scen in ["clean", "mild", "strong"]:
        noise = cdiag.SCEN[scen]
        allout[scen] = {}
        for fam in FAMS:
            delta_accum = {"C": [], "L": [], "R": [], "CI": []}
            for seed in SEEDS:
                base = BasePolicy(n_joints=6, family=fam, seed=seed)
                mb, mp = cdiag.run_metrics(base, seed, noise, make_v12)
                for k in delta_accum:
                    delta_accum[k].append(mp[k] - mb[k])
            row = {k: float(np.mean(v)) for k, v in delta_accum.items()}
            allout[scen][fam] = row
            line = (f"[{scen}/{fam:11s}]  ΔC{row['C']:+.3f}  ΔL{row['L']:+.3f} "
                    f"ΔR{row['R']:+.3f}  ΔCI{row['CI']:+.3f}")
            print(line, flush=True)
    # 汇总
    print("\n汇总(全9细胞×3seed均值):", flush=True)
    totC = sum(allout[s][f]["C"] for s in allout for f in FAMS)
    totCI = sum(allout[s][f]["CI"] for s in allout for f in FAMS)
    totR = sum(allout[s][f]["R"] for s in allout for f in FAMS)
    totL = sum(allout[s][f]["L"] for s in allout for f in FAMS)
    print(f"  ΣΔC={totC:+.3f}  ΣΔL={totL:+.3f}  ΣΔR={totR:+.3f}  ΣΔCI={totCI:+.3f}", flush=True)
    json.dump({"T": T, "SEEDS": SEEDS, "del": allout},
              open("v12l_v12.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()