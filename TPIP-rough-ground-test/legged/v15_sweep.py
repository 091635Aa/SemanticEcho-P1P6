#!/usr/bin/env python3
"""v15_sweep.py — V15 联动/参照增强扫描: 扫 GBL 强度 (gbl & gbl 门控变体)
用户目标: 在"无损失 或 代价与增益成正比"前提下, 推高 L联动 / R参照 与 C一致性/CI连贯。
V14 (gbl=0.8) 已全正向但 L~+0.01 / R~+0.02(近饱和)。探测是否还有 L/R 空间。
判定: 每个 cfg 看 (ΣΔL 提高, ΣΔR 提高) 是否带来 ΔC/ΔCI 代价, 以及是否仍无净损失。
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy
from plugins_robust import PhaseRecomb
import cdiag

T = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
cdiag.T = T
SEEDS = [99, 7]
FAMS = ["standard", "transformer"]   # p2p 由 r2 门保护恒 0, 不逐一重测
SCENS = ["clean", "mild", "strong"]


def make(gbl, gbl_alpha=0.05, lam=1.0):
    return PhaseRecomb(lam=lam, alpha=0.05, blend=True,
                       r2b_lo=0.45, r2b_hi=0.85, min_cyc=2,
                       gbl=gbl, gbl_alpha=gbl_alpha)


CFGS = {
    "v14 (g0.8)": dict(gbl=0.8),
    "g1.2":       dict(gbl=1.2),
    "g1.6":       dict(gbl=1.6),
    "g2.0":       dict(gbl=2.0),
    "g1.2_a10":   dict(gbl=1.2, gbl_alpha=0.10),
}


def main():
    print(f"V15 GBL sweep [T={T}, seeds={SEEDS}, fams={FAMS}]", flush=True)
    out = {"T": T, "SEEDS": SEEDS, "scens": SCENS, "fams": FAMS, "cfgs": {}}
    for name, kw in CFGS.items():
        agg = {k: [] for k in ("C", "L", "R", "CI")}
        nneg = {"C": 0, "CI": 0, "L": 0, "R": 0}
        ncell = 0
        for scen in SCENS:
            noise = cdiag.SCEN[scen]
            for fam in FAMS:
                for sd in SEEDS:
                    base = BasePolicy(n_joints=6, family=fam, seed=sd)
                    mb, _ = cdiag.run_metrics(base, sd, noise, (lambda: None))
                    _, mp = cdiag.run_metrics(base, sd, noise, (lambda: make(**kw)))
                    for k in agg:
                        agg[k].append(mp[k] - mb[k])
                        if mp[k] - mb[k] < 0:
                            nneg[k] += 1
                    ncell += 1
        m = {k: float(np.mean(agg[k])) for k in agg}
        tot = {k: float(np.sum(agg[k])) for k in agg}
        out["cfgs"][name] = {"mean": m, "sum": tot, "nneg": nneg, "ncell": ncell}
        line = (f"[{name:9s}] ΣΔC{tot['C']:+.3f} ΣΔL{tot['L']:+.3f} ΣΔR{tot['R']:+.3f} "
                f"ΣΔCI{tot['CI']:+.3f} | 均值ΔC{m['C']:+.3f} ΔL{m['L']:+.3f} "
                f"ΔR{m['R']:+.3f} ΔCI{m['CI']:+.3f} | nneg(C/L/R/CI)={nneg['C']}/{nneg['L']}/{nneg['R']}/{nneg['CI']}")
        print(line, flush=True)
    json.dump({"out": out}, open("v15_sweep.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()