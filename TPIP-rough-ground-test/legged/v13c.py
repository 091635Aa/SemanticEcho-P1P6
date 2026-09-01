#!/usr/bin/env python3
"""v13c.py — g8lam1 vs g5 跨seed全矩阵最终对比, 看 L 增益是否值其 C 代价.
输出每细胞 ΔC/ΔL/ΔR/ΔCI, 及 ΣΔC ΣΔL ΣΔR ΣΔCI / 负C细胞数. 选被采纳配置。"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy
from plugins_robust import PhaseRecomb
import cdiag

T = 6000
cdiag.T = T
SEEDS = [99, 7, 42]
FAMS = ["standard", "transformer", "p2p"]


def mk(gbl, lam):
    return lambda: PhaseRecomb(lam=lam, alpha=0.05, blend=True,
                               r2b_lo=0.45, r2b_hi=0.85, min_cyc=2, gbl=gbl)


def main():
    CONFIGS = {"g5.85": (0.5, 0.85), "g8.1": (0.8, 1.0)}
    acc = {n: {"cells": {}, "sum": {k: 0.0 for k in ("C", "L", "R", "CI")}, "nneg": 0}
           for n in CONFIGS}
    for scen in ["clean", "mild", "strong"]:
        noise = cdiag.SCEN[scen]
        for fam in FAMS:
            mbb = {}
            cells = {n: {k: [] for k in ("C", "L", "R", "CI")} for n in CONFIGS}
            for seed in SEEDS:
                base = BasePolicy(n_joints=6, family=fam, seed=seed)
                mb, _ = cdiag.run_metrics(base, seed, noise, (lambda: None))
                mbb = mb
                for n, (g, lam) in CONFIGS.items():
                    _, mp = cdiag.run_metrics(base, seed, noise, mk(g, lam))
                    for k in cells[n]:
                        cells[n][k].append(mp[k] - mb[k])
            key = f"{scen}/{fam}"
            line = f"[{key:16s}] base C{mbb['C']:.3f} L{mbb['L']:.3f} R{mbb['R']:.3f}"
            for n in CONFIGS:
                m = {k: float(np.mean(cells[n][k])) for k in cells[n]}
                acc[n]["cells"][key] = m
                line += f"  {n:6s} ΔC{m['C']:+.3f} ΔL{m['L']:+.3f} ΔR{m['R']:+.3f} ΔCI{m['CI']:+.3f}"
                for k in acc[n]["sum"]:
                    acc[n]["sum"][k] += m[k]
                if m["C"] < 0:
                    acc[n]["nneg"] += 1
            print(line, flush=True)
    print("\n采纳判定:", flush=True)
    for n in CONFIGS:
        s = acc[n]["sum"]
        print(f"  {n:6s}: ΣΔC{s['C']:+.3f} ΣΔL{s['L']:+.3f} ΣΔR{s['R']:+.3f} ΣΔCI{s['CI']:+.3f} | C负细胞数={acc[n]['nneg']}", flush=True)
    json.dump({"T": T, "SEEDS": SEEDS, "acc": acc},
              open("v13c.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()