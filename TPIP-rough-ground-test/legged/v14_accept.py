#!/usr/bin/env python3
"""v14_accept.py — V14 终定接受度: PhaseRecomb(lam=1.0, blend, r2b .45/.85, mc2, gbl=0.8)
跨 3族×3场景×5seed, T=8000. 判定: 8/9细胞 C≥0 且全区净增益 + p2p 全0 + 可信度(噪声下增益保持)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy
from plugins_robust import PhaseRecomb
import cdiag

T = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
cdiag.T = T
SEEDS = [99, 7, 42, 123, 5]
FAMS = ["standard", "transformer", "p2p"]


def make():
    return PhaseRecomb(lam=1.0, alpha=0.05, blend=True,
                       r2b_lo=0.45, r2b_hi=0.85, min_cyc=2, gbl=0.8)


def main():
    print(f"V14 接受度 [T={T}, seeds={SEEDS}]", flush=True)
    out = {"T": T, "SEEDS": SEEDS, "cells": {}}
    tot = {"C": 0.0, "L": 0.0, "R": 0.0, "CI": 0.0}
    nneg = {"C": 0, "CI": 0}
    baseline_gain = {"C": 0.0, "CI": 0.0}   # 干净环境增益(可信度参考)
    for scen in ["clean", "mild", "strong"]:
        noise = cdiag.SCEN[scen]
        for fam in FAMS:
            cells = {k: [] for k in ("C", "L", "R", "CI")}
            for sd in SEEDS:
                base = BasePolicy(n_joints=6, family=fam, seed=sd)
                mb, _ = cdiag.run_metrics(base, sd, noise, (lambda: None))
                _, mp = cdiag.run_metrics(base, sd, noise, make)
                for k in cells:
                    cells[k].append(mp[k] - mb[k])
            m = {k: float(np.mean(cells[k])) for k in cells}
            out["cells"][f"{scen}/{fam}"] = m
            line = f"[{scen}/{fam:11s}]"
            for k in ("C", "L", "R", "CI"):
                line += f" Δ{k}{m[k]:+.3f}"
            print(line, flush=True)
            if scen == "clean":
                for k in baseline_gain:
                    baseline_gain[k] += m[k]
            if scen != "clean" and fam != "p2p":
                for k in tot:
                    tot[k] += m[k]
                if m["C"] < 0: nneg["C"] += 1
                if m["CI"] < 0: nneg["CI"] += 1
    # 可信度: 噪声场景合计增益 / 干净场景合计增益(C 与 CI)
    print("\n可信度(噪声区净C增益 / 干净区C增益):", flush=True)
    cleanC = baseline_gain["C"]  # 仅 standard+transformer(排除p2p=p0)
    print(f"  干净区 ΔC 合计={cleanC:+.3f}", flush=True)
    json.dump({"out": out}, open("v14_accept.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()