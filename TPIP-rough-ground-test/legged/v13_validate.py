#!/usr/bin/env python3
"""v13_validate.py — GBL终定跨seed验证: 3族×3场景×3seed, 扫 gbl∈{0.3,0.5,0.7,1.0}.
目标: 找到所有 9 细胞 dC/dR≥0 且 CI 尽量高的鲁棒配置(实现全矩阵无净损失)."""
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


def mk(gbl):
    return lambda: PhaseRecomb(lam=0.85, alpha=0.05, blend=True,
                               r2b_lo=0.45, r2b_hi=0.85, min_cyc=2, gbl=gbl)


def main():
    GL = [0.3, 0.5, 0.7, 1.0]
    print(f"GBL跨seed验证 [T={T}, seeds={SEEDS}]", flush=True)
    acc = {g: {} for g in GL}
    for scen in ["clean", "mild", "strong"]:
        noise = cdiag.SCEN[scen]
        for fam in FAMS:
            cells = {g: {k: [] for k in ("C", "L", "R", "CI")} for g in GL}
            for seed in SEEDS:
                base = BasePolicy(n_joints=6, family=fam, seed=seed)
                mb, _ = cdiag.run_metrics(base, seed, noise, (lambda: None))
                for g in GL:
                    _, mp = cdiag.run_metrics(base, seed, noise, mk(g))
                    for k in cells[g]:
                        cells[g][k].append(mp[k] - mb[k])
            line = f"[{scen}/{fam:11s}]  base C{mb['C']:.3f} L{mb['L']:.3f} R{mb['R']:.3f} CI{mb['CI']:.3f}"
            for g in GL:
                m = {k: float(np.mean(cells[g][k])) for k in cells[g]}
                acc[g][f"{scen}/{fam}"] = m
                line += f"  g{g:g} dC{m['C']:+.3f} dR{m['R']:+.3f} dCI{m['CI']:+.3f}"
            print(line, flush=True)
    print("\n各gbl下的9细胞总增: ΣdC / ΣdCI / 负C细胞数:", flush=True)
    for g in GL:
        totC = sum(acc[g][c]["C"] for c in acc[g])
        totCI = sum(acc[g][c]["CI"] for c in acc[g])
        totR = sum(acc[g][c]["R"] for c in acc[g])
        nneg = sum(1 for c in acc[g] if acc[g][c]["C"] < 0)
        print(f"  gbl={g:g}: ΣΔC{totC:+.3f}  ΣΔR{totR:+.3f}  ΣΔCI{totCI:+.3f}  C负细胞数{nneg}", flush=True)
    json.dump({"T": T, "SEEDS": SEEDS, "acc": acc},
              open("v13_val.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()