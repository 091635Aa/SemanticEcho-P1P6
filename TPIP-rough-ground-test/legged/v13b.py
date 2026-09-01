#!/usr/bin/env python3
"""v13b.py — 检查 gbl=0.5 的 L联动 跨seed净值 + 试探较强耦合(更高lam+更强GBL)对L的影响.
目标: 联动性 L 目前基本不动(base~0.57), 看是否能安全抬升。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy
from plugins_robust import PhaseRecomb
import cdiag

T = 6000
cdiag.T = T
SEEDS = [99, 7, 42]
FAMS = ["standard", "transformer"]


def mk(**kw):
    d = dict(alpha=0.05, blend=True, r2b_lo=0.45, r2b_hi=0.85, min_cyc=2)
    d.update(kw)
    return lambda: PhaseRecomb(**d)


def main():
    cfgs = {
        "g0":      mk(),
        "g5":      mk(gbl=0.5),
        "g5lam1":  mk(gbl=0.5, lam=1.0),
        "g8lam1":  mk(gbl=0.8, lam=1.0),
        "g12lam1": mk(gbl=1.2, lam=1.0),
    }
    order = ["g0", "g5", "g5lam1", "g8lam1", "g12lam1"]
    print(f"L联动探针 [T={T}]", flush=True)
    for scen in ["clean", "mild", "strong"]:
        noise = cdiag.SCEN[scen]
        for fam in FAMS:
            cells = {c: [] for c in order}
            mbb = None
            for seed in SEEDS:
                base = BasePolicy(n_joints=6, family=fam, seed=seed)
                mb, _ = cdiag.run_metrics(base, seed, noise, (lambda: None))
                mbb = mb
                for c in order:
                    _, mp = cdiag.run_metrics(base, seed, noise, cfgs[c])
                    cells[c].append(mp["L"] - mb["L"])
            line = f"[{scen}/{fam:11s}] baseL{mbb['L']:.3f}"
            for c in order:
                line += f"  {c:7s} ΔL{np.mean(cells[c]):+.4f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()