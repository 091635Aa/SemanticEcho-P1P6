#!/usr/bin/env python3
"""v12i.py — 校验"输入抖动率"是否分得开族×场景。
如果 clean/transformer 输入抖动≈0(无可修), 而 clean/standard 与 mild/strong-transformer
输入抖动>0(需去抖), 则可用"输入抖动→lam 门"实现: 保护clean/trans 同时保住 mild/strong 增益。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy, LeggedMicroSim
import cdiag

T = 5000
SEED = 99
DT = cdiag.DT
FAMS = ["standard", "transformer", "p2p"]


def jerk_stats(base, noise):
    s = LeggedMicroSim(base, plugins=[], T=T, dt=DT, seed=SEED, **noise)
    tr = s.run()
    a = tr["a"]                                   # (T,n) 原始动作(注入前)
    d = np.abs(np.diff(a, axis=0))
    jr = np.mean(d, axis=0) / (np.std(a, axis=0) + 1e-6)
    dd = np.abs(np.diff(np.diff(a, axis=0), axis=0))
    jj = np.mean(dd, axis=0) / (np.mean(d, axis=0) + 1e-6)
    return jr, jj


def main():
    print(f"输入抖动率(一阶jr / 二阶jj) [T={T}, seed={SEED}]", flush=True)
    for scen in ["clean", "mild", "strong"]:
        noise = cdiag.SCEN[scen]
        for fam in FAMS:
            base = BasePolicy(n_joints=6, family=fam, seed=SEED)
            jr, jj = jerk_stats(base, noise)
            print(f"[{scen}/{fam:11s}]  jr={np.round(jr.mean(),4)}  jj={np.round(jj.mean(),3)}"
                  f"  (jr_min={jr.min():.4f} jr_max={jr.max():.4f})", flush=True)


if __name__ == "__main__":
    main()