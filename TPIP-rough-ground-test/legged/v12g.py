#!/usr/bin/env python3
"""v12g.py — 诊断 PhaseRecomb 的逐关节操作点 (rr 残差比 / r2 1Hz相干 / 门值/β)。
目标：量化 clean/transformer 为何门仍开 → 导致 C 净损，为"bandpass门"和"β封顶"调参。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy, LeggedMicroSim
from plugins_robust import PhaseRecomb
import cdiag

T = 5000
SEED = 99
FAMS = ["standard", "transformer", "p2p"]
DT = cdiag.DT


def run_one(base, noise, plug):
    if hasattr(plug, "reset"):
        plug.reset()
    s = LeggedMicroSim(base, plugins=[plug], T=T, dt=DT, seed=SEED, **noise)
    s.run()
    e = plug._rr_ema if plug._rr_ema is not None else np.zeros(6)
    r2 = plug._r2_ema if plug._r2_ema is not None else np.zeros(6)
    lamj = np.full(6, plug.lam)
    rrcut, r2f, r2c = plug.rrcut, plug.r2_floor, plug.r2_ceil
    coh = (1.0 - np.minimum(e / rrcut, 1.0)) ** plug.rr_pow
    fac = np.clip((r2 - r2f) / (r2c - r2f + 1e-9), 0.0, 1.0) ** plug.r2_pow
    beta = np.clip((r2 - plug.r2b_lo) / (plug.r2b_hi - plug.r2b_lo + 1e-9), 0.0, 1.0) ** plug.blend_pow
    return e, r2, coh * fac * lamj, beta


def main():
    print(f"诊断 rr/r2/门值/β 逐关节 [T={T}, seed={SEED}]", flush=True)
    plug = PhaseRecomb(lam=0.85, alpha=0.05, blend=True,
                       r2b_lo=0.45, r2b_hi=0.85, min_cyc=2)
    for scen in ["clean", "mild", "strong"]:
        noise = cdiag.SCEN[scen]
        for fam in FAMS:
            base = BasePolicy(n_joints=6, family=fam, seed=SEED)
            e, r2, gate, beta = run_one(base, noise, plug)
            print(f"[{scen}/{fam:11s}]", flush=True)
            print(f"   rr = {np.round(e,3)}   r2 = {np.round(r2,3)}", flush=True)
            print(f"   门值= {np.round(gate,3)}   β  = {np.round(beta,3)}", flush=True)


if __name__ == "__main__":
    main()