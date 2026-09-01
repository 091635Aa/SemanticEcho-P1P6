#!/usr/bin/env python3
"""v12b.py — 定位 transformer-clean 净损失根因：扫描低 lam，并测逐关节 rr。
目标：让已高度一致(transformer-clean) 的家族近乎恒等(不造损)，同时保留 standard
与 transformer-mild/strong 的增益。
用法: python3 v12b.py [seed]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy, LeggedMicroSim
from plugins_robust import PhaseRecomb
import cdiag

T = 5000; SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 99
cdiag.T = T   # 覆写 cdiag 模块级 T(否则它读 argv 把 SEED 当 T, 轨迹过短)


def run_and_metrics(base, make, noise):
    mb, mp = cdiag.run_metrics(base, SEED, noise, make)
    dC = mp['C'] - mb['C']; dL = mp['L'] - mb['L']
    dR = mp['R'] - mb['R']; dCI = mp['CI'] - mb['CI']
    return dC, dL, dR, dCI


def probe_rr(fam, lam, noise):
    """跑一遍采样 rr_ema 终值，看干扰量级。"""
    base = BasePolicy(n_joints=6, family=fam, seed=SEED)
    p = PhaseRecomb(lam=lam, alpha=0.05, blend=False)
    p.reset()
    s = LeggedMicroSim(base, plugins=[p], T=T, dt=cdiag.DT, seed=SEED, **noise)
    s.run()
    return None if p._rr_ema is None else p._rr_ema.copy()


def main():
    cfgs = {
        "tpl-l30": (lambda: PhaseRecomb(lam=0.30, alpha=0.05, blend=False)),
        "tpl-l20": (lambda: PhaseRecomb(lam=0.20, alpha=0.05, blend=False)),
        "tpl-l12": (lambda: PhaseRecomb(lam=0.12, alpha=0.05, blend=False)),
        "tpl-l30-ada": (lambda: PhaseRecomb(lam=0.30, alpha=0.05, blend=False,
                                           adapt=True, lam_ref=0.08, lam_min=0.0)),
        "tpl-l30-ada15": (lambda: PhaseRecomb(lam=0.30, alpha=0.05, blend=False,
                                              adapt=True, lam_ref=0.15, lam_min=0.0)),
        "bl-l7-ada20": (lambda: PhaseRecomb(lam=0.7, alpha=0.05, blend=True,
                                            r2b_lo=0.35, r2b_hi=0.75,
                                            adapt=True, lam_ref=0.20, lam_min=0.0)),
    }
    order = ["tpl-l30", "tpl-l20", "tpl-l12", "tpl-l30-ada", "tpl-l30-ada15", "bl-l7-ada20"]
    for scen in ["clean", "mild"]:
        noise = cdiag.SCEN[scen]
        for fam in ["standard", "transformer"]:
            base = BasePolicy(n_joints=6, family=fam, seed=SEED)
            mb = cdiag.run_metrics(base, SEED, noise, (lambda: None))[0]
            line = f"[{scen}/{fam:10s}] base C{mb['C']:.3f} R{mb['R']:.3f} CI{mb['CI']:.3f}"
            for cfg in order:
                dC, dL, dR, dCI = run_and_metrics(base, cfgs[cfg], noise)
                line += f" | {cfg:14s} dC{dC:+.3f} dL{dL:+.3f} dR{dR:+.3f} dCI{dCI:+.3f}"
            print(line, flush=True)
    # 探针：transformer-clean/standard-clean 的 rr 终值
    print("rr probe:", flush=True)
    for fam in ["standard", "transformer"]:
        for lam in [0.3, 0.7]:
            rr = probe_rr(fam, lam, cdiag.SCEN["clean"])
            print(f"  {fam}/clean lam{lam:.1f}: rr={np.round(rr,3) if rr is not None else None}", flush=True)


if __name__ == "__main__":
    main()