#!/usr/bin/env python3
"""v12c.py — 分清 transformer-clean 损失是稳态效应还是瞬态/模板预热污染。
对照 base/plugin 的 C 在前50% vs 后50% 分段值。若后段 plugin≈base，则损失来自
模板预热(前几个非稳态周期污染模板→把干净稳态周期拉歪)。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy, LeggedMicroSim
from plugins_robust import PhaseRecomb
import cdiag

T = 5000; SEED = 99
cdiag.T = T
DT = cdiag.DT


def half_C(traj, fr):
    a = int(fr * len(traj["q"]))
    tr = {"q": traj["q"][a:], "dq": traj["dq"][a:], "t": traj["t"][a:]}
    return cdiag.c_cons(tr)


def probe(fam, lam):
    base = BasePolicy(n_joints=6, family=fam, seed=SEED)
    p = PhaseRecomb(lam=lam, alpha=0.05, blend=False); p.reset()
    s0 = LeggedMicroSim(base, T=T, dt=DT, seed=SEED, **cdiag.SCEN["clean"])
    tb = s0.run()
    s1 = LeggedMicroSim(base, plugins=[p], T=T, dt=DT, seed=SEED, **cdiag.SCEN["clean"])
    tp = s1.run()
    print(f"  {fam} lam{lam}: base Cfull={cdiag.c_cons(tb):.3f} "
          f"Ctr{half_C(tb,0.5):.3f}  plug Cfull={cdiag.c_cons(tp):.3f} "
          f"Ctr{half_C(tp,0.5):.3f}  (后50%差={half_C(tp,0.5)-half_C(tb,0.5):+.3f})", flush=True)


if __name__ == "__main__":
    print("transformer-clean vs standard-clean 分段一致性诊断:", flush=True)
    for lam in [0.3, 0.7]:
        probe("transformer", lam)
        probe("standard", lam)