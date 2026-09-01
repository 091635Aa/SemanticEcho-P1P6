#!/usr/bin/env python3
"""v12d.py — transformer-clean: 把目标推向"纯1Hz骨干"(漂移无关) + 高lam，看能否升C。
对比 模板(慢速自适应,会跟踪漂移) vs 纯1Hz骨干(β→1, 漂移无关)。"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legged_env import BasePolicy, LeggedMicroSim
from plugins_robust import PhaseRecomb
import cdiag

T = 5000; SEED = 99
cdiag.T = T; DT = cdiag.DT


def steady_C(traj):
    a = int(0.5 * len(traj["q"]))
    return cdiag.c_cons({"q": traj["q"][a:], "dq": traj["dq"][a:], "t": traj["t"][a:]})


def probe(fam, make, label):
    base = BasePolicy(n_joints=6, family=fam, seed=SEED)
    p = make(); p.reset()
    tb = LeggedMicroSim(base, T=T, dt=DT, seed=SEED, **cdiag.SCEN["clean"]).run()
    tp = LeggedMicroSim(base, plugins=[p], T=T, dt=DT, seed=SEED, **cdiag.SCEN["clean"]).run()
    bC = cdiag.c_cons(tb); bCs = steady_C(tb)
    pC = cdiag.c_cons(tp); pCs = steady_C(tp)
    print(f"  {fam}/{label:16s} Cfull {bC:.3f}->{pC:.3f}{pC-bC:+.3f}  "
          f"Ctr {bCs:.3f}->{pCs:.3f}{pCs-bCs:+.3f}", flush=True)


def main():
    c = {
        "tpl-l85":   (lambda: PhaseRecomb(lam=0.85, alpha=0.05, blend=False)),
        "bb-l85":    (lambda: PhaseRecomb(lam=0.85, alpha=0.05, blend=True, r2b_lo=0.0, r2b_hi=0.01)),
        "bb-l95":    (lambda: PhaseRecomb(lam=0.95, alpha=0.05, blend=True, r2b_lo=0.0, r2b_hi=0.01)),
        "tpl-l95":   (lambda: PhaseRecomb(lam=0.95, alpha=0.05, blend=False)),
        "bb-l85-lam":(lambda: PhaseRecomb(lam=0.85, alpha=0.35, blend=True, r2b_lo=0.0, r2b_hi=0.01)),
    }
    for fan, make in c.items():
        probe("transformer", make, fan)
    print("--- standard 对照 ---", flush=True)
    for fan, make in {"bb-l85": c["bb-l85"], "tpl-l85": c["tpl-l85"]}.items():
        probe("standard", make, fan)


if __name__ == "__main__":
    main()