#!/usr/bin/env python3
"""blb_screen.py — 第47轮 BlueprintLockedBoost(BLB) 三维度+CI 扫描。

对比 baseline / champ(BestCombo+ULI) / v6g3s5 / v8(BestCombo+ULI+BLB)。
输出各阶段 绝对三维度(C,L,R)+CI，定位 R_ref 被砸的源头，并验证 BLB 是否
在保住 R_ref 的同时拿到 CI/SNR 增益。用法: python3 blb_screen.py [T] [seed]
"""
import sys, os, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from learnable_bypass import ULI
from verify_architecture import BestCombo
from plugins import BasePlugin
from plugins_robust import RobustComboV6, RobustComboV8, RobustComboV9, RobustComboV10
import cdiag

T = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 99
FAMS = ["standard", "transformer", "p2p"]


def train_uli():
    plug = ULI(strength=1.0, W=40, lam=5e-3)
    data = plug.collect_trajectory(BasePolicy(n_joints=6, family="standard", seed=42), seed=42, t_len=60000)
    plug.fit_from(data); plug.strength = 0.5; plug._amp = min(plug._amp, 0.5)
    return plug


def metrics(traj):
    return {"C": cdiag.c_cons(traj), "L": cdiag.l_link(traj), "R": cdiag.r_ref(traj),
            "CI": compute_coherence(traj["q"], traj["dq"], central_diff(traj["dq"], cdiag.DT), dt=cdiag.DT)["coherence_index"]}


class ComboOnly(BasePlugin):
    def __init__(self, combo): self.combo = combo
    def reset(self): pass
    def inject(self, a, **kw): return self.combo.inject(a, **kw)


def main():
    print(f"BLB 阶段扫描 [T={T}, seed={SEED}]", flush=True)
    p0 = train_uli()
    cfgs = {
        "base":             (lambda: None),
        "v6g3s5":           (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=3.0, thr_hf=0.06, p=1.0, sh_cut=0.5)),
        "v9-lam4":          (lambda: RobustComboV9(copy.deepcopy(p0), lam=0.4, rrcut=0.35)),
        "v9-lam6":          (lambda: RobustComboV9(copy.deepcopy(p0), lam=0.6, rrcut=0.35)),
        "v10-lam4":         (lambda: RobustComboV10(BestCombo(), copy.deepcopy(p0), lam=0.4, rrcut=0.35)),
        "v10-lam6":         (lambda: RobustComboV10(BestCombo(), copy.deepcopy(p0), lam=0.6, rrcut=0.35)),
        "v9-lam6-c40":      (lambda: RobustComboV9(copy.deepcopy(p0), lam=0.6, rrcut=0.40)),
    }
    out = {}
    for scen, noise in cdiag.SCEN.items():
        out[scen] = {}
        print(f"--- {scen} ---", flush=True)
        for fam in FAMS:
            base = BasePolicy(n_joints=6, family=fam, seed=SEED)
            row = {}
            for name, make in cfgs.items():
                mb, mp = cdiag.run_metrics(base, SEED, noise, make)
                row[name] = {"base": mb, "plug": mp}
            out[scen][fam] = row
            b = row["base"]; pb = row["base"]
            def fmt_delta(cfg):
                p = row[cfg]["plug"]
                return (f"{cfg:9s} dC={p['C']-b['C']:+.3f} dL={p['L']-b['L']:+.3f} "
                        f"dR={p['R']-b['R']:+.3f} dCI={p['CI']-b['CI']:+.3f}")
            line = f"[{scen}/{fam:11s}]"
            for cfg in ["base","v6g3s5","v10-lam4","v10-lam6","v9-lam4","v9-lam6"]:
                p = row[cfg]["plug"]
                line += (f"  | {cfg[:8]:8s} C{p['C']:+.3f} L{p['L']:+.3f} R{p['R']:+.3f} CI{p['CI']:+.3f}")
            print(line, flush=True)
    json.dump({"T": T, "seed": SEED, "results": out},
              open(f"blb_s{SEED}_T{T}.json", "w"), ensure_ascii=False, indent=1, default=float)


import json
if __name__ == "__main__":
    main()