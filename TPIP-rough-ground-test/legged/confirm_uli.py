#!/usr/bin/env python3
"""
confirm_uli.py — 长 T 确认：BestCombo+可学引导(s0.5) 与 无参底座 对比
T=204800, seed=42, 三族。抗短 T 方差，判定可学权重版本在长轨迹下的稳定表现与 Universal。
"""
import sys, os, json, time, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy
from learnable_bypass import ULI, run_sim
from verify_architecture import BestCombo
from plugins import BasePlugin

T = 204800
SEEDS = [42]
FAMS = ["standard", "transformer", "p2p"]


def opt(cp, cb): return (cp - cb) / (1 - cb + 1e-9)


class Chain(BasePlugin):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def inject(self, a, **kw):
        return self.b.inject(self.a.inject(a, **kw), **kw)


def train_uli():
    plug = ULI(strength=1.0, W=40, lam=5e-3)
    data = plug.collect_trajectory(BasePolicy(n_joints=6, family="standard", seed=42),
                                   seed=42, t_len=80000)
    plug.fit_from(data)
    return plug


def est(fam, seed, make):
    base = BasePolicy(n_joints=6, family=fam, seed=seed)
    tb = run_sim(base, [], T, seed)
    cb = compute_coherence(tb["q"], tb["dq"], central_diff(tb["dq"], tb["dt"]),
                          dt=tb["dt"])["coherence_index"]
    p = make()
    if hasattr(p, "reset"): p.reset()
    tp = run_sim(base, [p], T, seed)
    cp = compute_coherence(tp["q"], tp["dq"], central_diff(tp["dq"], tp["dt"]),
                          dt=tp["dt"])["coherence_index"]
    return opt(cp, cb)


def main():
    print("训练可学 ULI ...", flush=True)
    plug = train_uli()
    plug._amp = min(plug._amp, 0.5)
    plug.strength = 0.5
    print("  amp→0.5, 结合 BestCombo+uli(0.5)", flush=True)

    res = {"T": T, "seeds": SEEDS, "configs": {}}
    configs = {
        "ref_BestCombo": lambda: BestCombo(),
        "BestCombo_uli_0.5": lambda: Chain(BestCombo(), copy.deepcopy(plug)),
    }
    for name, make in configs.items():
        t0 = time.time()
        per = {fam: [est(fam, s, make) for s in SEEDS] for fam in FAMS}
        avgs = [(per["standard"][i] + per["transformer"][i]) / 2 for i in range(1)]
        uni = all(float(x) > 0 for fam in FAMS for x in per[fam])
        r = {"name": name, "avg": float(np.mean(avgs)),
             "families": {fam: float(np.mean(v)) for fam, v in per.items()},
             "universal": uni, "time_s": round(time.time() - t0)}
        res["configs"][name] = r
        print(f"  {name:22s} avg={r['avg']:+.4f} s={r['families']['standard']:+.4f} "
              f"t={r['families']['transformer']:+.4f} p={r['families']['p2p']:+.4f} "
              f"uni={'Y' if uni else 'N'}  {r['time_s']}s", flush=True)
    json.dump(res, open("confirm_uli.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()