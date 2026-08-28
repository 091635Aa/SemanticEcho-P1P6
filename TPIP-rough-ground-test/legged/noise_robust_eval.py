#!/usr/bin/env python3
"""
noise_robust_eval.py — 抗噪方案评估 (第38轮)：RobustCombo vs 基线(noise_suite)

对每档噪声测：BestCombo、BestCombo+ULI、RobustCombo(去噪前置+ULI)。
关注：avg、每族、Universal、相对 clean 衰减。
验证"去噪前置让干净ULI恢复有效"假设。
"""
import sys, os, json, time, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from learnable_bypass import ULI
from verify_architecture import BestCombo
from plugins_robust import RobustCombo
from plugins import BasePlugin


class Chain(BasePlugin):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def inject(self, x, **kw):
        return self.b.inject(self.a.inject(x, **kw), **kw)

T = 51200
SEEDS = [42]
FAMS = ["standard", "transformer", "p2p"]


def opt(cp, cb): return (cp - cb) / (1 - cb + 1e-9)


def ci_for(base, seed, noise, make):
    s0 = LeggedMicroSim(base, T=T, dt=0.01, seed=seed, **noise)
    tb = s0.run()
    cb = compute_coherence(tb["q"], tb["dq"], central_diff(tb["dq"], s0.dt),
                          dt=s0.dt)["coherence_index"]
    p = make()
    if hasattr(p, "reset"): p.reset()
    ts = LeggedMicroSim(base, plugins=[p], T=T, dt=0.01, seed=seed, **noise)
    tp = ts.run()
    cp = compute_coherence(tp["q"], tp["dq"], central_diff(tp["dq"], ts.dt),
                          dt=ts.dt)["coherence_index"]
    return opt(cp, cb)


def train_uli():
    plug = ULI(strength=1.0, W=40, lam=5e-3)
    data = plug.collect_trajectory(BasePolicy(n_joints=6, family="standard", seed=42),
                                   seed=42, t_len=60000)
    plug.fit_from(data)
    plug.strength = 0.5
    plug._amp = min(plug._amp, 0.5)
    return plug


SCENARIOS = {
    "clean":       {"obs_noise": 0.0,  "act_noise": 0.0,  "spike_p": 0.0,   "spike_amp": 0.0},
    "mild_sensor": {"obs_noise": 0.02, "act_noise": 0.01, "spike_p": 0.0,   "spike_amp": 0.0},
    "strong":      {"obs_noise": 0.05, "act_noise": 0.02, "spike_p": 0.003, "spike_amp": 0.4},
}


def main():
    print("训练 ULI ...", flush=True)
    plug = train_uli()

    build = {
        "BestCombo":     (lambda: BestCombo()),
        "BestCombo+ULI": (lambda: Chain(BestCombo(), copy.deepcopy(plug))),
        "RobustCombo":   (lambda: RobustCombo(BestCombo(), copy.deepcopy(plug))),
    }

    all_r = {}
    for scen, noise in SCENARIOS.items():
        t0 = time.time()
        res = {"scenario": scen, "configs": {}}
        for name, make in build.items():
            per = {fam: [ci_for(BasePolicy(n_joints=6, family=fam, seed=s), s, noise, make)
                         for s in SEEDS] for fam in FAMS}
            avgs = [(per["standard"][i] + per["transformer"][i]) / 2 for i in range(len(SEEDS))]
            uni = all(float(x) > 0 for fam in FAMS for x in per[fam])
            res["configs"][name] = {"avg": float(np.mean(avgs)),
                                    "families": {fam: round(float(np.mean(v)), 4) for fam, v in per.items()},
                                    "universal": uni}
        res["time_s"] = round(time.time() - t0)
        all_r[scen] = res
        print(f"\n[{scen}] {res['time_s']}s", flush=True)
        for name, r in res["configs"].items():
            print(f"  {name:16s} avg={r['avg']:+.4f} s={r['families']['standard']:+.4f} "
                  f"t={r['families']['transformer']:+.4f} p={r['families']['p2p']:+.4f} "
                  f"uni={'Y' if r['universal'] else 'N'}", flush=True)

    # 可信度
    clean = all_r["clean"]["configs"]
    print("\n[可信度: avg 相对 clean 衰减]")
    robust = {}
    for name in clean:
        b = clean[name]["avg"]
        drops = {sc: round(all_r[sc]["configs"][name]["avg"] - b, 4) for sc in SCENARIOS}
        robust[name] = {"clean": round(b, 4), "drops": drops}
        print(f"  {name:16s} clean={b:+.4f}  " + " ".join(f"{k}={v:+.4f}" for k, v in drops.items()), flush=True)
    json.dump({"T": T, "seeds": SEEDS, "scenarios": SCENARIOS,
               "results": all_r, "robustness": robust},
              open("noise_robust_eval.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("\n-> noise_robust_eval.json")


if __name__ == "__main__":
    main()