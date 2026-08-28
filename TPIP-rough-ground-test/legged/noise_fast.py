#!/usr/bin/env python3
"""noise_fast.py — 第39轮小T快速调参：PhaseMatchedSmoother(PMS) 三档噪声对比。
目的：找 β/ema 使 clean≈恒等 (avg≈+0.53)，且 mild/strong 优于 BestCombo+ULI。"""
import sys, os, json, time, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from learnable_bypass import ULI
from verify_architecture import BestCombo
from plugins_robust import RobustComboV2
from plugins import BasePlugin

class Chain(BasePlugin):
    def __init__(self, a, b): self.a, self.b = a, b
    def inject(self, x, **kw): return self.b.inject(self.a.inject(x, **kw), **kw)

T = 5120
SEEDS = [42]
FAMS = ["standard", "transformer", "p2p"]

def opt(cp, cb): return (cp - cb) / (1 - cb + 1e-9)

def ci_for(base, seed, noise, make):
    s0 = LeggedMicroSim(base, T=T, dt=0.01, seed=seed, **noise)
    tb = s0.run()
    cb = compute_coherence(tb["q"], tb["dq"], central_diff(tb["dq"], s0.dt), dt=s0.dt)["coherence_index"]
    p = make()
    if hasattr(p, "reset"): p.reset()
    ts = LeggedMicroSim(base, plugins=[p], T=T, dt=0.01, seed=seed, **noise)
    tp = ts.run()
    cp = compute_coherence(tp["q"], tp["dq"], central_diff(tp["dq"], ts.dt), dt=ts.dt)["coherence_index"]
    return opt(cp, cb)

def train_uli():
    plug = ULI(strength=1.0, W=40, lam=5e-3)
    data = plug.collect_trajectory(BasePolicy(n_joints=6, family="standard", seed=42), seed=42, t_len=60000)
    plug.fit_from(data); plug.strength = 0.5
    plug._amp = min(plug._amp, 0.5)
    return plug

SCENARIOS = {
    "clean":       {"obs_noise": 0.0,  "act_noise": 0.0,  "spike_p": 0.0,   "spike_amp": 0.0},
    "mild_sensor": {"obs_noise": 0.02, "act_noise": 0.01, "spike_p": 0.0,   "spike_amp": 0.0},
    "strong":      {"obs_noise": 0.05, "act_noise": 0.02, "spike_p": 0.003, "spike_amp": 0.4},
}

BETAS = [0.08, 0.12, 0.18, 0.25]
EMAS = [0.985, 0.970]

def main():
    print("训练 ULI ...", flush=True)
    plug = train_uli()
    build = {
        "BestCombo+ULI": lambda: Chain(BestCombo(), copy.deepcopy(plug)),
    }
    for be in BETAS:
        for em in EMAS:
            build[f"PMS(b={be},e={em})"] = (lambda be=be, em=em:
                RobustComboV2(BestCombo(), copy.deepcopy(plug), beta=be, beta_max=be+0.25, ema_win=em))
    out = {}
    for scen, noise in SCENARIOS.items():
        t0 = time.time()
        res = {"scenario": scen, "configs": {}}
        for name, make in build.items():
            per = {fam: [ci_for(BasePolicy(n_joints=6, family=fam, seed=s), s, noise, make) for s in SEEDS] for fam in FAMS}
            avgs = [(per["standard"][i] + per["transformer"][i]) / 2 for i in range(len(SEEDS))]
            uni = all(float(x) > 0 for fam in FAMS for x in per[fam])
            res["configs"][name] = {"avg": float(np.mean(avgs)),
                                    "families": {fam: round(float(np.mean(v)), 4) for fam, v in per.items()},
                                    "universal": uni}
        res["time_s"] = round(time.time() - t0)
        out[scen] = res
        print(f"\n[{scen}] {res['time_s']}s", flush=True)
        for name, r in res["configs"].items():
            print(f"  {name:22s} avg={r['avg']:+.4f} s={r['families']['standard']:+.4f} "
                  f"t={r['families']['transformer']:+.4f} p={r['families']['p2p']:+.4f} "
                  f"uni={'Y' if r['universal'] else 'N'}", flush=True)
    json.dump({"T": T, "results": out}, open("noise_fast.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("\n-> noise_fast.json")

if __name__ == "__main__":
    main()