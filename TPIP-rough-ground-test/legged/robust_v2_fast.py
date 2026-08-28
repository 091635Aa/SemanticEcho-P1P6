#!/usr/bin/env python3
"""robust_v2_fast.py — 第40轮小T：PMS2(HF残差门控) vs BestCombo+ULI，可感知传感器噪声。
同时记录绝对增益 CI_p-CI_b 计算可信度(增益保持率)。"""
import sys, os, json, time, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from learnable_bypass import ULI
from verify_architecture import BestCombo
from plugins_robust import RobustComboV3
from plugins import BasePlugin

class Chain(BasePlugin):
    def __init__(self, a, b): self.a, self.b = a, b
    def inject(self, x, **kw): return self.b.inject(self.a.inject(x, **kw), **kw)

T = 5120
SEEDS = [42]
FAMS = ["standard", "transformer", "p2p"]

def opt(cp, cb): return (cp - cb) / (1 - cb + 1e-9)

def ci_pair(base, seed, noise, make):
    s0 = LeggedMicroSim(base, T=T, dt=0.01, seed=seed, **noise)
    tb = s0.run()
    cb = compute_coherence(tb["q"], tb["dq"], central_diff(tb["dq"], s0.dt), dt=s0.dt)["coherence_index"]
    p = make()
    if hasattr(p, "reset"): p.reset()
    ts = LeggedMicroSim(base, plugins=[p], T=T, dt=0.01, seed=seed, **noise)
    tp = ts.run()
    cp = compute_coherence(tp["q"], tp["dq"], central_diff(tp["dq"], ts.dt), dt=ts.dt)["coherence_index"]
    return cb, cp

def train_uli():
    plug = ULI(strength=1.0, W=40, lam=5e-3)
    data = plug.collect_trajectory(BasePolicy(n_joints=6, family="standard", seed=42), seed=42, t_len=60000)
    plug.fit_from(data); plug.strength = 0.5
    plug._amp = min(plug._amp, 0.5)
    return plug

SCENARIOS = {
    "clean":          {"obs_noise": 0.0,  "obs_spike_p": 0.0,   "obs_spike_amp": 0.0},
    "sensor_noise":   {"obs_noise": 0.04, "obs_spike_p": 0.0,   "obs_spike_amp": 0.0},
    "sensor_outlier": {"obs_noise": 0.0,  "obs_spike_p": 0.004, "obs_spike_amp": 0.5},
    "full_sensor":    {"obs_noise": 0.03, "obs_spike_p": 0.002, "obs_spike_amp": 0.5},
}

def main():
    print("训练 ULI ...", flush=True)
    plug = train_uli()
    build = {
        "BestCombo+ULI": lambda: Chain(BestCombo(), copy.deepcopy(plug)),
        "PMS2+ULI(V3)": lambda: RobustComboV3(BestCombo(), copy.deepcopy(plug)),
    }
    out = {}
    for scen, noise in SCENARIOS.items():
        t0 = time.time()
        res = {"scenario": scen, "configs": {}}
        for name, make in build.items():
            rows = {}
            for fam in FAMS:
                cbs, cps = [], []
                for s in SEEDS:
                    cb, cp = ci_pair(BasePolicy(n_joints=6, family=fam, seed=s), s, noise, make)
                    cbs.append(cb); cps.append(cp)
                gain = float(np.mean(cps) - np.mean(cbs))
                rows[fam] = {"cb": round(float(np.mean(cbs)), 4), "cp": round(float(np.mean(cps)), 4),
                             "gain": round(gain, 4), "opt": round(opt(np.mean(cps), np.mean(cbs)), 4)}
            res["configs"][name] = rows
        res["time_s"] = round(time.time() - t0)
        out[scen] = res
        print(f"\n[{scen}] {res['time_s']}s", flush=True)
        for name, r in res["configs"].items():
            avggain = np.mean([r[f]["gain"] for f in FAMS])
            avgopt = np.mean([r[f]["opt"] for f in FAMS])
            print(f"  {name:18s} gain={avggain:+.4f} opt={avgopt:+.4f}  ", flush=True)
            for f in FAMS:
                print(f"      {f:12s} cb={r[f]['cb']:.4f} cp={r[f]['cp']:.4f} gain={r[f]['gain']:+.4f} opt={r[f]['opt']:+.4f}", flush=True)
    json.dump({"T": T, "results": out}, open("robust_v2_fast.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("\n-> robust_v2_fast.json")

if __name__ == "__main__":
    main()