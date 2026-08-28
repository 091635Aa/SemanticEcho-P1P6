#!/usr/bin/env python3
"""robust_full.py — 第41轮全T：可信度 = 绝对增益保持率 (gain retention)。
对比 BestCombo+ULI(冠军) vs PMS2+ULI(V3,HF残差门控)，
覆盖 clean / mild(obs+act) / strong(obs+act+双侧重噪声)。
可信度 cred = gain_noise / gain_clean，衡量插件在噪声下保留价值的能力(不受 opt 分母失真)。"""
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

T = 51200
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
    "clean":   {"obs_noise": 0.0, "act_noise": 0.0, "obs_spike_p": 0.0, "obs_spike_amp": 0.0,
                 "act_spike_p": 0.0, "act_spike_amp": 0.0},
    "mild":    {"obs_noise": 0.02, "act_noise": 0.01, "obs_spike_p": 0.0, "obs_spike_amp": 0.0,
                 "act_spike_p": 0.0, "act_spike_amp": 0.0},
    "strong":  {"obs_noise": 0.05, "act_noise": 0.02, "obs_spike_p": 0.002, "obs_spike_amp": 0.4,
                 "act_spike_p": 0.002, "act_spike_amp": 0.4},
}

def main():
    print("训练 ULI ...", flush=True)
    plug = train_uli()
    build = {
        "BestCombo+ULI": lambda: Chain(BestCombo(), copy.deepcopy(plug)),
        "PMS2+ULI(V3)":  lambda: RobustComboV3(BestCombo(), copy.deepcopy(plug)),
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
                cbm, cpm = float(np.mean(cbs)), float(np.mean(cps))
                rows[fam] = {"cb": round(cbm, 4), "cp": round(cpm, 4),
                             "gain": round(cpm - cbm, 4), "opt": round(opt(cpm, cbm), 4)}
            res["configs"][name] = rows
        res["time_s"] = round(time.time() - t0)
        out[scen] = res
        print(f"\n[{scen}] {res['time_s']}s", flush=True)
        for name, r in res["configs"].items():
            g = np.mean([r[f]["gain"] for f in FAMS]); o = np.mean([r[f]["opt"] for f in FAMS])
            print(f"  {name:18s} gain={g:+.4f} opt={o:+.4f}", flush=True)
            for f in FAMS:
                print(f"      {f:12s} cb={r[f]['cb']:.4f} cp={r[f]['cp']:.4f} gain={r[f]['gain']:+.4f} opt={r[f]['opt']:+.4f}", flush=True)
    # 可信度：gain retention = gain_scen / gain_clean (不取平均族时逐族算)
    clean = out["clean"]["configs"]
    print("\n[可信度 gain retention = gain_noise/gain_clean]")
    cred = {}
    for name in clean:
        row = {}
        for f in FAMS:
            gc = clean[name][f]["gain"]
            row[f] = {sc: (round(out[sc]["configs"][name][f]["gain"] / gc, 4)
                            if abs(gc) > 1e-9 else float("nan")) for sc in SCENARIOS}
        cred[name] = row
        print(f"\n  {name}:")
        for f in FAMS:
            print(f"      {f:12s} " + " ".join(f"{k}={row[f][k]:+.3f}" for k in SCENARIOS), flush=True)
    json.dump({"T": T, "seeds": SEEDS, "results": out, "credibility_retention": cred},
              open("robust_full.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("\n-> robust_full.json")

if __name__ == "__main__":
    main()