#!/usr/bin/env python3
"""robust_v4_screen.py — 第42轮中T筛选：StateHFBoost(V4) 是否提升执行噪声下的可信度保持率。
对比 BestCombo+ULI(冠军) 与 若干 V4 配置，看平滑族 mild/strong 的 gain 保持率。"""
import sys, os, json, time, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from learnable_bypass import ULI
from verify_architecture import BestCombo
from plugins_robust import RobustComboV4
from plugins import BasePlugin

class Chain(BasePlugin):
    def __init__(self, a, b): self.a, self.b = a, b
    def inject(self, x, **kw): return self.b.inject(self.a.inject(x, **kw), **kw)

T = 25600
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
    "clean":  {"obs_noise": 0.0, "act_noise": 0.0, "obs_spike_p": 0.0, "obs_spike_amp": 0.0,
                "act_spike_p": 0.0, "act_spike_amp": 0.0},
    "mild":   {"obs_noise": 0.02, "act_noise": 0.01, "obs_spike_p": 0.0, "obs_spike_amp": 0.0,
                "act_spike_p": 0.0, "act_spike_amp": 0.0},
    "strong": {"obs_noise": 0.05, "act_noise": 0.02, "obs_spike_p": 0.002, "obs_spike_amp": 0.4,
                "act_spike_p": 0.002, "act_spike_amp": 0.4},
}

def make_v4(build):
    return (lambda lam=build[0], thr=build[1], ema=build[2]:
            RobustComboV4(BestCombo(), copy.deepcopy(plug0_local()), lam=lam, thr_hf=thr, ema_win=ema))

plug0 = None
def plug0_local():
    global plug0
    return plug0

def main():
    global plug0
    print("训练 ULI ...", flush=True)
    plug0 = train_uli()
    variants = {
        "BestCombo+ULI": (lambda: RobustComboV4(BestCombo(), copy.deepcopy(plug0), g_max=0.0)),
        "V4(l=0.5,t=0.20)":  lambda: RobustComboV4(BestCombo(), copy.deepcopy(plug0), lam=0.5, thr_hf=0.20, ema_win=0.990),
        "V4(l=0.7,t=0.20)":  lambda: RobustComboV4(BestCombo(), copy.deepcopy(plug0), lam=0.7, thr_hf=0.20, ema_win=0.990),
        "V4(l=0.5,t=0.35)":  lambda: RobustComboV4(BestCombo(), copy.deepcopy(plug0), lam=0.5, thr_hf=0.35, ema_win=0.990),
    }
    out = {}
    for scen, noise in SCENARIOS.items():
        t0 = time.time()
        res = {"scenario": scen, "configs": {}}
        for name, make in variants.items():
            rows = {}
            for fam in FAMS:
                cbs, cps = [], []
                for s in SEEDS:
                    cb, cp = ci_pair(BasePolicy(n_joints=6, family=fam, seed=s), s, noise, make)
                    cbs.append(cb); cps.append(cp)
                cbm, cpm = float(np.mean(cbs)), float(np.mean(cps))
                rows[fam] = {"gain": round(cpm - cbm, 4), "opt": round(opt(cpm, cbm), 4)}
            res["configs"][name] = rows
        res["time_s"] = round(time.time() - t0)
        out[scen] = res
        print(f"\n[{scen}] {res['time_s']}s", flush=True)
        for name, r in res["configs"].items():
            g = np.mean([r[f]["gain"] for f in ["standard", "transformer"]]); gtot = np.mean([r[f]["gain"] for f in FAMS])
            print(f"  {name:18s} st_avg_gain={g:+.4f} total={gtot:+.4f}", flush=True)
    clean = out["clean"]["configs"]
    print("\n[平滑族 mild/strong retain = gain / gain_clean]")
    for name in clean:
        row = {}
        print(f"\n  {name}:", flush=True)
        for f in ["standard", "transformer"]:
            gc = clean[name][f]["gain"]
            row[f] = {sc: round(out[sc]["configs"][name][f]["gain"] / gc, 4) for sc in SCENARIOS}
            print(f"      {f:12s} " + " ".join(f"{k}={row[f][k]:+.3f}" for k in SCENARIOS), flush=True)
    json.dump({"T": T, "results": out}, open("robust_v4_screen.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("\n-> robust_v4_screen.json")

if __name__ == "__main__":
    main()