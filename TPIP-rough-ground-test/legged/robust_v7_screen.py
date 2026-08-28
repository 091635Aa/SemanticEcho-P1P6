#!/usr/bin/env python3
"""robust_v7_screen.py — 第45轮中T：提高 gammax 强度以最大化噪音鲁棒增益(平滑族,保护p2p)。"""
import sys, os, json, time, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from learnable_bypass import ULI
from verify_architecture import BestCombo
from plugins_robust import RobustComboV5, RobustComboV6
from plugins import BasePlugin

T = 8192
SEEDS = [42]
FAMS = ["standard", "transformer", "p2p"]

def opt(cp, cb): return (cp - cb) / (1 - cb + 1e-9)

def run_plugin_result(base, seed, noise, make):
    s0 = LeggedMicroSim(base, T=T, dt=0.01, seed=seed, **noise)
    tb = s0.run()
    cb = compute_coherence(tb["q"], tb["dq"], central_diff(tb["dq"], s0.dt), dt=s0.dt)["coherence_index"]
    p = make()
    if hasattr(p, "reset"): p.reset()
    ts = LeggedMicroSim(base, plugins=[p], T=T, dt=0.01, seed=seed, **noise)
    tp = ts.run()
    cp = compute_coherence(tp["q"], tp["dq"], central_diff(tp["dq"], ts.dt), dt=ts.dt)["coherence_index"]
    gate = shape = 0.0
    node = p
    while hasattr(node, "dnb"):
        node = node.dnb
    if hasattr(node, "mean_gate"):
        gate = node.mean_gate
    if hasattr(node, "mean_shape_gate"):
        shape = node.mean_shape_gate
    return cb, cp, gate, shape

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

def main():
    print("训练 ULI ...", flush=True)
    plug0 = train_uli()
    p0 = copy.deepcopy(plug0)
    variants = {
        "anchor-g3s5": (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=3.0, thr_hf=0.06, lam=0.6, ema_win=0.980, sh_cut=0.5)),
        "g4s5":        (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=4.0, thr_hf=0.06, lam=0.6, ema_win=0.980, sh_cut=0.5)),
        "g5s5":        (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=5.0, thr_hf=0.06, lam=0.6, ema_win=0.980, sh_cut=0.5)),
        "g4s7":        (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=4.0, thr_hf=0.06, lam=0.6, ema_win=0.980, sh_cut=0.7)),
        "g5s7":        (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=5.0, thr_hf=0.06, lam=0.6, ema_win=0.980, sh_cut=0.7)),
        "g6s7":        (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=6.0, thr_hf=0.06, lam=0.6, ema_win=0.980, sh_cut=0.7)),
    }
    out = {}
    for scen, noise in SCENARIOS.items():
        t0 = time.time()
        res = {"scenario": scen, "configs": {}}
        for name, make in variants.items():
            rows = {}
            for fam in FAMS:
                cb, cp, g, sg = run_plugin_result(BasePolicy(n_joints=6, family=fam, seed=42), 42, noise, make)
                rows[fam] = {"gain": round(cp - cb, 4), "opt": round(opt(cp, cb), 4),
                             "gate": round(float(g), 4), "shape": round(float(sg), 4)}
            res["configs"][name] = rows
        res["time_s"] = round(time.time() - t0)
        out[scen] = res
        print(f"\n[{scen}] {res['time_s']}s", flush=True)
        for name, r in res["configs"].items():
            st = np.mean([r[f]["gain"] for f in ["standard", "transformer"]])
            tot = np.mean([r[f]["gain"] for f in FAMS])
            print(f"  {name:12s} st_gain={st:+.4f} total={tot:+.4f}   "
                  + " ".join(f"{f[:2]}:{r[f]['gain']:+.3f}(g{r[f]['gate']:.1f}/s{r[f]['shape']:.1f})" for f in FAMS),
                  flush=True)
    json.dump({"T": T, "results": out}, open("robust_v7_screen.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("\n-> robust_v7_screen.json")

if __name__ == "__main__":
    main()