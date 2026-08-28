#!/usr/bin/env python3
"""robust_v6_screen.py — 第44轮中T：DNB2(相干形状门控,保护p2p) vs champ，测三噪音场景
平滑族增益/可信度 + Universal(p2p 保护)，并输出 shape_gate 确认相干门分离。"""
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
        "champ":      (lambda: RobustComboV5(BestCombo(), copy.deepcopy(p0), g_max=0.0)),
        "v6g2s5":     (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=2.0, thr_hf=0.06, lam=0.6, ema_win=0.980, sh_cut=0.5)),
        "v6g2s3":     (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=2.0, thr_hf=0.06, lam=0.6, ema_win=0.980, sh_cut=0.3)),
        "v6g2s8":     (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=2.0, thr_hf=0.06, lam=0.6, ema_win=0.980, sh_cut=0.8)),
        "v6g3s5":     (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=3.0, thr_hf=0.06, lam=0.6, ema_win=0.980, sh_cut=0.5)),
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
            print(f"  {name:10s} st_gain={st:+.4f} total={tot:+.4f}   "
                  + " ".join(f"{f[:2]}:{r[f]['gain']:+.3f}(g{r[f]['gate']:.2f}/s{r[f]['shape']:.2f})" for f in FAMS),
                  flush=True)
    clean = out["clean"]["configs"]
    print("\n[平滑族 retain = gain_scen/gain_clean]")
    for name in clean:
        print(f"\n  {name}:", flush=True)
        for f in ["standard", "transformer"]:
            gc = clean[name][f]["gain"]
            print(f"      {f:12s} " + " ".join(
                f"{sc}={(out[sc]['configs'][name][f]['gain']/gc if abs(gc)>1e-9 else float('nan')):+.3f}"
                for sc in SCENARIOS), flush=True)
    json.dump({"T": T, "results": out}, open("robust_v6_screen.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("\n-> robust_v6_screen.json")

if __name__ == "__main__":
    main()