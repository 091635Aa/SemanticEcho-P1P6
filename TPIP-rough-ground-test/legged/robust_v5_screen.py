#!/usr/bin/env python3
"""robust_v5_screen.py — 第43轮中T：DNB(相干幅度放大) vs 冠军，测执行噪声下平滑族增益/可信度，
并输出 DNB 门控均值以确认检测确实触发。"""
import sys, os, json, time, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from learnable_bypass import ULI
from verify_architecture import BestCombo
from plugins_robust import RobustComboV5
from plugins import BasePlugin

T = 25600
SEEDS = [42]
FAMS = ["standard", "transformer", "p2p"]

def opt(cp, cb): return (cp - cb) / (1 - cb + 1e-9)

def ci_pair(base, seed, noise, make, want_plugin=False):
    s0 = LeggedMicroSim(base, T=T, dt=0.01, seed=seed, **noise)
    tb = s0.run()
    cb = compute_coherence(tb["q"], tb["dq"], central_diff(tb["dq"], s0.dt), dt=s0.dt)["coherence_index"]
    p = make()
    if hasattr(p, "reset"): p.reset()
    ts = LeggedMicroSim(base, plugins=[p], T=T, dt=0.01, seed=seed, **noise)
    tp = ts.run()
    cp = compute_coherence(tp["q"], tp["dq"], central_diff(tp["dq"], ts.dt), dt=ts.dt)["coherence_index"]
    gate = 0.0
    if want_plugin:
        # 递归抓取 DNB .mean_gate
        node = p
        while hasattr(node, "dnb") or hasattr(node, "_dummy"):
            node = node.dnb
        if hasattr(node, "mean_gate"):
            gate = node.mean_gate
    return cb, cp, gate

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
    global plug0
    print("训练 ULI ...", flush=True)
    plug0 = train_uli()
    plug0_local = plug0
    variants = {
        "champ":     (lambda: RobustComboV5(BestCombo(), copy.deepcopy(plug0_local), g_max=0.0)),
        "DNB-g1t06": (lambda: RobustComboV5(BestCombo(), copy.deepcopy(plug0_local), gammax=1.0, thr_hf=0.06, lam=0.6, ema_win=0.980)),
        "DNB-g2t06": (lambda: RobustComboV5(BestCombo(), copy.deepcopy(plug0_local), gammax=2.0, thr_hf=0.06, lam=0.6, ema_win=0.980)),
        "DNB-g1t10": (lambda: RobustComboV5(BestCombo(), copy.deepcopy(plug0_local), gammax=1.0, thr_hf=0.10, lam=0.6, ema_win=0.980)),
        "DNB-g15t05":(lambda: RobustComboV5(BestCombo(), copy.deepcopy(plug0_local), gammax=1.5, thr_hf=0.05, lam=0.6, ema_win=0.980)),
    }
    out = {}
    for scen, noise in SCENARIOS.items():
        t0 = time.time()
        res = {"scenario": scen, "configs": {}}
        for name, make in variants.items():
            rows = {}
            for fam in FAMS:
                cbs, cps, gates = [], [], []
                for s in SEEDS:
                    cb, cp, g = ci_pair(BasePolicy(n_joints=6, family=fam, seed=s), s, noise, make, want_plugin=True)
                    cbs.append(cb); cps.append(cp); gates.append(g)
                cbm, cpm = float(np.mean(cbs)), float(np.mean(cps))
                rows[fam] = {"gain": round(cpm - cbm, 4), "opt": round(opt(cpm, cbm), 4),
                             "gate": round(float(np.mean(gates)), 4)}
            res["configs"][name] = rows
        res["time_s"] = round(time.time() - t0)
        out[scen] = res
        print(f"\n[{scen}] {res['time_s']}s", flush=True)
        for name, r in res["configs"].items():
            st = np.mean([r[f]["gain"] for f in ["standard", "transformer"]])
            tot = np.mean([r[f]["gain"] for f in FAMS])
            gs = np.mean([r[f]["gate"] for f in ["standard", "transformer"]])
            print(f"  {name:12s} st_gain={st:+.4f} total={tot:+.4f} gate_s={gs:.3f}   "
                  + " ".join(f"{f[:2]}:{r[f]['gain']:+.3f}(g{r[f]['gate']:.2f})" for f in FAMS), flush=True)
    clean = out["clean"]["configs"]
    print("\n[平滑族 mild/strong retain = gain/gain_clean]")
    for name in clean:
        print(f"\n  {name}:", flush=True)
        for f in ["standard", "transformer"]:
            gc = clean[name][f]["gain"]
            print(f"      {f:12s} " + " ".join(
                f"{sc}={(out[sc]['configs'][name][f]['gain']/gc if abs(gc)>1e-9 else float('nan')):+.3f}"
                for sc in SCENARIOS), flush=True)
    json.dump({"T": T, "results": out}, open("robust_v5_screen.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("\n-> robust_v5_screen.json")

if __name__ == "__main__":
    main()