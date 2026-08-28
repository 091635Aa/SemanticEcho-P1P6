#!/usr/bin/env python3
"""robust_v6_validate.py — 最终多seed验证 V6(gammax=3,sh_cut=0.5) vs champ(BestCombo+ULI)。
用法: python3 robust_v6_validate.py 42,7,99            # 逗号分隔 seeds，可并行分片跑"""
import sys, os, json, time, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from learnable_bypass import ULI
from verify_architecture import BestCombo
from plugins_robust import RobustComboV5, RobustComboV6
from plugins import BasePlugin

T = 25600
FAMS = ["standard", "transformer", "p2p"]
SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [42]

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

def main():
    print(f"训练 ULI [seeds={SEEDS}] ...", flush=True)
    plug0 = train_uli()
    p0 = copy.deepcopy(plug0)
    configs = {
        "champ": (lambda: RobustComboV5(BestCombo(), copy.deepcopy(p0), g_max=0.0)),
        "v6g3s5": (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=3.0, thr_hf=0.06, lam=0.6, ema_win=0.980, sh_cut=0.5)),
    }
    out = {}
    for scen, noise in SCENARIOS.items():
        t0 = time.time()
        res = {"scenario": scen, "T": T, "configs": {}}
        for name, make in configs.items():
            agg = {}
            for fam in FAMS:
                gains, opts = [], []
                for s in SEEDS:
                    cb, cp = run_plugin_result(BasePolicy(n_joints=6, family=fam, seed=s), s, noise, make)
                    gains.append(cp - cb); opts.append(opt(cp, cb))
                agg[fam] = {"gain_mean": round(float(np.mean(gains)), 4),
                            "gain_seed": round(float(np.std(gains)), 4),
                            "gain": [round(float(g), 4) for g in gains],
                            "opt_mean": round(float(np.mean(opts)), 4),
                            "opt": [round(float(x), 4) for x in opts]}
            res["configs"][name] = agg
        res["time_s"] = round(time.time() - t0)
        out[scen] = res
        print(f"\n[{scen}] {res['time_s']}s", flush=True)
        for name, agg in res["configs"].items():
            st = np.mean([agg[f]["gain_mean"] for f in ["standard", "transformer"]])
            tot = np.mean([agg[f]["gain_mean"] for f in FAMS])
            print(f"  {name:8s} st_gain={st:+.4f} total={tot:+.4f}   "
                  + " ".join(f"{f[:2]}:{agg[f]['gain_mean']:+.3f}(±{agg[f]['gain_seed']:.3f})" for f in FAMS),
                  flush=True)
    # 平滑族 retain 对照
    print("\n[平滑族 retain = gain_scen/gain_clean] (champ vs v6g3s5)")
    for name in configs:
        print(f"\n  {name}:", flush=True)
        for f in ["standard", "transformer"]:
            gc = out["clean"]["configs"][name][f]["gain_mean"]
            print(f"      {f:12s} " + " ".join(
                f"{sc}={(out[sc]['configs'][name][f]['gain_mean']/gc if abs(gc)>1e-9 else float('nan')):+.3f}"
                for sc in SCENARIOS), flush=True)
    tag = ",".join(str(s) for s in SEEDS)
    json.dump({"T": T, "seeds": SEEDS, "results": out},
              open(f"robust_v6_validate_s{tag}.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n-> robust_v6_validate_s{tag}.json")

if __name__ == "__main__":
    main()