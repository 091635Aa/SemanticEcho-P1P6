#!/usr/bin/env python3
"""robust_v8_screen.py — 验证『提高 thr_hf + 加陡 p』能否消除 clean 惩罚并保留噪声增益。
核心发现(hf_probe)：hf 在 clean/mild/strong = 0.20/0.22/0.29，旧 thr_hf=0.06 恒饱和→clean 也常开→clean 惩罚。
若 thr_hf∈[0.25,0.32] 且 p 陡化，则 clean gate≈低(≈champ)、strong gate≈高(保留噪声增益)。
用法: python3 robust_v8_screen.py [T] [seed]"""
import sys, os, json, time, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from learnable_bypass import ULI
from verify_architecture import BestCombo
from plugins_robust import RobustComboV5, RobustComboV6

T = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 99
FAMS = ["standard", "transformer", "p2p"]

def opt(cp, cb): return (cp - cb) / (1 - cb + 1e-9)
def run(base, seed, noise, make):
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
    plug.fit_from(data); plug.strength = 0.5; plug._amp = min(plug._amp, 0.5)
    return plug

SCEN = {
    "clean":  {"obs_noise":0.0,"act_noise":0.0,"obs_spike_p":0.0,"obs_spike_amp":0.0,"act_spike_p":0.0,"act_spike_amp":0.0},
    "mild":   {"obs_noise":0.02,"act_noise":0.01,"obs_spike_p":0.0,"obs_spike_amp":0.0,"act_spike_p":0.0,"act_spike_amp":0.0},
    "strong": {"obs_noise":0.05,"act_noise":0.02,"obs_spike_p":0.002,"obs_spike_amp":0.4,"act_spike_p":0.002,"act_spike_amp":0.4},
}

def main():
    print(f"训练 ULI seed-based [T={T}, seed={SEED}]", flush=True)
    plug0 = train_uli(); p0 = copy.deepcopy(plug0)
    cfgs = {
        "champ":     (lambda: RobustComboV5(BestCombo(), copy.deepcopy(p0), g_max=0.0)),
        "v6g3s5(t06)": (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=3.0, thr_hf=0.06, p=1.0, sh_cut=0.5)),
        "v6g3-thr25":   (lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=3.0, thr_hf=0.25, p=3.0, sh_cut=0.5)),
        "v6g3-thr28-p3":(lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=3.0, thr_hf=0.28, p=3.0, sh_cut=0.5)),
        "v6g3-thr30-p4":(lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=3.0, thr_hf=0.30, p=4.0, sh_cut=0.5)),
        "v6g4-thr28-p3":(lambda: RobustComboV6(BestCombo(), copy.deepcopy(p0), gammax=4.0, thr_hf=0.28, p=3.0, sh_cut=0.5)),
    }
    out = {}
    for scen, noise in SCEN.items():
        t0 = time.time()
        res = {"scenario": scen, "configs": {}}
        for name, make in cfgs.items():
            agg = {}
            for fam in FAMS:
                cb, cp = run(BasePolicy(n_joints=6, family=fam, seed=SEED), SEED, noise, make)
                agg[fam] = {"gain": round(float(cp - cb), 4), "opt": round(float(opt(cp, cb)), 4)}
            res["configs"][name] = agg
        out[scen] = res
        print(f"\n[{scen}] {round(time.time()-t0)}s", flush=True)
        for name, agg in res["configs"].items():
            st = (agg["standard"]["gain"] + agg["transformer"]["gain"]) / 2
            tot = (agg["standard"]["gain"] + agg["transformer"]["gain"] + agg["p2p"]["gain"]) / 3
            print(f"  {name:16s} st_gain={st:+.4f} total={tot:+.4f}  st:{agg['standard']['gain']:+.3f} tr:{agg['transformer']['gain']:+.3f} p2:{agg['p2p']['gain']:+.3f}", flush=True)
    json.dump({"T": T, "seed": SEED, "results": out}, open(f"robust_v8_screen_s{SEED}_T{T}.json", "w"), ensure_ascii=False, indent=1, default=float)

if __name__ == "__main__":
    main()