#!/usr/bin/env python3
"""
noise_suite.py — 传感器噪声/噪点鲁棒性测试 (第37轮)

非理想环境：观测噪声(σ)、执行噪声(σ)、偶发离群噪点(spike)。
在每档噪声下量测插件优化率、方差、Universal 及相对 clean 的衰减(可信度)。

插件候选：
  - BestCombo        （无参平滑底座）
  - BestCombo+ULI    （底座之上叠加可学习引导，锚定 standard 训练、零样本迁移）
"""
import sys, os, json, time, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from learnable_bypass import ULI
from verify_architecture import BestCombo
from plugins import BasePlugin

T = 51200
SEEDS = [42]
FAMS = ["standard", "transformer", "p2p"]


def opt(cp, cb): return (cp - cb) / (1 - cb + 1e-9)


class Chain(BasePlugin):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def inject(self, a, **kw):
        return self.b.inject(self.a.inject(a, **kw), **kw)


def ci_for(sim, plugin):
    p = copy.deepcopy(plugin) if plugin else None
    s = LeggedMicroSim(sim.base, plugins=[p] if p else [],
                       T=sim.T, dt=sim.dt, seed=sim._seed,
                       obs_noise=sim.obs_noise, act_noise=sim.act_noise,
                       spike_p=sim.spike_p, spike_amp=sim.spike_amp)
    tr = s.run()
    return compute_coherence(tr["q"], tr["dq"], central_diff(tr["dq"], s.dt),
                            dt=s.dt)["coherence_index"]


def train_uli():
    plug = ULI(strength=1.0, W=40, lam=5e-3)
    base = BasePolicy(n_joints=6, family="standard", seed=42)
    data = plug.collect_trajectory(base, seed=42, t_len=60000)
    plug.fit_from(data)
    plug.strength = 0.5
    plug._amp = min(plug._amp, 0.5)
    return plug


def make_sim(base, seed, noise):
    return LeggedMicroSim(base, T=T, dt=0.01, seed=seed,
                          obs_noise=noise["obs"], act_noise=noise["act"],
                          spike_p=noise["spike_p"], spike_amp=noise["spike_amp"])


def eval_scenario(scenario, checks, plug):
    out = {"scenario": scenario, "configs": {}}
    for name, factory in [("BestCombo", (lambda: BestCombo())),
                          ("BestCombo+ULI", (lambda: Chain(BestCombo(), copy.deepcopy(plug))))]:
        per = {}
        for fam in FAMS:
            vals = []
            for s in SEEDS:
                base = BasePolicy(n_joints=6, family=fam, seed=s)
                sim = make_sim(base, s, checks)
                sim._seed = s
                cb = ci_for(sim, None)
                cp = ci_for(sim, factory())
                vals.append(opt(cp, cb))
            per[fam] = vals
        avgs = [(per["standard"][i] + per["transformer"][i]) / 2 for i in range(len(SEEDS))]
        uni = all(float(x) > 0 for fam in FAMS for x in per[fam])
        out["configs"][name] = {"avg": float(np.mean(avgs)),
                                "avg_per_seed": [round(x, 4) for x in avgs],
                                "families": {fam: round(float(np.mean(v)), 4) for fam, v in per.items()},
                                "universal": uni}
    return out


SCENARIOS = {
    "clean":       {"obs": 0.0,  "act": 0.0,   "spike_p": 0.0,   "spike_amp": 0.0},
    "mild_sensor": {"obs": 0.02, "act": 0.01,  "spike_p": 0.0,   "spike_amp": 0.0},
    "strong":      {"obs": 0.05, "act": 0.02,  "spike_p": 0.003, "spike_amp": 0.4},
}


def main():
    print("训练 ULI (锚定 standard) ...", flush=True)
    plug = train_uli()
    print("  W_out", plug.W_out.shape, "amp->0.5", flush=True)

    all_r = {}
    for scen, n in SCENARIOS.items():
        t0 = time.time()
        r = eval_scenario(scen, n, plug)
        r["time_s"] = round(time.time() - t0)
        all_r[scen] = r
        print(f"\n[{scen}] obs={n['obs']} act={n['act']} spike=({n['spike_p']},{n['spike_amp']}) {r['time_s']}s", flush=True)
        for name, rr in r["configs"].items():
            print(f"  {name:16s} avg={rr['avg']:+.4f} "
                  f"s={rr['families']['standard']:+.4f} t={rr['families']['transformer']:+.4f} "
                  f"p={rr['families']['p2p']:+.4f} uni={'Y' if rr['universal'] else 'N'}", flush=True)

    # 鲁棒性（可信度）：相对 clean 的 avg 衰减
    clean = all_r["clean"]["configs"]
    print("\n[可信度: avg 相对 clean 衰减]")
    robust = {}
    for name in clean:
        base_v = clean[name]["avg"]
        drops = {scen: round(all_r[scen]["configs"][name]["avg"] - base_v, 4)
                 for scen in SCENARIOS}
        robust[name] = {"clean_avg": round(base_v, 4), "drops": drops}
        print(f"  {name:16s} clean={base_v:+.4f}  "
              + " ".join(f"{k}={v:+.4f}" for k, v in drops.items()), flush=True)

    json.dump({"T": T, "seeds": SEEDS, "scenarios": {k: v for k, v in SCENARIOS.items()},
               "results": all_r, "robustness": robust},
              open("noise_suite.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("\n-> noise_suite.json")


if __name__ == "__main__":
    main()