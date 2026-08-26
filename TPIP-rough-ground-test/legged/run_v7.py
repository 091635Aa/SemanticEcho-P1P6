#!/usr/bin/env python3
"""
run_v7.py — V7 JerkAdaptiveFusion 跨任务验证（足式 + 臂操作）
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from arm_env import ArmBasePolicy, ArmMicroSim
import plugins_v2 as PV2

SEEDS = [42, 137, 2024, 7777, 314159]

class Scaled:
    def __init__(self, p, s): self.p, self.s = p, s
    def inject(self, a, **kw):
        base = a.copy()
        ap = self.p.inject(base, **kw)
        return base + self.s * (ap - base)

def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)

def legged_ci(fam, plug, seed):
    base = BasePolicy(n_joints=6, family=fam, seed=seed)
    sim = LeggedMicroSim(base, plugins=[plug] if plug else [], seed=seed)
    t = sim.run(goal=3.0, terrain=0.3)
    return compute_coherence(t['q'], t['dq'], central_diff(t['dq'], sim.dt), dt=sim.dt)['coherence_index']

def arm_ci(fam, plug, seed):
    base = ArmBasePolicy(n_joints=6, family=fam, seed=seed)
    sim = ArmMicroSim(base, plugins=[plug] if plug else [], seed=seed)
    t = sim.run()
    return compute_coherence(t['q'], t['dq'], central_diff(t['dq'], sim.dt), dt=sim.dt)['coherence_index']

def test_plugin(name, make):
    """在足式(3基座) + 臂操作(3基座) × 5种子上测试通用性。"""
    results = {}
    all_positive = True
    print(f"\n  {name}:")
    for task, families, ci_fn in [
        ("legged", ["p2p", "standard", "transformer"], legged_ci),
        ("arm", ["p2p_arm", "pid_arm", "learned"], arm_ci),
    ]:
        for fam in families:
            opts = []
            for seed in SEEDS:
                ci_b = ci_fn(fam, None, seed)
                plug = Scaled(make(), 1.0) if make else None
                ci_p = ci_fn(fam, plug, seed)
                opts.append(opt(ci_p, ci_b))
            m, s = np.mean(opts), np.std(opts)
            ok = m > 0
            if not ok: all_positive = False
            results[f"{task}/{fam}"] = {"mean": float(m), "std": float(s), "positive": bool(ok)}
            print(f"    {task:6s}/{fam:12s}  opt={m:+.4f}±{s:.4f}  {'OK' if ok else 'FAIL'}")
    results["universal"] = bool(all_positive)
    print(f"    → 通用: {'YES' if all_positive else 'NO'}")
    return results

def main():
    print("=" * 80)
    print("V7 JerkAdaptiveFusion 跨任务通用性验证")
    print("=" * 80)
    candidates = [
        ("V5_SmartFusion", lambda: PV2.SmartFusion()),
        ("V7_JerkAdaptiveFusion", lambda: PV2.JerkAdaptiveFusion()),
    ]
    all_results = {}
    for name, make in candidates:
        all_results[name] = test_plugin(name, make)

    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    for name in all_results:
        r = all_results[name]
        print(f"  {name:30s}  通用: {'YES' if r['universal'] else 'NO'}")

    import json
    out = os.path.join(os.path.dirname(__file__), "results_v7.json")
    with open(out, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已写入 {out}")

if __name__ == "__main__":
    main()