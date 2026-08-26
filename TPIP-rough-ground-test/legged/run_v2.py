#!/usr/bin/env python3
"""
run_v2.py — 改进版插件(V1~V6) 实测：验证自适应注入是否实现通用性
"""

from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
import plugins_v2 as PV2

STRENGTHS = np.linspace(0.0, 1.0, 11)
FAMILIES = ["p2p", "standard", "transformer"]
MODES = list(PV2.PLUGINS_V2.keys())
TERRAINS = {"flat": 0.05, "rough": 0.30, "very_rough": 0.70}


class Scaled:
    def __init__(self, plugin, s):
        self.p, self.s = plugin, s
    def inject(self, a, **kw):
        base = a.copy()
        ap = self.p.inject(base, **kw)
        return base + self.s * (ap - base)


def eval_ci(mode, family, seed=42, terrain=0.3, strength=None):
    make = PV2.PLUGINS_V2[mode]
    plug = make()
    if plug is not None and strength is not None:
        plug = Scaled(plug, strength)
    base = BasePolicy(n_joints=6, family=family, seed=seed)
    sim = LeggedMicroSim(base, plugins=[plug] if plug else [], seed=seed)
    traj = sim.run(goal=3.0, terrain=terrain)
    q, dq = traj["q"], traj["dq"]
    qd = central_diff(dq, sim.dt)
    return compute_coherence(q, dq, qd, dt=sim.dt)["coherence_index"]


def opt_rate(ci_p, ci_b):
    return (ci_p - ci_b) / (1.0 - ci_b + 1e-9)


def main():
    # 阶段A：默认强度全矩阵
    print("=" * 80)
    print("改进版 V1~V6 实测 · 阶段A：默认强度全矩阵")
    print("=" * 80)
    R = {}
    for tname, terr in TERRAINS.items():
        print(f"\n  地形: {tname}")
        for fam in FAMILIES:
            bare = eval_ci("V1_AdaptiveEcho", fam, terrain=terr, strength=0.0)
            R.setdefault(fam, {})[tname] = {"bare": float(bare)}
            row = f"    {fam:12s} bare={bare:.4f}"
            for m in MODES:
                ci = eval_ci(m, fam, terrain=terr)
                R[fam][tname][m] = float(ci)
                o = opt_rate(ci, bare)
                row += f"  {m[:8]}={ci:.3f}({o:+.3f})"
            print(row)

    # 阶段B：强度扫描
    print("\n" + "=" * 80)
    print("阶段B：强度扫描（p2p/rough）")
    print("=" * 80)
    best = {}
    print(f"\n  {'方案':20s}  最佳s  最优CI  裸CI   优化率   增幅%")
    print("  " + "-" * 55)
    for m in MODES:
        bare = eval_ci(m, "p2p", terrain=0.3, strength=0.0)
        best_s, best_ci = 0.0, bare
        for s in STRENGTHS:
            ci = eval_ci(m, "p2p", terrain=0.3, strength=float(s))
            if ci > best_ci:
                best_s, best_ci = float(s), ci
        o = opt_rate(best_ci, bare)
        pct = (best_ci - bare) / (bare + 1e-9) * 100
        best[m] = {"s": float(best_s), "ci": float(best_ci), "bare": float(bare),
                   "opt_rate": float(o), "gain_pct": float(pct)}
        print(f"  {m:20s}  {best_s:.1f}   {best_ci:.4f}  {bare:.4f}  {o:+.4f}  {pct:+.1f}%")

    # 阶段C：通用架构验证
    print("\n" + "=" * 80)
    print("阶段C：通用架构验证（p2p最优s 直套 3 基座）")
    print("=" * 80)
    print(f"\n  {'方案':20s}  s    p2p     std      trans    通用?")
    print("  " + "-" * 60)
    universal = {}
    n_pass = 0
    for m in MODES:
        s = best[m]["s"]
        vals = {}
        for fam in FAMILIES:
            ci = eval_ci(m, fam, terrain=0.3, strength=s)
            bare = eval_ci(m, fam, terrain=0.3, strength=0.0)
            vals[fam] = {"ci": float(ci), "opt": float(opt_rate(ci, bare))}
        ok = bool(all(vals[f]["opt"] > 0 for f in FAMILIES))
        universal[m] = {**vals, "s": float(s), "universal": ok}
        if ok:
            n_pass += 1
        flag = "YES" if ok else "NO"
        print(f"  {m:20s}  {s:.1f}  {vals['p2p']['opt']:+.3f}  {vals['standard']['opt']:+.3f}  {vals['transformer']['opt']:+.3f}  {flag}")

    # 最终汇总
    print("\n" + "=" * 80)
    print("最终汇总")
    print("=" * 80)
    print(f"\n  通用架构通过: {n_pass}/{len(MODES)}")

    # 与 V1 原版对比
    print(f"\n  {'方案':20s}  p2p优化率  std优化率  trans优化率  通用?")
    print("  " + "-" * 65)
    for m in MODES:
        v = universal[m]
        flag = "是" if v["universal"] else "否"
        print(f"  {m:20s}  {v['p2p']['opt']:+.4f}   {v['standard']['opt']:+.4f}   {v['transformer']['opt']:+.4f}    {flag}")

    out = os.path.join(os.path.dirname(__file__), "results_v2.json")
    with open(out, "w") as f:
        json.dump({"phase_a": R, "phase_b": best, "phase_c": universal},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  结果已写入 {out}")


if __name__ == "__main__":
    main()