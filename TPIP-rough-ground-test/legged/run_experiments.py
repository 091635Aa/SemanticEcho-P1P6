#!/usr/bin/env python3
"""
run_experiments.py — 对 P1~P6（含融合）在 CPU 微仿真足式模型上实测，
输出连贯性指数 (CI) 与相对裸策略的优化率。

用两套指标：
  CI: 复用 metrics/coherence_index.compute_coherence（jerk 平滑度 + 步态相图重合度）
  优化率: opt_rate = (CI_plugin - CI_bare) / (1 - CI_bare)    相对补齐空间
          gain%%   = (CI_plugin - CI_bare) / CI_bare * 100     相对增幅
"""

from __future__ import annotations
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
import plugins as P


class Scaled:
    """通用强度包装器：a_eff = a + s*(a_plugin - a)。s=0 时与裸完全一致。"""
    def __init__(self, plugin, s: float):
        self.p = plugin
        self.s = s

    def inject(self, a, **kw):
        base = a.copy()
        ap = self.p.inject(base, **kw)
        return base + self.s * (ap - base)


def eval_mode(mode: str, family: str, seed: int = 42,
              terrain: float = 0.3, strength: float | None = None) -> float:
    make = P.PLUGINS[mode]
    plug = make()
    if mode == "P1.5_兼容层":
        plug = None                       # 兼容层本身不注入 = 对照
    if plug is not None and strength is not None:
        plug = Scaled(plug, strength)
    base = BasePolicy(n_joints=6, family=family, seed=seed)
    sim = LeggedMicroSim(base, plugins=[plug] if plug else [], seed=seed)
    traj = sim.run(goal=3.0, terrain=terrain)
    q = traj["q"]
    dq = traj["dq"]
    qd = central_diff(dq, sim.dt)
    r = compute_coherence(q, dq, qd, foot_contact=None, dt=sim.dt)
    return r["coherence_index"]


def main():
    families = ["p2p", "standard", "transformer"]
    modes = ["P1_语义回响", "P2.5_潮汐", "P3_锚点回响",
             "P4_KV共振", "P5_超融合", "P6_情感导演"]
    results = {}
    print("=" * 76)
    print("P1~P6 → 足式机器人 实测（CPU 微仿真, terrain=3cm 凸起档）")
    print("=" * 76)
    for fam in families:
        row = {}
        print(f"\n--- 基座模型族: {fam} ---")
        for m in modes:
            ci = eval_mode(m, fam)
            row[m] = ci
            bare = row.get("bare")
            tag = "   <-- 基线(P2P)"
            if bare is not None and m != "bare":
                gain = (ci - bare) / (1 - bare)
                pct = (ci - bare) / (bare + 1e-9) * 100
                tag = f"   opt={gain:+.3f}  gain={pct:+.1f}%"
            print(f"  {m:14s}  CI={ci:.4f}{tag}")
        results[fam] = row

    # 汇总
    print("\n" + "=" * 76)
    print("汇总表（CI by 基座模型族 × 方案）")
    hdr = f"{'方案':14s}" + "".join(f"{f[:5]:>12s}" for f in families)
    print(hdr)
    for m in modes:
        line = f"{m:14s}"
        for fam in families:
            line += f"{results[fam][m]:>12.4f}"
        print(line)

    # 优化率汇总
    print("\n优化率 opt_rate=(CI_plug-CI_bare)/(1-CI_bare)，纯方案 vs 裸")
    hdr = f"{'方案':14s}" + "".join(f"{f[:5]:>12s}" for f in families)
    print(hdr)
    for m in modes:
        if m == "bare":
            continue
        line = f"{m:14s}"
        for fam in families:
            bare = results[fam]["bare"]
            ci = results[fam][m]
            opt = (ci - bare) / (1 - bare)
            line += f"{opt:>12.3f}"
        print(line)

    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()