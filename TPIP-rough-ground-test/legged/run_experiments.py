#!/usr/bin/env python3
"""
run_experiments.py — P1~P6 足式机器人平移实测（CPU 微仿真）

三阶段测试：
  阶段A：默认强度下，各方案 × 3 基座模型族 × 3 地形 = 54 组 CI
  阶段B：强度扫描（s∈0~1），找每方案最优强度
  阶段C：通用架构验证——用阶段B在 p2p 上找到的最优 s，
         不调参直接套到 standard/transformer 上，看是否一致正向
"""

from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
import plugins as P

STRENGTHS = np.linspace(0.0, 1.0, 11)   # 0, 0.1, 0.2, ..., 1.0
FAMILIES = ["p2p", "standard", "transformer"]
MODES = ["P1_语义回响", "P2.5_潮汐", "P3_锚点回响",
         "P4_KV共振", "P5_超融合", "P6_情感导演"]
TERRAINS = {"flat(平地)": 0.05, "rough(3cm凸起)": 0.30, "very_rough(10cm碎石)": 0.70}


class Scaled:
    """通用强度包装器：a_eff = a + s*(a_plugin - a)。s=0 → 与裸一致。"""
    def __init__(self, plugin, s):
        self.p = plugin
        self.s = s
    def inject(self, a, **kw):
        base = a.copy()
        ap = self.p.inject(base, **kw)
        return base + self.s * (ap - base)


def eval_ci(mode, family, seed=42, terrain=0.3, strength=None):
    make = P.PLUGINS[mode]
    plug = make()
    if plug is not None and strength is not None:
        plug = Scaled(plug, strength)
    base = BasePolicy(n_joints=6, family=family, seed=seed)
    sim = LeggedMicroSim(base, plugins=[plug] if plug else [], seed=seed)
    traj = sim.run(goal=3.0, terrain=terrain)
    q, dq = traj["q"], traj["dq"]
    qd = central_diff(dq, sim.dt)
    return compute_coherence(q, dq, qd, dt=sim.dt)["coherence_index"]


def opt_rate(ci_plug, ci_bare):
    return (ci_plug - ci_bare) / (1.0 - ci_bare + 1e-9)


# ──────────────────────────────────────────────────────────────────── #
#  阶段 A：默认强度全矩阵                                          #
# ──────────────────────────────────────────────────────────────────── #
def phase_a():
    print("=" * 80)
    print("阶段A：默认强度全矩阵（3 基座 × 6 方案 × 3 地形 = 54 组）")
    print("=" * 80)
    R = {}
    for tname, terr in TERRAINS.items():
        print(f"\n  地形: {tname}")
        for fam in FAMILIES:
            bare = eval_ci("P1_语义回响", fam, terrain=terr, strength=0.0)  # s=0=裸
            R.setdefault(fam, {})[tname] = {"bare": float(bare)}
            row = f"    {fam:12s} bare_CI={bare:.4f}"
            for m in MODES:
                ci = eval_ci(m, fam, terrain=terr)
                R[fam][tname][m] = float(ci)
                o = opt_rate(ci, bare)
                tag = f"{o:+.3f}" if o >= 0 else f"{o:+.3f}"
                row += f"  {m[:6]}={ci:.3f}({tag})"
            print(row)
    return R


# ──────────────────────────────────────────────────────────────────── #
#  阶段 B：强度扫描，找每方案最优 s                              #
# ──────────────────────────────────────────────────────────────────── #
def phase_b():
    print("\n" + "=" * 80)
    print("阶段B：强度扫描（s=0→1, step=0.1），在 p2p/rough 上找最优强度")
    print("=" * 80)
    best = {}
    print(f"\n  {'方案':14s}  最佳s   最佳CI   裸CI    优化率    增幅%")
    print("  " + "-" * 60)
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
        print(f"  {m:14s}  {best_s:.1f}   {best_ci:.4f}  {bare:.4f}  {o:+.4f}  {pct:+.1f}%")
    return best


# ──────────────────────────────────────────────────────────────────── #
#  阶段 C：通用架构验证——用 p2p 上最优 s 直接套到其他基座           #
# ──────────────────────────────────────────────────────────────────── #
def phase_c(best):
    print("\n" + "=" * 80)
    print("阶段C：通用架构验证（用 p2p/rough 上找到的最优 s，不调参直套其他基座）")
    print("=" * 80)
    print(f"\n  {'方案':14s}  最优s   p2p     standard  transformer   一致正向?")
    print("  " + "-" * 65)
    universal = {}
    for m in MODES:
        s = best[m]["s"]
        ci_p2p = eval_ci(m, "p2p", terrain=0.3, strength=s)
        ci_std = eval_ci(m, "standard", terrain=0.3, strength=s)
        ci_tr  = eval_ci(m, "transformer", terrain=0.3, strength=s)
        bare_p2p = eval_ci(m, "p2p", terrain=0.3, strength=0.0)
        bare_std = eval_ci(m, "standard", terrain=0.3, strength=0.0)
        bare_tr  = eval_ci(m, "transformer", terrain=0.3, strength=0.0)
        o_p2p = opt_rate(ci_p2p, bare_p2p)
        o_std = opt_rate(ci_std, bare_std)
        o_tr  = opt_rate(ci_tr,  bare_tr)
        consistent = bool((o_p2p > 0) and (o_std > 0) and (o_tr > 0))
        flag = "YES ✓" if consistent else "NO ✗"
        universal[m] = {
            "s": float(s),
            "p2p": {"ci": float(ci_p2p), "opt": float(o_p2p)},
            "standard": {"ci": float(ci_std), "opt": float(o_std)},
            "transformer": {"ci": float(ci_tr), "opt": float(o_tr)},
            "universal": consistent,
        }
        print(f"  {m:14s}  {s:.1f}   {o_p2p:+.3f}  {o_std:+.3f}    {o_tr:+.3f}      {flag}")
    return universal


def main():
    R_a = phase_a()
    R_b = phase_b()
    R_c = phase_c(R_b)

    # 最终汇总
    print("\n" + "=" * 80)
    print("最终汇总")
    print("=" * 80)
    n_universal = sum(1 for v in R_c.values() if v["universal"])
    print(f"\n  通用架构（同一 s 在 3 基座上一致正向）: {n_universal}/{len(MODES)} 方案")

    print(f"\n  各方案最优表现（p2p/rough 上的最优强度）:")
    print(f"  {'方案':14s}  优化率    增幅%     通用?")
    for m in MODES:
        b = R_b[m]
        u = R_c[m]["universal"]
        print(f"  {m:14s}  {b['opt_rate']:+.4f}   {b['gain_pct']:+.1f}%    {'是' if u else '否'}")

    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump({"phase_a": R_a, "phase_b": R_b, "phase_c": R_c},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  完整结果已写入 {out}")


if __name__ == "__main__":
    main()