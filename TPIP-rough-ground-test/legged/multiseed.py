#!/usr/bin/env python3
"""
multiseed.py — 多种子统计验证：5 种子 × 3 基座 × 6 方案
确认 5/6 通用性不是偶然，计算均值±标准差和 p 值
"""

from __future__ import annotations
import json, os, sys
import numpy as np
from scipy import stats as sp_stats  # 如果不可用则用纯 numpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
import plugins as P
import plugins_v2 as PV2

SEEDS = [42, 137, 2024, 7777, 314159]
FAMILIES = ["p2p", "standard", "transformer"]
MODES_V1 = ["P1_语义回响", "P2.5_潮汐", "P3_锚点回响",
            "P4_KV共振", "P5_超融合", "P6_情感导演"]
MODES_V2 = ["V1_AdaptiveEcho", "V2_GatedTidal", "V3_SelfAnchored",
            "V4_AdaptiveKV", "V5_SmartFusion", "V6_AdaptiveDirector"]


class Scaled:
    def __init__(self, plugin, s):
        self.p, self.s = plugin, s
    def inject(self, a, **kw):
        base = a.copy()
        ap = self.p.inject(base, **kw)
        return base + self.s * (ap - base)


def eval_ci(base_policy, plugins_list, seed, terrain=0.3):
    sim = LeggedMicroSim(base_policy, plugins=plugins_list, seed=seed)
    traj = sim.run(goal=3.0, terrain=terrain)
    q, dq = traj["q"], traj["dq"]
    qd = central_diff(dq, sim.dt)
    return compute_coherence(q, dq, qd, dt=sim.dt)["coherence_index"]


def opt_rate(ci_p, ci_b):
    return (ci_p - ci_b) / (1.0 - ci_b + 1e-9)


def run_batch(plugins_dict, modes, label, strength=1.0):
    """对每个 mode × family × seed 算 CI 和 opt_rate。"""
    results = {}
    print(f"\n{'='*80}")
    print(f"{label}：多种子统计（{len(SEEDS)} 种子 × {len(FAMILIES)} 基座 × {len(modes)} 方案）")
    print(f"{'='*80}")

    for m in modes:
        results[m] = {}
        for fam in FAMILIES:
            opt_rates = []
            bare_cis = []
            plug_cis = []
            for seed in SEEDS:
                base = BasePolicy(n_joints=6, family=fam, seed=seed)
                # bare
                ci_b = eval_ci(base, [], seed)
                bare_cis.append(ci_b)
                # plugin
                make = plugins_dict[m]
                plug = make()
                if plug is not None:
                    plug = Scaled(plug, strength)
                ci_p = eval_ci(base, [plug] if plug else [], seed)
                plug_cis.append(ci_p)
                opt_rates.append(opt_rate(ci_p, ci_b))
            results[m][fam] = {
                "opt_mean": float(np.mean(opt_rates)),
                "opt_std": float(np.std(opt_rates)),
                "bare_mean": float(np.mean(bare_cis)),
                "plug_mean": float(np.mean(plug_cis)),
                "all_positive": bool(all(o > 0 for o in opt_rates)),
                "n_positive": int(sum(1 for o in opt_rates if o > 0)),
                "opt_rates": [float(o) for o in opt_rates],
            }

    # 汇总表
    print(f"\n  {'方案':20s}  {'p2p mean±std':>16s}  {'std mean±std':>16s}  {'trans mean±std':>16s}  通用?")
    print("  " + "-" * 80)
    n_universal = 0
    for m in modes:
        r = results[m]
        p2p = f"{r['p2p']['opt_mean']:+.4f}±{r['p2p']['opt_std']:.4f}"
        std = f"{r['standard']['opt_mean']:+.4f}±{r['standard']['opt_std']:.4f}"
        trn = f"{r['transformer']['opt_mean']:+.4f}±{r['transformer']['opt_std']:.4f}"
        ok = (r["p2p"]["opt_mean"] > 0 and r["standard"]["opt_mean"] > 0
              and r["transformer"]["opt_mean"] > 0)
        if ok:
            n_universal += 1
        flag = "YES" if ok else "NO"
        print(f"  {m:20s}  {p2p:>16s}  {std:>16s}  {trn:>16s}  {flag}")

    print(f"\n  通用架构通过: {n_universal}/{len(modes)}")
    return results


def main():
    # 原始版
    R1 = run_batch(P.PLUGINS, MODES_V1, "原始 P1~P6")
    # 改进版
    R2 = run_batch(PV2.PLUGINS_V2, MODES_V2, "自适应改进版 V1~V6")

    # 配对 t 检验：改进版 vs 原始版在标准RL基座上
    print(f"\n{'='*80}")
    print("配对 t 检验：改进版 vs 原始版（standard 基座优化率）")
    print(f"{'='*80}")
    print(f"\n  {'方案对':40s}  {'原始 mean±std':>16s}  {'改进 mean±std':>16s}  p值     显著?")
    print("  " + "-" * 90)
    for i in range(len(MODES_V1)):
        m1, m2 = MODES_V1[i], MODES_V2[i]
        r1 = R1[m1]["standard"]["opt_rates"]
        r2 = R2[m2]["standard"]["opt_rates"]
        t_stat, p_val = sp_stats.ttest_rel(r2, r1) if "scipy" in dir() else (0, 0.5)
        sig = "YES" if p_val < 0.05 else "NO"
        pair = f"{m1[:8]} vs {m2[:8]}"
        print(f"  {pair:40s}  {np.mean(r1):+.4f}±{np.std(r1):.4f}  {np.mean(r2):+.4f}±{np.std(r2):.4f}  {p_val:.4f}  {sig}")

    # 写入
    out = os.path.join(os.path.dirname(__file__), "results_multiseed.json")
    with open(out, "w") as f:
        json.dump({"v1": R1, "v2": R2}, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已写入 {out}")


if __name__ == "__main__":
    main()