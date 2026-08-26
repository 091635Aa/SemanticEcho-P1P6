#!/usr/bin/env python3
"""
stress_test17.py — 第十七轮：超长仿真 + 宽门保护 P2P

发现（stress_test16）：
- T=1600 突破 ceiling！std=+0.2955 trans=+0.4077 avg=+0.3516
- 但 p2p=-0.0677 (长仿真暴露 p2p 不稳定)

新尝试：
  1. T=3200 - 趋势估计更准
  2. T=1600 宽门(j_hi=2.0) 保护 p2p
  3. T=1600 级联5/7/9 扫参
  4. T=1600 + 双门蓝图
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion

SEEDS = [42, 137, 2024, 7777, 314159]


def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)


def eval_plugin(make_fn, family, seeds=SEEDS, T=800, **sim_kw):
    opts = []
    for seed in seeds:
        base = BasePolicy(n_joints=6, family=family, seed=seed)
        sim_b = LeggedMicroSim(base, plugins=[], seed=seed, T=T)
        t = sim_b.run(goal=3.0, terrain=0.3)
        ci_b = compute_coherence(t['q'], t['dq'],
                                 central_diff(t['dq'], sim_b.dt),
                                 dt=sim_b.dt)['coherence_index']
        plug = make_fn()
        sim_p = LeggedMicroSim(base, plugins=[plug], seed=seed, T=T)
        t2 = sim_p.run(goal=3.0, terrain=0.3)
        ci_p = compute_coherence(t2['q'], t2['dq'],
                                 central_diff(t2['dq'], sim_p.dt),
                                 dt=sim_p.dt)['coherence_index']
        opts.append(opt(ci_p, ci_b))
    return float(np.mean(opts)), float(np.std(opts))


def run_round(round_num, name, make_fn, T=800):
    results = {}
    for fam in ['p2p', 'standard', 'transformer']:
        m, s = eval_plugin(make_fn, fam, T=T)
        results[fam] = {'mean': m, 'std': s}
    uni = all(results[f]['mean'] > 0 for f in ['p2p', 'standard', 'transformer'])
    results['universal'] = uni
    tag = "OK" if uni else "--"
    avg = (results['standard']['mean'] + results['transformer']['mean']) / 2
    print(f"  R{round_num}: {name:50s}  std={results['standard']['mean']:+.4f}  trans={results['transformer']['mean']:+.4f}  avg={avg:+.4f}  p2p={results['p2p']['mean']:+.4f}  {tag}")
    return results


def main():
    print("=" * 120)
    print("第十七轮：超长仿真(T=3200) + 宽门保护P2P")
    print("=" * 120)
    all_r = {}

    # ── 对照：T=800 级联7宽门 ──
    wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}
    class RefCascade(BasePlugin):
        def __init__(self):
            self.passes = [GoldilocksFusion(**wide) for _ in range(7)]
        def inject(self, a, **kw):
            for p in self.passes:
                a = p.inject(a, **kw)
            return a
    all_r['R0_cascade7_T800_ref'] = run_round(0, "级联7宽门 T=800(对照)",
        lambda: RefCascade(), T=800)

    # ── T=1600 级联7宽门（确认上轮突破）──
    all_r['R1_cascade7_T1600'] = run_round(1, "级联7宽门 T=1600",
        lambda: RefCascade(), T=1600)

    # ── T=3200 级联7宽门 ──
    all_r['R2_cascade7_T3200'] = run_round(2, "级联7宽门 T=3200",
        lambda: RefCascade(), T=3200)

    # ── T=1600 宽门保护 P2P (j_hi=2.0, 3.0, 5.0) ──
    for j_hi in [1.0, 2.0, 3.0, 5.0]:
        cfg = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': j_hi}
        class C(BasePlugin):
            def __init__(self, cfg=cfg):
                self.passes = [GoldilocksFusion(**cfg) for _ in range(7)]
            def inject(self, a, **kw):
                for p in self.passes:
                    a = p.inject(a, **kw)
                return a
        all_r[f'R3_cascade7_T1600_jhi{j_hi}'] = run_round(3,
            f"级联7 T=1600 j_hi={j_hi}",
            lambda c=cfg: C(cfg=c), T=1600)

    # ── T=1600 极宽门 + 高 j_peak ──
    for j_peak, j_hi in [(0.3, 2.0), (0.5, 3.0), (0.8, 5.0), (1.0, 10.0)]:
        cfg = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        class C2(BasePlugin):
            def __init__(self, cfg=cfg):
                self.passes = [GoldilocksFusion(**cfg) for _ in range(7)]
            def inject(self, a, **kw):
                for p in self.passes:
                    a = p.inject(a, **kw)
                return a
        all_r[f'R4_cascade7_T1600_pk{j_peak}_hi{j_hi}'] = run_round(4,
            f"级联7 T=1600 j_pk={j_peak} j_hi={j_hi}",
            lambda c=cfg: C2(cfg=c), T=1600)

    # ── T=1600 级联5 ──
    class Cascade5(BasePlugin):
        def __init__(self, cfg=wide):
            self.passes = [GoldilocksFusion(**cfg) for _ in range(5)]
        def inject(self, a, **kw):
            for p in self.passes:
                a = p.inject(a, **kw)
            return a
    all_r['R5_cascade5_T1600'] = run_round(5, "级联5宽门 T=1600",
        lambda: Cascade5(), T=1600)

    # ── T=1600 级联9 ──
    class Cascade9(BasePlugin):
        def __init__(self, cfg=wide):
            self.passes = [GoldilocksFusion(**cfg) for _ in range(9)]
        def inject(self, a, **kw):
            for p in self.passes:
                a = p.inject(a, **kw)
            return a
    all_r['R6_cascade9_T1600'] = run_round(6, "级联9宽门 T=1600",
        lambda: Cascade9(), T=1600)

    # ── T=3200 宽门保护 P2P ──
    cfg_wide2 = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.5, 'j_hi': 3.0}
    class C3(BasePlugin):
        def __init__(self):
            self.passes = [GoldilocksFusion(**cfg_wide2) for _ in range(7)]
        def inject(self, a, **kw):
            for p in self.passes:
                a = p.inject(a, **kw)
            return a
    all_r['R7_cascade7_T3200_wide2'] = run_round(7, "级联7 T=3200 极宽门(pk=0.5 hi=3.0)",
        lambda: C3(), T=3200)

    # ── T=3200 级联5 ──
    all_r['R8_cascade5_T3200'] = run_round(8, "级联5宽门 T=3200",
        lambda: Cascade5(), T=3200)

    # ── 汇总 ──
    print(f"\n{'='*120}")
    print("汇总（按 std+trans 均值排序）")
    print(f"{'='*120}")
    sorted_r = sorted(all_r.items(),
                      key=lambda x: (x[1]['standard']['mean'] + x[1]['transformer']['mean']) / 2,
                      reverse=True)
    print(f"  {'配置':40s}  {'std':>8s}  {'trans':>8s}  {'avg':>8s}  {'p2p':>8s}  通用?")
    print("  " + "-" * 90)
    for k, v in sorted_r:
        uni = v.get('universal', False)
        avg = (v['standard']['mean'] + v['transformer']['mean']) / 2
        print(f"  {k:40s}  {v['standard']['mean']:+.4f}  {v['transformer']['mean']:+.4f}  {avg:+.4f}  {v['p2p']['mean']:+.4f}  {'YES' if uni else 'NO'}")

    best = sorted_r[0]
    print(f"\n  最优: {best[0]} → avg={(best[1]['standard']['mean']+best[1]['transformer']['mean'])/2:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "stress_test17.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
