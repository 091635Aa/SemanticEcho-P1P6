#!/usr/bin/env python3
"""
stress_test14.py — 第十四轮：幅度匹配蓝图 + 集成级联

发现：env 提供的 blueprint 幅度=0.25，但实际步态幅度 0.35(std)/0.40(trans)。
直接注入蓝图会 dampen 实际步态 → 反而下降。

新机制：
  V38 ScaledBlueprint - 蓝图 × 1.5 (匹配实际幅度)
  V39 AdaptiveScaledBlueprint - 在线估计实际幅度，自适应缩放蓝图
  V40 EnsembleCascade - 多个不同参数的级联并行，取中值
  V41 CascadePlusScaledBlueprint - 级联7 + 缩放蓝图精修
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion
from plugins_v2 import AdaptiveEcho, GatedTidal, AdaptiveKV

SEEDS = [42, 137, 2024, 7777, 314159]


def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)


def eval_plugin(make_fn, family, seeds=SEEDS):
    opts = []
    for seed in seeds:
        base = BasePolicy(n_joints=6, family=family, seed=seed)
        sim_b = LeggedMicroSim(base, plugins=[], seed=seed)
        t = sim_b.run(goal=3.0, terrain=0.3)
        ci_b = compute_coherence(t['q'], t['dq'],
                                 central_diff(t['dq'], sim_b.dt),
                                 dt=sim_b.dt)['coherence_index']
        plug = make_fn()
        sim_p = LeggedMicroSim(base, plugins=[plug], seed=seed)
        t2 = sim_p.run(goal=3.0, terrain=0.3)
        ci_p = compute_coherence(t2['q'], t2['dq'],
                                 central_diff(t2['dq'], sim_p.dt),
                                 dt=sim_p.dt)['coherence_index']
        opts.append(opt(ci_p, ci_b))
    return float(np.mean(opts)), float(np.std(opts))


def run_round(round_num, name, make_fn):
    results = {}
    for fam in ['p2p', 'standard', 'transformer']:
        m, s = eval_plugin(make_fn, fam)
        results[fam] = {'mean': m, 'std': s}
    uni = all(results[f]['mean'] > 0 for f in ['p2p', 'standard', 'transformer'])
    results['universal'] = uni
    tag = "OK" if uni else "--"
    avg = (results['standard']['mean'] + results['transformer']['mean']) / 2
    print(f"  R{round_num}: {name:50s}  std={results['standard']['mean']:+.4f}  trans={results['transformer']['mean']:+.4f}  avg={avg:+.4f}  p2p={results['p2p']['mean']:+.4f}  {tag}")
    return results


def measure_jerk(obj, a):
    if obj._last_a is None:
        obj._last_a = a.copy()
        return 0.0
    j = float(np.linalg.norm(a - obj._last_a))
    obj._last_a = a.copy()
    obj._jerk_hist.append(j)
    if len(obj._jerk_hist) > 20:
        obj._jerk_hist.pop(0)
    if len(obj._jerk_hist) < 3:
        return 0.0
    return float(np.mean(obj._jerk_hist))


def goldilocks_gate(j, j_lo, j_peak, j_hi):
    if j < j_lo or j > j_hi:
        return 0.0
    if j <= j_peak:
        t = (j - j_lo) / (j_peak - j_lo + 1e-9)
        return float(np.sin(t * np.pi / 2))
    else:
        t = (j - j_peak) / (j_hi - j_peak + 1e-9)
        return float(np.cos(t * np.pi / 2))


# ──────────────────────────────────────────────────────────────────── #
#  V38: 缩放蓝图 (ScaledBlueprint)                                     #
#  蓝图 × scale_factor (1.5 = std 1.4 / trans 1.6 的折中)             #
# ──────────────────────────────────────────────────────────────────── #
class ScaledBlueprint(BasePlugin):
    def __init__(self, scale=1.5, strength=1.0,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        self.scale = scale
        self.strength = strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._jerk_hist = []
        self._last_a = None

    def inject(self, a, **kw):
        bp = kw.get('blueprint', None)
        if bp is None:
            return a
        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a
        bp = np.asarray(bp, dtype=float) * self.scale
        return a + self.strength * gate * (bp - a)


# ──────────────────────────────────────────────────────────────────── #
#  V39: 自适应缩放蓝图                                                 #
#  在线估计实际步态幅度，动态调整蓝图缩放                                #
# ──────────────────────────────────────────────────────────────────── #
class AdaptiveScaledBlueprint(BasePlugin):
    def __init__(self, strength=1.0, window=200, dt=0.01,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        self.strength = strength
        self.window = window
        self.dt = dt
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None
        self._scale = 1.5  # 初始

    def _estimate_scale(self, bp):
        if len(self._hist) < 32:
            return
        H = np.array(self._hist)
        # 实际步态幅度 = std(a) * sqrt(2) (正弦的 RMS→amplitude)
        actual_amp = float(np.std(H)) * np.sqrt(2)
        # 蓝图幅度 ≈ 0.25
        bp_amp = 0.25
        if bp_amp > 1e-6:
            new_scale = actual_amp / bp_amp
            # 平滑更新
            self._scale = 0.9 * self._scale + 0.1 * new_scale

    def inject(self, a, **kw):
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)
        bp = kw.get('blueprint', None)
        if bp is None:
            return a
        if len(self._hist) % 16 == 0:
            self._estimate_scale(bp)
        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a
        bp = np.asarray(bp, dtype=float) * self._scale
        return a + self.strength * gate * (bp - a)


# ──────────────────────────────────────────────────────────────────── #
#  V40: 集成级联 (Ensemble Cascade)                                    #
#  多个不同参数的级联并行，取中值                                       #
# ──────────────────────────────────────────────────────────────────── #
class EnsembleCascade(BasePlugin):
    def __init__(self, n_ensembles=3, n_passes=7):
        configs = [
            {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.1, 'j_hi': 0.3},   # 窄门
            {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5},   # 宽门
            {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.20, 'j_hi': 0.8},   # 极宽门
        ][:n_ensembles]
        self.cascades = [[GoldilocksFusion(**c) for _ in range(n_passes)]
                         for c in configs]
        self._jerk_hist = []
        self._last_a = None

    def inject(self, a, **kw):
        outputs = []
        for cascade in self.cascades:
            a_copy = a.copy()
            for p in cascade:
                a_copy = p.inject(a_copy, **kw)
            outputs.append(a_copy)
        return np.median(np.array(outputs), axis=0)


# ──────────────────────────────────────────────────────────────────── #
#  V41: 级联 + 缩放蓝图                                                #
# ──────────────────────────────────────────────────────────────────── #
class CascadePlusScaledBlueprint(BasePlugin):
    def __init__(self, n_passes=7, bp_scale=1.5, bp_strength=0.5,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_passes)]
        self.bp = ScaledBlueprint(scale=bp_scale, strength=bp_strength,
                                    j_lo=0.005, j_peak=j_peak, j_hi=j_hi)
    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.bp.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  V42: 缩放蓝图先行 + 级联                                             #
# ──────────────────────────────────────────────────────────────────── #
class ScaledBlueprintPlusCascade(BasePlugin):
    def __init__(self, bp_scale=1.5, bp_strength=0.5, n_passes=7,
                 j_lo=0.005, j_peak=0.15, j_hi=0.5):
        self.bp = ScaledBlueprint(scale=bp_scale, strength=bp_strength,
                                    j_lo=j_lo, j_peak=j_peak, j_hi=j_hi)
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_passes)]
    def inject(self, a, **kw):
        a = self.bp.inject(a, **kw)
        for p in self.cascade:
            a = p.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  V43: 自适应缩放蓝图 + 级联                                            #
# ──────────────────────────────────────────────────────────────────── #
class AdaptiveScaledPlusCascade(BasePlugin):
    def __init__(self, bp_strength=0.5, n_passes=7,
                 j_lo=0.005, j_peak=0.15, j_hi=0.5):
        self.bp = AdaptiveScaledBlueprint(strength=bp_strength,
                                           j_lo=j_lo, j_peak=j_peak, j_hi=j_hi)
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_passes)]
    def inject(self, a, **kw):
        a = self.bp.inject(a, **kw)
        for p in self.cascade:
            a = p.inject(a, **kw)
        return a


def main():
    print("=" * 120)
    print("第十四轮：幅度匹配蓝图 + 集成级联")
    print("=" * 120)
    all_r = {}
    wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}

    # ── 对照 ──
    class RefCascade(BasePlugin):
        def __init__(self):
            self.passes = [GoldilocksFusion(**wide) for _ in range(7)]
        def inject(self, a, **kw):
            for p in self.passes:
                a = p.inject(a, **kw)
            return a
    all_r['R0_cascade7_wide_ref'] = run_round(0, "级联7宽门(对照)",
        lambda: RefCascade())

    # ── V38 缩放蓝图 单独 ──
    for scale in [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]:
        all_r[f'R1_scaled_bp_{scale}'] = run_round(1, f"缩放蓝图 scale={scale}",
            lambda s=scale: ScaledBlueprint(scale=s, strength=1.0))

    # ── V39 自适应缩放蓝图 单独 ──
    all_r['R2_adaptive_scaled'] = run_round(2, "自适应缩放蓝图",
        lambda: AdaptiveScaledBlueprint(strength=1.0))

    # ── V40 集成级联 ──
    all_r['R3_ensemble3'] = run_round(3, "集成级联(3模型×7趟)",
        lambda: EnsembleCascade(n_ensembles=3, n_passes=7))

    # ── V41 级联 + 缩放蓝图 ──
    for scale in [1.4, 1.5, 1.6]:
        for strength in [0.3, 0.5, 0.8]:
            all_r[f'R4_cascade7_bp_{scale}_{strength}'] = run_round(4,
                f"级联7 + 缩放蓝图({scale}×{strength})",
                lambda s=scale, st=strength: CascadePlusScaledBlueprint(
                    bp_scale=s, bp_strength=st))

    # ── V42 缩放蓝图先行 + 级联 ──
    for scale in [1.4, 1.5, 1.6]:
        for strength in [0.3, 0.5, 0.8]:
            all_r[f'R5_bp_{scale}_{strength}_cascade7'] = run_round(5,
                f"缩放蓝图({scale}×{strength}) + 级联7",
                lambda s=scale, st=strength: ScaledBlueprintPlusCascade(
                    bp_scale=s, bp_strength=st))

    # ── V43 自适应缩放蓝图 + 级联 ──
    for strength in [0.3, 0.5, 0.8]:
        all_r[f'R6_adapt_bp_{strength}_cascade7'] = run_round(6,
            f"自适应缩放蓝图({strength}) + 级联7",
            lambda st=strength: AdaptiveScaledPlusCascade(bp_strength=st))

    # ── 集成级联 + 缩放蓝图 ──
    class EnsemblePlusScaledBlueprint(BasePlugin):
        def __init__(self, bp_scale=1.5, bp_strength=0.5):
            self.ensemble = EnsembleCascade(n_ensembles=3, n_passes=7)
            self.bp = ScaledBlueprint(scale=bp_scale, strength=bp_strength,
                                        j_lo=0.005, j_peak=0.15, j_hi=0.5)
        def inject(self, a, **kw):
            a = self.ensemble.inject(a, **kw)
            a = self.bp.inject(a, **kw)
            return a
    all_r['R7_ensemble_bp'] = run_round(7, "集成级联 + 缩放蓝图(1.5×0.5)",
        lambda: EnsemblePlusScaledBlueprint())

    # ── 汇总 ──
    print(f"\n{'='*120}")
    print("汇总（按 std+trans 均值排序）")
    print(f"{'='*120}")
    sorted_r = sorted(all_r.items(),
                      key=lambda x: (x[1]['standard']['mean'] + x[1]['transformer']['mean']) / 2,
                      reverse=True)
    print(f"  {'配置':35s}  {'std':>8s}  {'trans':>8s}  {'avg':>8s}  {'p2p':>8s}  通用?")
    print("  " + "-" * 85)
    for k, v in sorted_r:
        uni = v.get('universal', False)
        avg = (v['standard']['mean'] + v['transformer']['mean']) / 2
        print(f"  {k:35s}  {v['standard']['mean']:+.4f}  {v['transformer']['mean']:+.4f}  {avg:+.4f}  {v['p2p']['mean']:+.4f}  {'YES' if uni else 'NO'}")

    best = sorted_r[0]
    print(f"\n  最优: {best[0]} → avg={(best[1]['standard']['mean']+best[1]['transformer']['mean'])/2:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "stress_test14.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
