#!/usr/bin/env python3
"""
stress_test15.py — 第十五轮：双门控蓝图 + 族自适应

发现（stress_test14）：
- 级联7 + 缩放蓝图(1.6×0.8) → std=+0.2833(↑) 但 trans=+0.3651(↓)
- 原因：transformer 的 jerk 很低(0.014)，蓝图仍微弱注入(有效强度0.078)，干扰了已有的纯净步态

新机制：
  V44 DualGateBlueprint - 级联用宽门，蓝图用窄门(j_lo=0.05, j_peak=0.12)
                         → 只对 standard 之类中等 jerk 激活，transformer 不激活
  V45 FamilyAdaptiveGate - 在线估计"基座族"，自适应选择门控参数
  V46 CascadePlusDualGate - 级联7宽门 + 双门控蓝图
  V47 CascadePlusNarrowBp - 扫参窄门蓝图
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
#  V44: 双门控缩放蓝图                                                  #
#  使用窄门 (j_lo=0.05, j_peak=0.12, j_hi=0.3)                          #
#  → 对 standard (jerk~0.12) 全力，对 transformer (jerk~0.014) 关闭       #
# ──────────────────────────────────────────────────────────────────── #
class DualGateScaledBlueprint(BasePlugin):
    def __init__(self, scale=1.6, strength=0.8,
                 j_lo=0.05, j_peak=0.12, j_hi=0.3):
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
#  V45: 族自适应门控                                                   #
#  在线检测 jerk 水平，分类为 low/mid/high                              #
#  low (trans): 关闭蓝图，仅级联                                         #
#  mid (std): 全力蓝图                                                  #
#  high (p2p): 关闭蓝图，仅 KV                                          #
# ──────────────────────────────────────────────────────────────────── #
class FamilyAdaptiveBlueprint(BasePlugin):
    def __init__(self, scale=1.6, strength=0.8, dt=0.01,
                 low_thresh=0.03, high_thresh=0.5):
        self.scale = scale
        self.strength = strength
        self.dt = dt
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self._jerk_hist = []
        self._last_a = None

    def inject(self, a, **kw):
        bp = kw.get('blueprint', None)
        if bp is None:
            return a
        j = measure_jerk(self, a)
        # 族分类
        if j < self.low_thresh or j > self.high_thresh:
            # low (transformer) 或 high (p2p): 不注入蓝图
            return a
        # mid (standard): 全力注入
        bp = np.asarray(bp, dtype=float) * self.scale
        # 平滑过渡：在 low_thresh 处渐变
        if j < self.low_thresh * 1.5:
            fade = (j - self.low_thresh) / (self.low_thresh * 0.5 + 1e-9)
            fade = max(0.0, min(1.0, fade))
        else:
            fade = 1.0
        return a + self.strength * fade * (bp - a)


# ──────────────────────────────────────────────────────────────────── #
#  V46: 级联 + 双门控蓝图                                               #
# ──────────────────────────────────────────────────────────────────── #
class CascadePlusDualGateBp(BasePlugin):
    def __init__(self, n_passes=7, bp_scale=1.6, bp_strength=0.8,
                 bp_j_lo=0.05, bp_j_peak=0.12, bp_j_hi=0.3,
                 cascade_j_peak=0.15, cascade_j_hi=0.5):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': cascade_j_peak, 'j_hi': cascade_j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_passes)]
        self.bp = DualGateScaledBlueprint(scale=bp_scale, strength=bp_strength,
                                            j_lo=bp_j_lo, j_peak=bp_j_peak, j_hi=bp_j_hi)
    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.bp.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  V47: 级联 + 族自适应蓝图                                              #
# ──────────────────────────────────────────────────────────────────── #
class CascadePlusFamilyAdaptiveBp(BasePlugin):
    def __init__(self, n_passes=7, bp_scale=1.6, bp_strength=0.8,
                 low_thresh=0.03, high_thresh=0.5,
                 cascade_j_peak=0.15, cascade_j_hi=0.5):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': cascade_j_peak, 'j_hi': cascade_j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_passes)]
        self.bp = FamilyAdaptiveBlueprint(scale=bp_scale, strength=bp_strength,
                                            low_thresh=low_thresh, high_thresh=high_thresh)
    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.bp.inject(a, **kw)
        return a


def main():
    print("=" * 120)
    print("第十五轮：双门控蓝图 + 族自适应")
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

    # ── V44 双门控蓝图 单独 ──
    all_r['R1_dual_bp_s16_08'] = run_round(1, "双门控蓝图 scale=1.6 str=0.8",
        lambda: DualGateScaledBlueprint(scale=1.6, strength=0.8))
    all_r['R2_dual_bp_s15_05'] = run_round(2, "双门控蓝图 scale=1.5 str=0.5",
        lambda: DualGateScaledBlueprint(scale=1.5, strength=0.5))

    # ── V46 级联 + 双门控蓝图 扫参 ──
    # 扫 scale
    for scale in [1.4, 1.5, 1.6, 1.7, 1.8]:
        all_r[f'R3_cascade7_dual_bp_s{scale}'] = run_round(3,
            f"级联7 + 双门蓝图(scale={scale})",
            lambda s=scale: CascadePlusDualGateBp(bp_scale=s, bp_strength=0.8))

    # 扫 strength
    for strength in [0.3, 0.5, 0.8, 1.0, 1.2]:
        all_r[f'R4_cascade7_dual_bp_str{strength}'] = run_round(4,
            f"级联7 + 双门蓝图(str={strength})",
            lambda st=strength: CascadePlusDualGateBp(bp_scale=1.6, bp_strength=st))

    # 扫窄门参数
    for j_lo, j_peak in [(0.03, 0.10), (0.05, 0.12), (0.08, 0.15), (0.10, 0.18)]:
        all_r[f'R5_cascade7_dual_bp_lo{j_lo}_pk{j_peak}'] = run_round(5,
            f"级联7 + 双门蓝图(j_lo={j_lo},j_pk={j_peak})",
            lambda jl=j_lo, jp=j_peak: CascadePlusDualGateBp(
                bp_scale=1.6, bp_strength=0.8, bp_j_lo=jl, bp_j_peak=jp))

    # ── V47 级联 + 族自适应蓝图 ──
    all_r['R6_cascade7_fam_bp'] = run_round(6, "级联7 + 族自适应蓝图",
        lambda: CascadePlusFamilyAdaptiveBp(bp_scale=1.6, bp_strength=0.8))

    # 扫 low_thresh
    for lt in [0.02, 0.03, 0.05, 0.08]:
        all_r[f'R7_cascade7_fam_bp_lt{lt}'] = run_round(7,
            f"级联7 + 族自适应蓝图(lt={lt})",
            lambda l=lt: CascadePlusFamilyAdaptiveBp(
                bp_scale=1.6, bp_strength=0.8, low_thresh=l))

    # ── 级联5 + 双门蓝图（更快）──
    all_r['R8_cascade5_dual_bp'] = run_round(8, "级联5 + 双门蓝图(scale=1.6)",
        lambda: CascadePlusDualGateBp(n_passes=5, bp_scale=1.6, bp_strength=0.8))

    # ── 级联9 + 双门蓝图 ──
    all_r['R9_cascade9_dual_bp'] = run_round(9, "级联9 + 双门蓝图(scale=1.6)",
        lambda: CascadePlusDualGateBp(n_passes=9, bp_scale=1.6, bp_strength=0.8))

    # ── 双门蓝图先行 + 级联 ──
    class DualBpPlusCascade(BasePlugin):
        def __init__(self):
            self.bp = DualGateScaledBlueprint(scale=1.6, strength=0.8,
                                                j_lo=0.05, j_peak=0.12, j_hi=0.3)
            self.cascade = [GoldilocksFusion(**wide) for _ in range(7)]
        def inject(self, a, **kw):
            a = self.bp.inject(a, **kw)
            for p in self.cascade:
                a = p.inject(a, **kw)
            return a
    all_r['R10_dual_bp_then_cascade7'] = run_round(10, "双门蓝图 + 级联7",
        lambda: DualBpPlusCascade())

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

    out = os.path.join(os.path.dirname(__file__), "stress_test15.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
