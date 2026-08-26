#!/usr/bin/env python3
"""
stress_test10.py — 第十轮：蓝图锚定注入 (BlueprintAnchored)

发现：plugins 接收 blueprint kwarg —— 这是 env 提供的"全局步态相位蓝图"，
对应 TPIP 协议中的电路 B（全局目标传播）。在真实系统中由高层轨迹规划器生成。

V19 BlueprintAnchored：以 blueprint 为注入目标，jerk 钟形门控。
  - 对 p2p: 强力推向纯净步态蓝图 → 巨大提升
  - 对 standard/transformer: 蓝图已是其本身步态 → 不破坏，门控只在噪声出现时激活

V20 CascadeV19：蓝图锚定 + 短池自回响级联
V21 AdaptiveBlueprint：蓝图权重 + 自适应权重动态融合
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


# ──────────────────────────────────────────────────────────────────── #
#  V19: 蓝图锚定注入 (BlueprintAnchored)                                #
#  使用 kwarg['blueprint'] 作为注入目标，jerk 钟形门控                  #
# ──────────────────────────────────────────────────────────────────── #
class BlueprintAnchored(BasePlugin):
    """
    V19: 蓝图锚定。注入方向 = (blueprint - a)。
    门控：jerk 高(噪声大)时强力推向蓝图；jerk 低(已平滑)时仅轻推。
    """
    def __init__(self, strength=1.0,
                 j_lo=0.01, j_peak=0.1, j_hi=0.3):
        self.strength = strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._jerk_hist = []
        self._last_a = None

    def _measure_jerk(self, a):
        if self._last_a is None:
            self._last_a = a.copy()
            return 0.0
        j = float(np.linalg.norm(a - self._last_a))
        self._last_a = a.copy()
        self._jerk_hist.append(j)
        if len(self._jerk_hist) > 20:
            self._jerk_hist.pop(0)
        if len(self._jerk_hist) < 3:
            return 0.0
        return float(np.mean(self._jerk_hist))

    def _gate(self, j):
        if j < self.j_lo or j > self.j_hi:
            return 0.0
        if j <= self.j_peak:
            t = (j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)
            return float(np.sin(t * np.pi / 2))
        else:
            t = (j - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)
            return float(np.cos(t * np.pi / 2))

    def inject(self, a, **kw):
        bp = kw.get('blueprint', None)
        if bp is None:
            return a
        j = self._measure_jerk(a)
        gate = self._gate(j)
        if gate < 0.01:
            return a
        bp = np.asarray(bp, dtype=float)
        # 蓝图注入
        return a + self.strength * gate * (bp - a)


# ──────────────────────────────────────────────────────────────────── #
#  V20: 蓝图锚定 + 短池自回响 级联                                      #
# ──────────────────────────────────────────────────────────────────── #
class CascadeBlueprint(BasePlugin):
    """蓝图锚定 + GoldilocksFusion 级联"""
    def __init__(self, n_passes=3, lam=1.0, alpha=1.0, bp_strength=1.0,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        self.bp = BlueprintAnchored(strength=bp_strength,
                                     j_lo=j_lo, j_peak=j_peak, j_hi=j_hi)
        self.gold = [GoldilocksFusion(lam=lam, alpha=alpha,
                                       j_peak=j_peak, j_hi=j_hi)
                     for _ in range(n_passes)]
    def inject(self, a, **kw):
        a = self.bp.inject(a, **kw)
        for g in self.gold:
            a = g.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  V21: 自适应蓝图融合                                                  #
#  蓝图权重 + 自身惯性 权重 动态加权                                    #
# ──────────────────────────────────────────────────────────────────── #
class AdaptiveBlueprintFusion(BasePlugin):
    """
    V21: 蓝图 × 自身惯性 自适应融合。
    - 高 jerk: 蓝图权重高（向纯净步态靠）
    - 低 jerk: 自身惯性权重高（保留自己的好步态）
    """
    def __init__(self, bp_strength=1.0, echo_strength=1.0, pool=30,
                 j_lo=0.01, j_peak=0.1, j_hi=0.3):
        self.bp_strength = bp_strength
        self.echo_strength = echo_strength
        self.pool = pool
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None

    def _measure_jerk(self, a):
        if self._last_a is None:
            self._last_a = a.copy()
            return 0.0
        j = float(np.linalg.norm(a - self._last_a))
        self._last_a = a.copy()
        self._jerk_hist.append(j)
        if len(self._jerk_hist) > 20:
            self._jerk_hist.pop(0)
        if len(self._jerk_hist) < 3:
            return 0.0
        return float(np.mean(self._jerk_hist))

    def _gate(self, j):
        if j < self.j_lo or j > self.j_hi:
            return 0.0
        if j <= self.j_peak:
            t = (j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)
            return float(np.sin(t * np.pi / 2))
        else:
            t = (j - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)
            return float(np.cos(t * np.pi / 2))

    def _weighted_median(self, H):
        """指数加权中位数趋势"""
        n = len(H)
        w = np.exp(-0.3 * np.arange(n)[::-1])
        w /= w.sum()
        trend = np.zeros(H.shape[1])
        for j in range(H.shape[1]):
            col = H[:, j]
            order = np.argsort(col)
            cum = np.cumsum(w[order])
            idx = int(np.searchsorted(cum, 0.5))
            trend[j] = col[order[min(idx, n-1)]]
        return trend

    def inject(self, a, **kw):
        bp = kw.get('blueprint', None)
        if bp is None:
            return a
        bp = np.asarray(bp, dtype=float)

        # 更新历史
        self._hist.append(a.copy())
        if len(self._hist) > self.pool:
            self._hist.pop(0)

        j = self._measure_jerk(a)
        gate = self._gate(j)
        if gate < 0.01:
            return a

        # 自身惯性趋势
        if len(self._hist) >= 3:
            H = np.array(self._hist)
            trend = self._weighted_median(H)
        else:
            trend = a

        # jerk 越高 → 蓝图权重越重；jerk 越低 → 自身惯性权重越重
        # 用 gate 作为蓝图权重，1-gate 作为自身惯性权重
        w_bp = gate
        w_self = 1.0 - gate

        target = w_bp * bp + w_self * trend
        return a + (self.bp_strength * w_bp + self.echo_strength * w_self) * gate * (target - a)


# ──────────────────────────────────────────────────────────────────── #
#  V22: 蓝图强锚定 + Goldilocks 级联（极限配置）                       #
# ──────────────────────────────────────────────────────────────────── #
class StrongBlueprintCascade(BasePlugin):
    def __init__(self, bp_strength=2.0, n_passes=5, lam=1.0, alpha=1.0,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5):
        self.bp = BlueprintAnchored(strength=bp_strength,
                                     j_lo=j_lo, j_peak=j_peak, j_hi=j_hi)
        self.gold = [GoldilocksFusion(lam=lam, alpha=alpha,
                                       j_peak=j_peak, j_hi=j_hi)
                     for _ in range(n_passes)]
    def inject(self, a, **kw):
        # 蓝图先行：强力推向纯净步态
        a = self.bp.inject(a, **kw)
        # 再用 Goldilocks 精修
        for g in self.gold:
            a = g.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  V23: 极宽门 + 极强蓝图                                              #
# ──────────────────────────────────────────────────────────────────── #
class ExtremeBlueprint(BasePlugin):
    def __init__(self, strength=3.0, j_lo=0.005, j_peak=0.2, j_hi=1.5):
        self.strength = strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._jerk_hist = []
        self._last_a = None

    def _measure_jerk(self, a):
        if self._last_a is None:
            self._last_a = a.copy()
            return 0.0
        j = float(np.linalg.norm(a - self._last_a))
        self._last_a = a.copy()
        self._jerk_hist.append(j)
        if len(self._jerk_hist) > 20:
            self._jerk_hist.pop(0)
        if len(self._jerk_hist) < 3:
            return 0.0
        return float(np.mean(self._jerk_hist))

    def _gate(self, j):
        # 极宽门：几乎不降级
        if j < self.j_lo:
            return 0.0
        if j > self.j_hi:
            return 0.5  # 极高也半开
        if j <= self.j_peak:
            t = (j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)
            return float(np.sin(t * np.pi / 2))
        else:
            t = (j - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)
            return max(0.5, float(np.cos(t * np.pi / 2)))

    def inject(self, a, **kw):
        bp = kw.get('blueprint', None)
        if bp is None:
            return a
        j = self._measure_jerk(a)
        gate = self._gate(j)
        if gate < 0.01:
            return a
        bp = np.asarray(bp, dtype=float)
        return a + self.strength * gate * (bp - a)


def main():
    print("=" * 120)
    print("第十轮：蓝图锚定注入 (BlueprintAnchored) — 利用全局步态蓝图（电路B）")
    print("=" * 120)
    all_r = {}
    wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}

    # ── 对照：上轮最优 ──
    class RefCascade(BasePlugin):
        def __init__(self):
            self.passes = [GoldilocksFusion(**wide) for _ in range(7)]
        def inject(self, a, **kw):
            for p in self.passes:
                a = p.inject(a, **kw)
            return a
    all_r['R0_cascade7_wide_ref'] = run_round(0, "级联7宽门(对照)",
        lambda: RefCascade())

    # ── V19 蓝图锚定 单独 ──
    all_r['R1_bp_s05'] = run_round(1, "蓝图锚定 strength=0.5",
        lambda: BlueprintAnchored(strength=0.5,
                                   j_lo=0.01, j_peak=0.1, j_hi=0.3))
    all_r['R2_bp_s10_wide'] = run_round(2, "蓝图锚定 宽门 strength=1.0",
        lambda: BlueprintAnchored(strength=1.0,
                                   j_lo=0.01, j_peak=0.15, j_hi=0.5))
    all_r['R3_bp_s15_xwide'] = run_round(3, "蓝图锚定 极宽门 strength=1.5",
        lambda: BlueprintAnchored(strength=1.5,
                                   j_lo=0.01, j_peak=0.20, j_hi=0.8))
    all_r['R4_bp_s20_xxwide'] = run_round(4, "蓝图锚定 极宽门 strength=2.0",
        lambda: BlueprintAnchored(strength=2.0,
                                   j_lo=0.01, j_peak=0.25, j_hi=1.5))
    all_r['R5_bp_s30_xxwide'] = run_round(5, "蓝图锚定 极宽门 strength=3.0",
        lambda: BlueprintAnchored(strength=3.0,
                                   j_lo=0.01, j_peak=0.30, j_hi=2.0))

    # ── V20 蓝图锚定 + Goldilocks 级联 ──
    all_r['R6_bp_s10_cascade3_wide'] = run_round(6, "蓝图1.0 + 级联3 宽门",
        lambda: CascadeBlueprint(n_passes=3, lam=1.0, alpha=1.0, bp_strength=1.0,
                                   j_lo=0.01, j_peak=0.15, j_hi=0.5))
    all_r['R7_bp_s10_cascade5_wide'] = run_round(7, "蓝图1.0 + 级联5 宽门",
        lambda: CascadeBlueprint(n_passes=5, lam=1.0, alpha=1.0, bp_strength=1.0,
                                   j_lo=0.01, j_peak=0.15, j_hi=0.5))
    all_r['R8_bp_s15_cascade5_wide'] = run_round(8, "蓝图1.5 + 级联5 宽门",
        lambda: CascadeBlueprint(n_passes=5, lam=1.0, alpha=1.0, bp_strength=1.5,
                                   j_lo=0.01, j_peak=0.15, j_hi=0.5))
    all_r['R9_bp_s20_cascade5_wide'] = run_round(9, "蓝图2.0 + 级联5 宽门",
        lambda: CascadeBlueprint(n_passes=5, lam=1.0, alpha=1.0, bp_strength=2.0,
                                   j_lo=0.01, j_peak=0.20, j_hi=0.8))
    all_r['R10_bp_s20_cascade7_wide'] = run_round(10, "蓝图2.0 + 级联7 宽门",
        lambda: CascadeBlueprint(n_passes=7, lam=1.0, alpha=1.0, bp_strength=2.0,
                                   j_lo=0.01, j_peak=0.20, j_hi=0.8))

    # ── V21 自适应蓝图融合 ──
    all_r['R11_adaptive_bp_wide'] = run_round(11, "自适应蓝图融合 宽门",
        lambda: AdaptiveBlueprintFusion(bp_strength=1.0, echo_strength=1.0, pool=30,
                                         j_lo=0.01, j_peak=0.15, j_hi=0.5))
    all_r['R12_adaptive_bp_xwide'] = run_round(12, "自适应蓝图融合 极宽门",
        lambda: AdaptiveBlueprintFusion(bp_strength=1.5, echo_strength=1.5, pool=50,
                                         j_lo=0.01, j_peak=0.20, j_hi=0.8))

    # ── V22 强蓝图级联 ──
    all_r['R13_strong_bp_cascade5'] = run_round(13, "强蓝图(2.0)+级联5宽门",
        lambda: StrongBlueprintCascade(bp_strength=2.0, n_passes=5,
                                        lam=1.0, alpha=1.0,
                                        j_lo=0.01, j_peak=0.15, j_hi=0.5))
    all_r['R14_strong_bp_cascade7'] = run_round(14, "强蓝图(3.0)+级联7宽门",
        lambda: StrongBlueprintCascade(bp_strength=3.0, n_passes=7,
                                        lam=1.0, alpha=1.0,
                                        j_lo=0.01, j_peak=0.20, j_hi=0.8))

    # ── V23 极宽门极强蓝图 ──
    all_r['R15_extreme_bp_s3'] = run_round(15, "极宽门极强蓝图 strength=3.0",
        lambda: ExtremeBlueprint(strength=3.0))
    all_r['R16_extreme_bp_s5'] = run_round(16, "极宽门极强蓝图 strength=5.0",
        lambda: ExtremeBlueprint(strength=5.0))
    all_r['R17_extreme_bp_s10'] = run_round(17, "极宽门极强蓝图 strength=10.0",
        lambda: ExtremeBlueprint(strength=10.0))

    # ── 极强蓝图 + 级联 ──
    class ExtremeBpPlusCascade(BasePlugin):
        def __init__(self, bp_s, n_passes):
            self.bp = ExtremeBlueprint(strength=bp_s)
            self.gold = [GoldilocksFusion(**wide) for _ in range(n_passes)]
        def inject(self, a, **kw):
            a = self.bp.inject(a, **kw)
            for g in self.gold:
                a = g.inject(a, **kw)
            return a
    all_r['R18_extreme_bp5_cascade5'] = run_round(18, "极宽门蓝图5.0 + 级联5宽门",
        lambda: ExtremeBpPlusCascade(5.0, 5))
    all_r['R19_extreme_bp10_cascade7'] = run_round(19, "极宽门蓝图10.0 + 级联7宽门",
        lambda: ExtremeBpPlusCascade(10.0, 7))

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

    out = os.path.join(os.path.dirname(__file__), "stress_test10.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
