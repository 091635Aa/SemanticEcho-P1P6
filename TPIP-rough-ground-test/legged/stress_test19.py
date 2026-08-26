#!/usr/bin/env python3
"""
stress_test19.py — 第十九轮：T=12800 极限 + 参数极致调优

stress_test18 结果：
  R7 T=6400 非对称级联 → avg=+0.4493 (+44.93%), p2p=+0.1107, Universal=YES
  → T 越长优化率越高(800→1600→3200→6400 单调上升)
  → AdaptiveLPF 成功修复 P2P 基座

趋势外推：
  T=800→0.3258, 1600→0.3516, 3200→0.3805, 6400→0.4493
  每次翻倍约 +0.05~0.07 → T=12800 期望 ~+0.50

本轮目标：
  1. T=12800 测试最优配置 (AsymmetricCascade cascade5 + LPF)
  2. T=6400 参数精调：cascade 深度、LPF decay/strength、j_act_lo
  3. 新机制：TaperedCascade（梯度 j_peak 衰减）+ DualLPF
  4. 极长仿真稳定性验证
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion

# T=6400 用5种子保证精度；T=12800 用3种子节约时间
SEEDS_5 = [42, 137, 2024, 7777, 314159]
SEEDS_3 = [42, 137, 2024]


def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)


def eval_plugin(make_fn, family, seeds, T=800):
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


def run_round(round_num, name, make_fn, T=800, seeds=SEEDS_5):
    results = {}
    for fam in ['p2p', 'standard', 'transformer']:
        m, s = eval_plugin(make_fn, fam, seeds, T=T)
        results[fam] = {'mean': m, 'std': s}
    uni = all(results[f]['mean'] > 0 for f in ['p2p', 'standard', 'transformer'])
    results['universal'] = uni
    tag = "OK" if uni else "--"
    avg = (results['standard']['mean'] + results['transformer']['mean']) / 2
    print(f"  R{round_num}: {name:50s}  std={results['standard']['mean']:+.4f}  trans={results['transformer']['mean']:+.4f}  avg={avg:+.4f}  p2p={results['p2p']['mean']:+.4f}  {tag}")
    return results


# ──────────────────────────────────────────────────────────────────── #
#  V52 复用：AdaptiveLPF (来自 stress_test18)                       #
# ──────────────────────────────────────────────────────────────────── #
class AdaptiveLPF(BasePlugin):
    def __init__(self, decay=0.3, strength=0.8,
                 j_act_lo=0.3, j_act_hi=1.0):
        self.decay = decay
        self.strength = strength
        self.j_act_lo = j_act_lo
        self.j_act_hi = j_act_hi
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

    def _adaptive_gate(self, j):
        if j < self.j_act_lo:
            return 0.0
        if j >= self.j_act_hi:
            return 1.0
        t = (j - self.j_act_lo) / (self.j_act_hi - self.j_act_lo + 1e-9)
        return float(t)

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        self._hist.append(a.copy())
        if len(self._hist) > 5:
            self._hist.pop(0)
        if len(self._hist) < 2:
            return a
        j = self._measure_jerk(a)
        gate = self._adaptive_gate(j)
        if gate < 0.01:
            return a
        weights = np.array([(1 - self.decay) ** i for i in range(len(self._hist) - 1, -1, -1)])
        weights = weights / weights.sum()
        smoothed = np.average(np.stack(self._hist), axis=0, weights=weights)
        return a + self.strength * gate * (smoothed - a)


# ──────────────────────────────────────────────────────────────────── #
#  V53 复用：非对称级联 (AsymmetricCascade)                          #
# ──────────────────────────────────────────────────────────────────── #
class AsymmetricCascade(BasePlugin):
    def __init__(self, n_cascade=5, lpf_decay=0.3, lpf_strength=0.8,
                 j_peak=0.15, j_hi=0.5,
                 j_act_lo=0.3, j_act_hi=1.0):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_cascade)]
        self.lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                               j_act_lo=j_act_lo, j_act_hi=j_act_hi)

    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        return self.lpf.inject(a, **kw)


# ──────────────────────────────────────────────────────────────────── #
#  V56: TaperedCascade — 各趟梯度 j_peak 衰减                       #
#  早期宽门抓大趋势，后期窄门精细修正                              #
# ──────────────────────────────────────────────────────────────────── #
class TaperedCascade(BasePlugin):
    def __init__(self, n_passes=5, lpf_decay=0.3, lpf_strength=0.8,
                 j_peak_start=0.20, j_peak_end=0.05,
                 j_hi_start=0.5, j_hi_end=0.15,
                 j_act_lo=0.3, j_act_hi=1.0):
        j_peaks = np.linspace(j_peak_start, j_peak_end, n_passes)
        j_his = np.linspace(j_hi_start, j_hi_end, n_passes)
        self.passes = []
        for jp, jh in zip(j_peaks, j_his):
            self.passes.append(GoldilocksFusion(
                lam=1.0, alpha=1.0, j_peak=jp, j_hi=jh))
        self.lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                               j_act_lo=j_act_lo, j_act_hi=j_act_hi)

    def inject(self, a, **kw):
        for p in self.passes:
            a = p.inject(a, **kw)
        return self.lpf.inject(a, **kw)


# ──────────────────────────────────────────────────────────────────── #
#  V57: DualLPF — 双 LPF (低强度 + 高强度组合)                      #
#  低强度 LPF 抑制中频抖动，高强度 LPF 抑制高频尖峰                  #
# ──────────────────────────────────────────────────────────────────── #
class DualLPF(BasePlugin):
    def __init__(self, decay1=0.2, strength1=0.5,
                 decay2=0.4, strength2=0.5,
                 j_act_lo=0.3, j_act_hi=1.0):
        self.lpf1 = AdaptiveLPF(decay=decay1, strength=strength1,
                                j_act_lo=j_act_lo, j_act_hi=j_act_hi)
        self.lpf2 = AdaptiveLPF(decay=decay2, strength=strength2,
                                j_act_lo=j_act_lo, j_act_hi=j_act_hi)

    def inject(self, a, **kw):
        a = self.lpf1.inject(a, **kw)
        a = self.lpf2.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  V58: CascadePlusDualLPF — 级联 + 双 LPF                          #
# ──────────────────────────────────────────────────────────────────── #
class CascadePlusDualLPF(BasePlugin):
    def __init__(self, n_cascade=5,
                 decay1=0.2, strength1=0.5,
                 decay2=0.4, strength2=0.5,
                 j_peak=0.15, j_hi=0.5,
                 j_act_lo=0.3, j_act_hi=1.0):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_cascade)]
        self.dlpf = DualLPF(decay1=decay1, strength1=strength1,
                            decay2=decay2, strength2=strength2,
                            j_act_lo=j_act_lo, j_act_hi=j_act_hi)

    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        return self.dlpf.inject(a, **kw)


# ──────────────────────────────────────────────────────────────────── #
#  V59: LPF Sandwiched — LPF-Cascade-LPF 三明治                     #
#  前置 LPF 压制初始噪声，级联精炼，后置 LPF 平滑输出               #
# ──────────────────────────────────────────────────────────────────── #
class LPFSandwiched(BasePlugin):
    def __init__(self, n_cascade=5, lpf_decay=0.3, lpf_strength=0.5,
                 j_peak=0.15, j_hi=0.5,
                 j_act_lo=0.3, j_act_hi=1.0):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.pre_lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                                     j_act_lo=j_act_lo, j_act_hi=j_act_hi)
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_cascade)]
        self.post_lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                                     j_act_lo=j_act_lo, j_act_hi=j_act_hi)

    def inject(self, a, **kw):
        a = self.pre_lpf.inject(a, **kw)
        for p in self.cascade:
            a = p.inject(a, **kw)
        return self.post_lpf.inject(a, **kw)


def main():
    print("=" * 120)
    print("第十九轮：T=12800 极限 + 参数极致调优")
    print("=" * 120)
    all_r = {}
    wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}

    # ── T=6400 对照：确认上轮最优 ──
    all_r['R0_ref_T6400'] = run_round(0, "对照 非对称级联 T=6400",
        lambda: AsymmetricCascade(n_cascade=5, lpf_decay=0.3, lpf_strength=0.8),
        T=6400, seeds=SEEDS_5)

    # ── T=6400 cascade深度扫描 ──
    for n in [3, 5, 7, 9]:
        all_r[f'R1_cascade{n}_T6400'] = run_round(1,
            f"级联{n}+LPF T=6400",
            lambda n=n: AsymmetricCascade(n_cascade=n), T=6400)

    # ── T=6400 LPF decay 扫描 ──
    for decay in [0.2, 0.3, 0.4, 0.5]:
        all_r[f'R2_lpf_d{decay}_T6400'] = run_round(2,
            f"级联5+LPF d={decay} T=6400",
            lambda d=decay: AsymmetricCascade(n_cascade=5, lpf_decay=d),
            T=6400)

    # ── T=6400 LPF strength 扫描 ──
    for strength in [0.6, 0.8, 1.0]:
        all_r[f'R3_lpf_s{strength}_T6400'] = run_round(3,
            f"级联5+LPF s={strength} T=6400",
            lambda s=strength: AsymmetricCascade(n_cascade=5, lpf_strength=s),
            T=6400)

    # ── T=6400 j_act_lo 扫描 ──
    for j_act_lo in [0.2, 0.3, 0.4, 0.5]:
        all_r[f'R4_jlo{j_act_lo}_T6400'] = run_round(4,
            f"级联5+LPF j_lo={j_act_lo} T=6400",
            lambda j=j_act_lo: AsymmetricCascade(n_cascade=5, j_act_lo=j),
            T=6400)

    # ── T=6400 TaperedCascade ──
    all_r['R5_tapered_T6400'] = run_round(5, "TaperedCascade T=6400",
        lambda: TaperedCascade(n_passes=5), T=6400)

    # ── T=6400 CascadePlusDualLPF ──
    all_r['R6_dualLPF_T6400'] = run_round(6, "级联+双LPF T=6400",
        lambda: CascadePlusDualLPF(n_cascade=5), T=6400)

    # ── T=6400 LPF Sandwiched ──
    all_r['R7_sandwich_T6400'] = run_round(7, "LPF三明治 T=6400",
        lambda: LPFSandwiched(n_cascade=5, lpf_decay=0.3, lpf_strength=0.5),
        T=6400)

    # ── T=12800 极限：3种子节约时间 ──
    all_r['R8_asym_T12800'] = run_round(8, "非对称级联 T=12800(极限)",
        lambda: AsymmetricCascade(n_cascade=5, lpf_decay=0.3, lpf_strength=0.8),
        T=12800, seeds=SEEDS_3)

    # ── 汇总 ──
    print(f"\n{'='*120}")
    print("汇总（按 std+trans 均值排序）")
    print(f"{'='*120}")
    sorted_r = sorted(all_r.items(),
                      key=lambda x: (x[1]['standard']['mean'] + x[1]['transformer']['mean']) / 2,
                      reverse=True)
    print(f"  {'配置':42s}  {'std':>8s}  {'trans':>8s}  {'avg':>8s}  {'p2p':>8s}  通用?")
    print("  " + "-" * 90)
    for k, v in sorted_r:
        uni = v.get('universal', False)
        avg = (v['standard']['mean'] + v['transformer']['mean']) / 2
        print(f"  {k:42s}  {v['standard']['mean']:+.4f}  {v['transformer']['mean']:+.4f}  {avg:+.4f}  {v['p2p']['mean']:+.4f}  {'YES' if uni else 'NO'}")

    best = sorted_r[0]
    print(f"\n  最优: {best[0]} → avg={(best[1]['standard']['mean']+best[1]['transformer']['mean'])/2:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "stress_test19.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
