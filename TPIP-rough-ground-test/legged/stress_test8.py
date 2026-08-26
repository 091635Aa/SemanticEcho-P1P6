#!/usr/bin/env python3
"""
stress_test8.py — 第八轮：深度级联(9/11/13趟宽门) + 在线轨迹优化器 + CI反馈
基于 stress_test7 最优 R5(级联7宽门) 继续 push：
  1. 深度级联：9/11/13 趟宽门
  2. 级联 × 在线优化器（多项式拟合）组合
  3. 在线优化器扫参（window/polyorder/strength）
  4. 在线优化器 × 深度级联组合
  5. 在线 CI 反馈：估计本地 jerk→smooth，做自适应强度
  6. 宽门极限：j_hi=0.8/1.0
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion

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
#  级联：N 趟 GoldilocksFusion                                         #
# ──────────────────────────────────────────────────────────────────── #
class Cascade(BasePlugin):
    def __init__(self, configs):
        self.passes = [GoldilocksFusion(**c) for c in configs]
    def inject(self, a, **kw):
        for p in self.passes:
            a = p.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  在线轨迹优化器（多项式拟合 + 钟形门控）                              #
# ──────────────────────────────────────────────────────────────────── #
class OnlineTrajectoryOptimizer(BasePlugin):
    """
    V11+：对最近 N 步做多项式拟合，用拟合值替换当前动作。
    钟形门控控制强度，平滑基座也能获得增益。
    """
    def __init__(self, window=7, polyorder=2, strength=0.5,
                 j_lo=0.01, j_peak=0.1, j_hi=0.3):
        self.window = window
        self.polyorder = polyorder
        self.strength = strength
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

    def _polyfit_smooth(self, a):
        self._hist.append(a.copy())
        if len(self._hist) > self.window:
            self._hist.pop(0)
        if len(self._hist) < self.polyorder + 1:
            return a
        H = np.array(self._hist)
        n = len(H)
        x = np.arange(n, dtype=float)
        try:
            coeffs = np.polyfit(x, H, self.polyorder)
            predicted = np.polyval(coeffs, n)  # 预测下一步
            return predicted
        except Exception:
            return a

    def inject(self, a, **kw):
        j = self._measure_jerk(a)
        gate = self._gate(j)
        if gate < 0.01:
            return a
        smoothed = self._polyfit_smooth(a)
        return a + self.strength * gate * (smoothed - a)


# ──────────────────────────────────────────────────────────────────── #
#  在线 CI 反馈：基于本地 jerk 与一阶 phase 估计反馈强度               #
# ──────────────────────────────────────────────────────────────────── #
class CIFeedbackFusion(BasePlugin):
    """
    V13：在线估计"本地 smoothness"，作为反馈信号调节 Echo/Tidal 强度。
    当本地 smoothness 高时，说明轨迹已好，适度减弱；当处于"可救"区间时增强。
    """
    def __init__(self, lam=1.0, alpha=1.0, kappa=0.12,
                 j_lo=0.01, j_peak=0.1, j_hi=0.3,
                 fb_gain=0.5, fb_window=30):
        self._echo = AdaptiveEcho(lam=lam)
        self._tidal = GatedTidal(alpha=alpha)
        self._kv = AdaptiveKV(kappa=kappa)
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self.fb_gain = fb_gain
        self.fb_window = fb_window
        self._jerk_hist = []
        self._smooth_hist = []  # 在线 smoothness 估计
        self._last_a = None

    def _goldilocks_gate(self, j):
        if j < self.j_lo or j > self.j_hi:
            return 0.0
        if j <= self.j_peak:
            t = (j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)
            return float(np.sin(t * np.pi / 2))
        else:
            t = (j - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)
            return float(np.cos(t * np.pi / 2))

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

    def _smoothness(self, j):
        """在线 smoothness 估计：1/(1+j/jitter_mult)。返回 (s, trend)。"""
        s = 1.0 / (1.0 + j / 0.05)
        self._smooth_hist.append(s)
        if len(self._smooth_hist) > self.fb_window:
            self._smooth_hist.pop(0)
        if len(self._smooth_hist) < 3:
            return s, 0.0
        # 反馈：若 smoothness 趋势上升(轨迹越来越好)，减弱；若下降，增强
        trend = self._smooth_hist[-1] - np.mean(self._smooth_hist[:-1])
        return s, float(trend)

    def inject(self, a, **kw):
        j = self._measure_jerk(a)
        gate = self._goldilocks_gate(j)
        gate_kv = 0.20

        a_orig = a.copy()
        a = self._kv.inject(a, **kw)
        a = a_orig + gate_kv * (a - a_orig)

        if gate > 0.01:
            # CI 反馈调节
            s, trend = self._smoothness(j)
            # trend>0: smoothness 上升 → 轨迹在改善，减弱注入(避免过冲)
            # trend<0: 在恶化 → 增强注入
            fb = 1.0 - self.fb_gain * trend  # trend∈(-1,1) → fb∈(0.5,1.5)
            fb = max(0.5, min(1.5, fb))
            eff_gate = gate * fb

            a_mid = a.copy()
            a_mid = self._echo.inject(a_mid, **kw)
            a_mid = self._tidal.inject(a_mid, **kw)
            a = a + eff_gate * (a_mid - a)

        return a


# 延迟导入：AdaptiveEcho 等在 plugins_v2 中
from plugins_v2 import AdaptiveEcho, GatedTidal, AdaptiveKV


# ──────────────────────────────────────────────────────────────────── #
#  组合：Cascade × OnlineOptimizer                                     #
# ──────────────────────────────────────────────────────────────────── #
class CascadePlusOptimizer(BasePlugin):
    def __init__(self, configs, opt_cfg=None):
        self.cascade = Cascade(configs)
        opt_cfg = opt_cfg or {}
        self.opt = OnlineTrajectoryOptimizer(**opt_cfg)
    def inject(self, a, **kw):
        a = self.cascade.inject(a, **kw)
        a = self.opt.inject(a, **kw)
        return a


def main():
    print("=" * 120)
    print("第八轮极限压测：深度级联(9/11/13宽门) + 在线优化器 + CI反馈")
    print("=" * 120)
    all_r = {}

    wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}
    xwide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.20, 'j_hi': 0.8}
    b = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.1, 'j_hi': 0.3}

    # ── 对照：级联7宽门（上一轮最优）──
    all_r['R0_cascade7_wide_ref'] = run_round(0, "级联7宽门(上轮最优对照)",
        lambda: Cascade([wide]*7))

    # ── 深度级联 ──
    all_r['R1_cascade9_wide'] = run_round(1, "级联9 宽门",
        lambda: Cascade([wide]*9))
    all_r['R2_cascade11_wide'] = run_round(2, "级联11 宽门",
        lambda: Cascade([wide]*11))
    all_r['R3_cascade13_wide'] = run_round(3, "级联13 宽门",
        lambda: Cascade([wide]*13))

    # ── 极宽门深度级联 ──
    all_r['R4_cascade9_xwide'] = run_round(4, "级联9 极宽门(j_hi=0.8)",
        lambda: Cascade([xwide]*9))
    all_r['R5_cascade11_xwide'] = run_round(5, "级联11 极宽门",
        lambda: Cascade([xwide]*11))

    # ── 在线优化器扫参 ──
    all_r['R6_opt_w7_p2_s05'] = run_round(6, "在线优化 w7 p2 s0.5",
        lambda: OnlineTrajectoryOptimizer(window=7, polyorder=2, strength=0.5,
                                          j_lo=0.01, j_peak=0.1, j_hi=0.3))
    all_r['R7_opt_w9_p3_s08'] = run_round(7, "在线优化 w9 p3 s0.8",
        lambda: OnlineTrajectoryOptimizer(window=9, polyorder=3, strength=0.8,
                                          j_lo=0.01, j_peak=0.1, j_hi=0.3))
    all_r['R8_opt_wide_w11_p3_s10'] = run_round(8, "在线优化 宽门 w11 p3 s1.0",
        lambda: OnlineTrajectoryOptimizer(window=11, polyorder=3, strength=1.0,
                                          j_lo=0.01, j_peak=0.15, j_hi=0.5))

    # ── 级联 + 在线优化器 组合 ──
    all_r['R9_cascade7_opt_w9'] = run_round(9, "级联7宽门 + 在线优化(w9 p3 s0.8)",
        lambda: CascadePlusOptimizer(
            [wide]*7,
            opt_cfg={'window': 9, 'polyorder': 3, 'strength': 0.8,
                     'j_lo': 0.01, 'j_peak': 0.1, 'j_hi': 0.3}))

    all_r['R10_cascade9_opt_w11'] = run_round(10, "级联9宽门 + 在线优化(w11 p3 s1.0宽门)",
        lambda: CascadePlusOptimizer(
            [wide]*9,
            opt_cfg={'window': 11, 'polyorder': 3, 'strength': 1.0,
                     'j_lo': 0.01, 'j_peak': 0.15, 'j_hi': 0.5}))

    all_r['R11_cascade11_opt_w11'] = run_round(11, "级联11宽门 + 在线优化(w11 p3 s1.0宽门)",
        lambda: CascadePlusOptimizer(
            [wide]*11,
            opt_cfg={'window': 11, 'polyorder': 3, 'strength': 1.0,
                     'j_lo': 0.01, 'j_peak': 0.15, 'j_hi': 0.5}))

    # ── CI 反馈融合 ──
    all_r['R12_ci_fb_wide'] = run_round(12, "CI反馈融合 宽门 fb=0.5",
        lambda: CIFeedbackFusion(lam=1.0, alpha=1.0, kappa=0.12,
                                 j_lo=0.01, j_peak=0.15, j_hi=0.5,
                                 fb_gain=0.5))
    all_r['R13_ci_fb_xwide'] = run_round(13, "CI反馈融合 极宽门 fb=0.5",
        lambda: CIFeedbackFusion(lam=1.0, alpha=1.0, kappa=0.12,
                                 j_lo=0.01, j_peak=0.20, j_hi=0.8,
                                 fb_gain=0.5))

    # ── CI反馈 × 在线优化器 组合 ──
    class CIFbPlusOpt(BasePlugin):
        def __init__(self):
            self.ci = CIFeedbackFusion(lam=1.0, alpha=1.0, kappa=0.12,
                                       j_lo=0.01, j_peak=0.15, j_hi=0.5,
                                       fb_gain=0.5)
            self.opt = OnlineTrajectoryOptimizer(window=11, polyorder=3, strength=1.0,
                                                  j_lo=0.01, j_peak=0.15, j_hi=0.5)
        def inject(self, a, **kw):
            a = self.ci.inject(a, **kw)
            a = self.opt.inject(a, **kw)
            return a
    all_r['R14_ci_fb_plus_opt'] = run_round(14, "CI反馈 + 在线优化",
        lambda: CIFbPlusOpt())

    # ── 三件套：级联5 + CI反馈 + 在线优化 ──
    class TripleCombo(BasePlugin):
        def __init__(self):
            self.cascade = Cascade([wide]*5)
            self.ci = CIFeedbackFusion(lam=1.0, alpha=1.0, kappa=0.12,
                                       j_lo=0.01, j_peak=0.15, j_hi=0.5,
                                       fb_gain=0.5)
            self.opt = OnlineTrajectoryOptimizer(window=11, polyorder=3, strength=1.0,
                                                  j_lo=0.01, j_peak=0.15, j_hi=0.5)
        def inject(self, a, **kw):
            a = self.cascade.inject(a, **kw)
            a = self.ci.inject(a, **kw)
            a = self.opt.inject(a, **kw)
            return a
    all_r['R15_triple_combo'] = run_round(15, "级联5宽门 + CI反馈 + 在线优化(三件套)",
        lambda: TripleCombo())

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
    print(f"\n  最优: {best[0]} → avg={best[1]['standard']['mean']+best[1]['transformer']['mean']/2:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "stress_test8.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
