#!/usr/bin/env python3
"""
stress_test20.py — 第二十轮：突破天花板的新机制

stress_test19 结果：
  T=12800 → avg=+0.4510, p2p=+0.2261
  但 T 翻倍仅 +0.0017 → T-scaling 已饱和！
  估计天花板在 ~0.45 (45% 改进)

诊断：
  当前 Goldilocks+LPF 只能平滑噪声/抖振
  无法改变基座的"基本步态模式"
  → 突破天花板需要"模式重塑"：直接注入蓝图/谐波

新机制（V60-V64）：
  V60 BlueprintForcer — 门控混合基座动作与蓝图动作
  V61 HarmonicInjector — 提取主频谐波并注入对齐信号
  V62 PhaseLockedLoop — PLL 相位跟踪 + 相位对齐注入
  V63 MultiScaleBlueprint — 基波+二次谐波注入
  V64 FamilyAdaptiveForcer — 自适应族检测 + 强制策略
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion

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
                                 dt=sim_b.dt)['coherence_index']
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
#  复用：AdaptiveLPF + AsymmetricCascade (来自 stress_test18/19)      #
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
#  V60: BlueprintForcer — 直接混合蓝图                              #
#  仅在中等 jerk 时激活，强制动作向蓝图对齐                          #
#  低 jerk(已平滑): 不打扰; 高 jerk(P2P): 由 LPF 处理               #
# ──────────────────────────────────────────────────────────────────── #
class BlueprintForcer(BasePlugin):
    def __init__(self, strength=0.5,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5):
        self.strength = strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._jerk_hist = []
        self._last_a = None

    def inject(self, a, **kw):
        bp = kw.get('blueprint', None)
        if bp is None:
            return a
        a = np.asarray(a, dtype=float)
        bp = np.asarray(bp, dtype=float)
        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a
        return a + self.strength * gate * (bp - a)


# ──────────────────────────────────────────────────────────────────── #
#  V61: HarmonicInjector — 提取主频并注入对齐谐波                   #
#  在线 FFT 估计主频，构造对齐的正弦信号注入                        #
# ──────────────────────────────────────────────────────────────────── #
class HarmonicInjector(BasePlugin):
    def __init__(self, strength=0.4, window=64, dt=0.01,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5):
        self.strength = strength
        self.window = window
        self.dt = dt
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._hist = []
        self._jerk_hist = []
        self._last_a = None
        self._dominant_freq = 1.0
        self._phase = 0.0

    def _estimate_dominant_freq(self):
        if len(self._hist) < self.window:
            return 1.0
        sig = np.array(self._hist[-self.window:])
        sig = sig - sig.mean(axis=0, keepdims=True)
        # 取每个关节的均值
        sig1d = sig.mean(axis=1)
        fft = np.fft.rfft(sig1d)
        freqs = np.fft.rfftfreq(len(sig1d), d=self.dt)
        if len(fft) < 2:
            return 1.0
        # 跳过 DC
        peak_idx = np.argmax(np.abs(fft[1:])) + 1
        return float(freqs[peak_idx])

    def inject(self, a, **kw):
        t = kw.get('t', 0.0)
        a = np.asarray(a, dtype=float)
        self._hist.append(a.copy())
        if len(self._hist) > self.window * 2:
            self._hist.pop(0)

        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a

        # 在线主频估计
        new_freq = self._estimate_dominant_freq()
        # 平滑频率更新
        self._dominant_freq = 0.95 * self._dominant_freq + 0.05 * new_freq
        # 累积相位
        self._phase += 2 * np.pi * self._dominant_freq * self.dt

        # 生成对齐谐波：单频正弦
        harmonic = 0.25 * np.sin(self._phase + np.linspace(0, np.pi, len(a)))
        return a + self.strength * gate * (harmonic - a)


# ──────────────────────────────────────────────────────────────────── #
#  V62: PhaseLockedLoop — PLL 相位跟踪                              #
#  通过相位误差反馈调整相位估计，使注入信号与基座同步                #
# ──────────────────────────────────────────────────────────────────── #
class PhaseLockedLoop(BasePlugin):
    def __init__(self, strength=0.4, dt=0.01,
                 natural_freq=1.0, kp_pll=0.5,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5):
        self.strength = strength
        self.dt = dt
        self.natural_freq = natural_freq
        self.kp_pll = kp_pll
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._phase = 0.0
        self._freq = natural_freq
        self._last_sig = None
        self._jerk_hist = []
        self._last_a = None

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        sig = float(np.mean(a))  # 单标量信号
        self._jerk_hist_append(a)

        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            self._update_pll(sig)
            return a

        self._update_pll(sig)
        # PLL 注入对齐信号
        aligned = 0.25 * np.sin(self._phase + np.linspace(0, np.pi, len(a)))
        return a + self.strength * gate * (aligned - a)

    def _jerk_hist_append(self, a):
        pass  # 由 measure_jerk 处理

    def _update_pll(self, sig):
        # 二阶 PLL：相位误差 = sin(θ_sig - θ_pll)
        # 简化：用 sig 的符号作为相位参考
        if self._last_sig is not None:
            d_sig = sig - self._last_sig
            # 当 sig 过零上升时，phase 应 = 0
            if abs(sig) > 1e-6:
                phase_err = np.arctan2(self._last_sig, sig) * 0.1
                self._freq += self.kp_pll * phase_err * self.dt
                self._freq = max(0.5, min(2.0, self._freq))
        self._phase += 2 * np.pi * self._freq * self.dt
        self._last_sig = sig


# ──────────────────────────────────────────────────────────────────── #
#  V63: MultiScaleBlueprint — 基波 + 二次谐波蓝图                    #
#  在标准蓝图基础上叠加二次谐波，制造更"自然"的步态                  #
# ──────────────────────────────────────────────────────────────────── #
class MultiScaleBlueprintForcer(BasePlugin):
    def __init__(self, strength=0.4, h2_amp=0.08,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5):
        self.strength = strength
        self.h2_amp = h2_amp
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._jerk_hist = []
        self._last_a = None

    def inject(self, a, **kw):
        bp = kw.get('blueprint', None)
        t = kw.get('t', 0.0)
        if bp is None:
            return a
        a = np.asarray(a, dtype=float)
        bp = np.asarray(bp, dtype=float)
        # 叠加二次谐波
        h2 = self.h2_amp * np.sin(2 * 2 * np.pi * 1.0 * t + np.linspace(0, np.pi, len(a)))
        bp_multi = bp + h2

        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a
        return a + self.strength * gate * (bp_multi - a)


# ──────────────────────────────────────────────────────────────────── #
#  V64: FamilyAdaptiveForcer — 在线族检测 + 自适应                   #
#  通过 jerk 统计自动识别族：低=transformer, 中=standard, 高=p2p    #
#  对不同族应用不同的强制策略                                       #
# ──────────────────────────────────────────────────────────────────── #
class FamilyAdaptiveForcer(BasePlugin):
    def __init__(self,
                 trans_strength=0.5,  # 低 jerk (transformer)
                 std_strength=0.6,    # 中 jerk (standard)
                 p2p_lpf_decay=0.3, p2p_lpf_strength=0.8,  # 高 jerk (p2p)
                 j_lo=0.05, j_mid=0.15, j_hi=0.5,
                 j_p2p_threshold=0.5):
        self.trans_strength = trans_strength
        self.std_strength = std_strength
        self.p2p_lpf = AdaptiveLPF(decay=p2p_lpf_decay, strength=p2p_lpf_strength,
                                    j_act_lo=j_p2p_threshold, j_act_hi=1.0)
        self.j_lo, self.j_mid, self.j_hi = j_lo, j_mid, j_hi
        self._jerk_hist = []
        self._last_a = None
        self._act_hist = []

    def inject(self, a, **kw):
        bp = kw.get('blueprint', None)
        if bp is None:
            return a
        a = np.asarray(a, dtype=float)
        self._act_hist.append(a.copy())
        if len(self._act_hist) > 5:
            self._act_hist.pop(0)

        j = measure_jerk(self, a)

        # 分级处理
        if j > self.j_hi:
            # P2P 路径：用 LPF
            return self.p2p_lpf.inject(a, **kw)
        elif j > self.j_lo:
            # standard/transformer 路径：蓝图强制
            gate = goldilocks_gate(j, self.j_lo, self.j_mid, self.j_hi)
            strength = self.std_strength if j > self.j_mid else self.trans_strength
            return a + strength * gate * (np.asarray(bp, dtype=float) - a)
        else:
            # 极低 jerk：不干预
            return a


# ──────────────────────────────────────────────────────────────────── #
#  组合：Cascade+LPF+BlueprintForcer                                #
# ──────────────────────────────────────────────────────────────────── #
class CascadeLPFForcer(BasePlugin):
    def __init__(self, n_cascade=5, lpf_decay=0.3, lpf_strength=0.8,
                 force_strength=0.4,
                 j_peak=0.15, j_hi=0.5,
                 j_act_lo=0.3, j_act_hi=1.0):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_cascade)]
        self.forcer = BlueprintForcer(strength=force_strength,
                                       j_lo=0.05, j_peak=0.15, j_hi=0.5)
        self.lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                                j_act_lo=j_act_lo, j_act_hi=j_act_hi)

    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.forcer.inject(a, **kw)
        return self.lpf.inject(a, **kw)


def main():
    print("=" * 120)
    print("第二十轮：突破天花板的新机制 (BlueprintForcer/HarmonicInjector/PLL)")
    print("=" * 120)
    all_r = {}
    wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}

    # ── T=6400 对照：上轮最优 ──
    class RefAsym(BasePlugin):
        def __init__(self):
            self.cascade = [GoldilocksFusion(**wide) for _ in range(5)]
            self.lpf = AdaptiveLPF(decay=0.3, strength=0.8,
                                    j_act_lo=0.3, j_act_hi=1.0)
        def inject(self, a, **kw):
            for p in self.cascade:
                a = p.inject(a, **kw)
            return self.lpf.inject(a, **kw)
    all_r['R0_ref_T6400'] = run_round(0, "对照 cascade5+LPF T=6400",
        lambda: RefAsym(), T=6400)

    # ── 纯 BlueprintForcer (T=800) 验证基本有效性 ──
    for strength in [0.3, 0.5, 0.7]:
        all_r[f'R1_forcer_s{strength}_T800'] = run_round(1,
            f"BlueprintForcer s={strength} T=800",
            lambda s=strength: BlueprintForcer(strength=s), T=800)

    # ── 纯 HarmonicInjector (T=800) ──
    for strength in [0.3, 0.5]:
        all_r[f'R2_harmonic_s{strength}_T800'] = run_round(2,
            f"HarmonicInjector s={strength} T=800",
            lambda s=strength: HarmonicInjector(strength=s), T=800)

    # ── 纯 PLL (T=800) ──
    for strength in [0.3, 0.5]:
        all_r[f'R3_pll_s{strength}_T800'] = run_round(3,
            f"PLL s={strength} T=800",
            lambda s=strength: PhaseLockedLoop(strength=s), T=800)

    # ── 纯 MultiScaleBlueprint (T=800) ──
    for h2_amp in [0.05, 0.10]:
        all_r[f'R4_multi_h2{h2_amp}_T800'] = run_round(4,
            f"MultiScaleBlueprint h2={h2_amp} T=800",
            lambda h=h2_amp: MultiScaleBlueprintForcer(h2_amp=h), T=800)

    # ── FamilyAdaptiveForcer (T=800) ──
    all_r['R5_family_T800'] = run_round(5, "FamilyAdaptiveForcer T=800",
        lambda: FamilyAdaptiveForcer(), T=800)

    # ── T=6400 组合：Cascade+LPF+Forcer ──
    for force_strength in [0.3, 0.5, 0.7]:
        all_r[f'R6_combo_f{force_strength}_T6400'] = run_round(6,
            f"Cascade+LPF+Forcer f={force_strength} T=6400",
            lambda f=force_strength: CascadeLPFForcer(force_strength=f),
            T=6400)

    # ── T=6400 FamilyAdaptiveForcer ──
    all_r['R7_family_T6400'] = run_round(7, "FamilyAdaptiveForcer T=6400",
        lambda: FamilyAdaptiveForcer(), T=6400)

    # ── T=6400 组合最优配置 ──
    # 测：Cascade5 + Forcer + LPF 三段
    class BestCombo(BasePlugin):
        def __init__(self):
            self.cascade = [GoldilocksFusion(**wide) for _ in range(5)]
            self.forcer = BlueprintForcer(strength=0.5)
            self.lpf = AdaptiveLPF(decay=0.3, strength=0.8,
                                    j_act_lo=0.3, j_act_hi=1.0)
        def inject(self, a, **kw):
            for p in self.cascade:
                a = p.inject(a, **kw)
            a = self.forcer.inject(a, **kw)
            return self.lpf.inject(a, **kw)
    all_r['R8_best_combo_T6400'] = run_round(8, "最佳组合 T=6400",
        lambda: BestCombo(), T=6400)

    # ── T=12800 极限测试最优组合 ──
    all_r['R9_best_combo_T12800'] = run_round(9, "最佳组合 T=12800(极限)",
        lambda: BestCombo(), T=12800, seeds=SEEDS_3)

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

    out = os.path.join(os.path.dirname(__file__), "stress_test20.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
