#!/usr/bin/env python3
"""
stress_test18.py — 第十八轮：P2P 修复 + 极致突破

stress_test17 结果：
  R8 cascade5 T=3200 → avg=+0.3805, std=+0.3718, trans=+0.3892, p2p=-0.0133
  → avg 已突破 38%，但 p2p 略负

诊断：
  P2P 基座 jerk 极高 → Goldilocks 门控(j_hi=0.5)截断 → 仅 KV 轻注入
  KV 太弱无法抑制 P2P 的高频抖振(23.7Hz)与阶跃量化
  → 需要专用"高频噪声杀手"：低通滤波器(LPF)

新策略：
  1. ExponentialLPF：指数衰减平滑动作序列，直接压制高频
  2. AdaptiveLPF：jerk 自适应门控——只在极高 jerk(P2P特征)激活
  3. Cascade5 + LPF 组合：级联处理 std/trans，LPF 处理 p2p
  4. T=6400 极长仿真：进一步延长趋势估计
  5. 强度扫描：LPF decay=0.2/0.3/0.4
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
#  V51: 指数低通滤波 (Exponential LPF)                              #
#  a_smoothed = decay * a_history[-1] + (1-decay) * a                #
#  decay=0.3 → 30% 旧 + 70% 新（轻度平滑，防 P2P 颤振）            #
# ──────────────────────────────────────────────────────────────────── #
class ExponentialLPF(BasePlugin):
    def __init__(self, decay=0.3, strength=1.0):
        self.decay = decay
        self.strength = strength
        self._hist = []

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        self._hist.append(a.copy())
        if len(self._hist) > 5:
            self._hist.pop(0)
        if len(self._hist) < 2:
            return a
        # 加权平均：最近的权重最大
        weights = np.array([(1 - self.decay) ** i for i in range(len(self._hist) - 1, -1, -1)])
        weights = weights / weights.sum()
        smoothed = np.average(np.stack(self._hist), axis=0, weights=weights)
        return a + self.strength * (smoothed - a)


# ──────────────────────────────────────────────────────────────────── #
#  V52: 自适应 LPF — 仅在极高 jerk 时激活                           #
#  P2P 的高频抖动会被门控识别并压制，平滑基座不受干扰              #
# ──────────────────────────────────────────────────────────────────── #
class AdaptiveLPF(BasePlugin):
    def __init__(self, decay=0.3, strength=0.8,
                 j_act_lo=0.3, j_act_hi=2.0):
        self.decay = decay
        self.strength = strength
        self.j_act_lo = j_act_lo  # jerk 高于此值才激活 LPF
        self.j_act_hi = j_act_hi  # 完全激活阈值
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
        """高 jerk 时门控=1，低 jerk 时门控=0"""
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
#  V53: 非对称双分支级联                                           #
#  分支1: Goldilocks 级联处理平滑基座(std/trans)                   #
#  分支2: AdaptiveLPF 处理高 jerk 基座(p2p)                        #
# ──────────────────────────────────────────────────────────────────── #
class AsymmetricCascade(BasePlugin):
    def __init__(self, n_cascade=5, lpf_decay=0.3, lpf_strength=0.8,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5,
                 j_act_lo=0.3, j_act_hi=1.0):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_cascade)]
        self.lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                               j_act_lo=j_act_lo, j_act_hi=j_act_hi)

    def inject(self, a, **kw):
        # 先级联（处理 std/trans）
        for p in self.cascade:
            a = p.inject(a, **kw)
        # 再 LPF（处理 p2p 的高频）
        a = self.lpf.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  V54: 先 LPF 后级联（顺序反转）                                   #
#  先压制高频，让级联能更稳定地工作                                 #
# ──────────────────────────────────────────────────────────────────── #
class LPFThenCascade(BasePlugin):
    def __init__(self, n_cascade=5, lpf_decay=0.3, lpf_strength=0.8,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5,
                 j_act_lo=0.3, j_act_hi=1.0):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_cascade)]
        self.lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                               j_act_lo=j_act_lo, j_act_hi=j_act_hi)

    def inject(self, a, **kw):
        # 先 LPF
        a = self.lpf.inject(a, **kw)
        # 再级联
        for p in self.cascade:
            a = p.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  V55: 中嵌 LPF 的级联                                             #
#  级联中段插入 LPF，让 Goldilocks 与 LPF 交替工作                  #
# ──────────────────────────────────────────────────────────────────── #
class InterleavedCascadeLPF(BasePlugin):
    def __init__(self, n_cascade=5, lpf_decay=0.3, lpf_strength=0.6,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5,
                 j_act_lo=0.3, j_act_hi=1.0,
                 lpf_positions=(2, 4)):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.passes = []
        for i in range(n_cascade):
            self.passes.append(('gold', GoldilocksFusion(**wide)))
            if (i + 1) in lpf_positions:
                self.passes.append(('lpf', AdaptiveLPF(
                    decay=lpf_decay, strength=lpf_strength,
                    j_act_lo=j_act_lo, j_act_hi=j_act_hi)))

    def inject(self, a, **kw):
        for kind, p in self.passes:
            a = p.inject(a, **kw)
        return a


def main():
    print("=" * 120)
    print("第十八轮：P2P 修复(LPF) + 极致突破")
    print("=" * 120)
    all_r = {}
    wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}

    # ── 对照：T=3200 cascade5 (上轮最优) ──
    class RefCascade5(BasePlugin):
        def __init__(self):
            self.passes = [GoldilocksFusion(**wide) for _ in range(5)]
        def inject(self, a, **kw):
            for p in self.passes:
                a = p.inject(a, **kw)
            return a
    all_r['R0_ref_cascade5_T3200'] = run_round(0, "对照 cascade5 T=3200 (上轮最优)",
        lambda: RefCascade5(), T=3200)

    # ── 纯 LPF 测试（基线）：检验 LPF 对各基座的影响 ──
    for decay in [0.2, 0.3, 0.5]:
        all_r[f'R1_pureLPF_d{decay}_T800'] = run_round(1,
            f"纯LPF decay={decay} T=800",
            lambda d=decay: ExponentialLPF(decay=d), T=800)

    # ── AdaptiveLPF 测试：仅高 jerk 激活 ──
    for j_act_lo in [0.2, 0.3, 0.5]:
        all_r[f'R2_adaptiveLPF_jlo{j_act_lo}_T800'] = run_round(2,
            f"自适应LPF j_act_lo={j_act_lo} T=800",
            lambda j=j_act_lo: AdaptiveLPF(decay=0.3, strength=0.8,
                                            j_act_lo=j, j_act_hi=1.0), T=800)

    # ── T=3200 非对称双分支级联：cascade5 + AdaptiveLPF ──
    for lpf_decay, lpf_strength in [(0.3, 0.8), (0.4, 0.6), (0.2, 1.0)]:
        all_r[f'R3_asym_d{lpf_decay}_s{lpf_strength}_T3200'] = run_round(3,
            f"非对称级联 d={lpf_decay} s={lpf_strength} T=3200",
            lambda d=lpf_decay, s=lpf_strength: AsymmetricCascade(
                n_cascade=5, lpf_decay=d, lpf_strength=s), T=3200)

    # ── T=3200 先 LPF 后级联 ──
    for lpf_decay in [0.3, 0.4, 0.5]:
        all_r[f'R4_lpf_first_d{lpf_decay}_T3200'] = run_round(4,
            f"先LPF后级联 d={lpf_decay} T=3200",
            lambda d=lpf_decay: LPFThenCascade(
                n_cascade=5, lpf_decay=d, lpf_strength=0.8), T=3200)

    # ── T=3200 交错级联 ──
    for lpf_positions in [(2,), (2, 4), (1, 3, 5)]:
        all_r[f'R5_interleaved_pos{lpf_positions}_T3200'] = run_round(5,
            f"交错级联 pos={lpf_positions} T=3200",
            lambda pos=lpf_positions: InterleavedCascadeLPF(
                n_cascade=5, lpf_decay=0.3, lpf_strength=0.6,
                lpf_positions=pos), T=3200)

    # ── T=3200 cascade3 + AdaptiveLPF（浅级联） ──
    class Cascade3PlusLPF(BasePlugin):
        def __init__(self, lpf_decay=0.3, lpf_strength=0.8):
            self.cascade = [GoldilocksFusion(**wide) for _ in range(3)]
            self.lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                                   j_act_lo=0.3, j_act_hi=1.0)
        def inject(self, a, **kw):
            for p in self.cascade:
                a = p.inject(a, **kw)
            return self.lpf.inject(a, **kw)
    all_r['R6_cascade3_plus_LPF_T3200'] = run_round(6, "级联3 + LPF T=3200",
        lambda: Cascade3PlusLPF(), T=3200)

    # ── T=6400 极长仿真（只测最优配置以节约时间） ──
    best_so_far = AsymmetricCascade(n_cascade=5, lpf_decay=0.3, lpf_strength=0.8)
    all_r['R7_asym_T6400_extreme'] = run_round(7, "非对称级联 T=6400(极长)",
        lambda: AsymmetricCascade(n_cascade=5, lpf_decay=0.3, lpf_strength=0.8),
        T=6400)

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

    out = os.path.join(os.path.dirname(__file__), "stress_test18.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
