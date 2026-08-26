#!/usr/bin/env python3
"""
stress_test16.py — 第十六轮：长时仿真 + 状态反馈注入

新尝试：
  1. T=1600 (双倍长度) - 趋势估计更准
  2. V48 StateFeedbackInjection - 利用 q, dq 观测做反馈注入
  3. V49 PDBlueprintTracker - PD 控制器追踪蓝图位置
  4. V50 CascadePlusStateFeedback - 级联 + 状态反馈
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, LeggedMicroSim
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion
from plugins_v2 import AdaptiveEcho, GatedTidal, AdaptiveKV

SEEDS = [42, 137, 2024, 7777, 314159]
T_LONG = 1600  # 双倍长度


def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)


def eval_plugin(make_fn, family, seeds=SEEDS, T=800):
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
#  V48: 状态反馈注入 (State Feedback Injection)                        #
#  利用 q, dq 观测，做 PD 反馈追踪蓝图                                  #
#  desired_a = blueprint_a + Kp*(blueprint_q - q) + Kd*(blueprint_dq - dq)#
# ──────────────────────────────────────────────────────────────────── #
class StateFeedbackInjection(BasePlugin):
    def __init__(self, kp=10.0, kd=1.0, strength=0.5,
                 j_lo=0.05, j_peak=0.12, j_hi=0.3, dt=0.01):
        self.kp = kp
        self.kd = kd
        self.strength = strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self.dt = dt
        self._jerk_hist = []
        self._last_a = None
        # 蓝图状态（在线积分）
        self._bp_q = None
        self._bp_dq = None

    def _update_blueprint_state(self, bp_a):
        """积分蓝图动作得到蓝图位置和速度。"""
        bp_a = np.asarray(bp_a, dtype=float)
        if self._bp_q is None:
            self._bp_q = np.zeros_like(bp_a)
            self._bp_dq = np.zeros_like(bp_a)
        self._bp_dq = self._bp_dq + bp_a * self.dt
        self._bp_q = self._bp_q + self._bp_dq * self.dt

    def inject(self, a, **kw):
        bp_a = kw.get('blueprint', None)
        q = kw.get('q', None)
        dq = kw.get('dq', None)
        if bp_a is None or q is None or dq is None:
            return a

        self._update_blueprint_state(bp_a)
        bp_a = np.asarray(bp_a, dtype=float)
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)

        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a

        # PD 反馈：desired_a = bp_a + Kp*(bp_q - q) + Kd*(bp_dq - dq)
        desired = bp_a + self.kp * (self._bp_q - q) + self.kd * (self._bp_dq - dq)
        return a + self.strength * gate * (desired - a)


# ──────────────────────────────────────────────────────────────────── #
#  V49: 级联 + 状态反馈                                                 #
# ──────────────────────────────────────────────────────────────────── #
class CascadePlusStateFeedback(BasePlugin):
    def __init__(self, n_passes=7, kp=10.0, kd=1.0, fb_strength=0.3,
                 j_lo=0.05, j_peak=0.12, j_hi=0.3,
                 cascade_j_peak=0.15, cascade_j_hi=0.5):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': cascade_j_peak, 'j_hi': cascade_j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_passes)]
        self.fb = StateFeedbackInjection(kp=kp, kd=kd, strength=fb_strength,
                                            j_lo=j_lo, j_peak=j_peak, j_hi=j_hi)
    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.fb.inject(a, **kw)
        return a


# ──────────────────────────────────────────────────────────────────── #
#  V50: 仅蓝图 PD 追踪 (无级联)                                        #
# ──────────────────────────────────────────────────────────────────── #
class PDBlueprintTracker(BasePlugin):
    def __init__(self, kp=5.0, kd=0.5, strength=1.0,
                 j_lo=0.01, j_peak=0.15, j_hi=0.5, dt=0.01):
        self.kp = kp
        self.kd = kd
        self.strength = strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self.dt = dt
        self._jerk_hist = []
        self._last_a = None
        self._bp_q = None
        self._bp_dq = None

    def _update_blueprint_state(self, bp_a):
        bp_a = np.asarray(bp_a, dtype=float)
        if self._bp_q is None:
            self._bp_q = np.zeros_like(bp_a)
            self._bp_dq = np.zeros_like(bp_a)
        self._bp_dq = self._bp_dq + bp_a * self.dt
        self._bp_q = self._bp_q + self._bp_dq * self.dt

    def inject(self, a, **kw):
        bp_a = kw.get('blueprint', None)
        q = kw.get('q', None)
        dq = kw.get('dq', None)
        if bp_a is None or q is None or dq is None:
            return a
        self._update_blueprint_state(bp_a)
        bp_a = np.asarray(bp_a, dtype=float)
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        j = measure_jerk(self, a)
        gate = goldilocks_gate(j, self.j_lo, self.j_peak, self.j_hi)
        if gate < 0.01:
            return a
        desired = bp_a + self.kp * (self._bp_q - q) + self.kd * (self._bp_dq - dq)
        return a + self.strength * gate * (desired - a)


def main():
    print("=" * 120)
    print("第十六轮：长时仿真(T=1600) + 状态反馈注入")
    print("=" * 120)
    all_r = {}
    wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': 0.15, 'j_hi': 0.5}

    # ── 对照：T=800 级联7宽门 ──
    class RefCascade(BasePlugin):
        def __init__(self):
            self.passes = [GoldilocksFusion(**wide) for _ in range(7)]
        def inject(self, a, **kw):
            for p in self.passes:
                a = p.inject(a, **kw)
            return a
    all_r['R0_cascade7_T800_ref'] = run_round(0, "级联7宽门 T=800(对照)",
        lambda: RefCascade(), T=800)

    # ── T=1600 级联7宽门 ──
    all_r['R1_cascade7_T1600'] = run_round(1, "级联7宽门 T=1600(长时)",
        lambda: RefCascade(), T=1600)

    # ── T=1600 级联5宽门 ──
    class Cascade5(BasePlugin):
        def __init__(self):
            self.passes = [GoldilocksFusion(**wide) for _ in range(5)]
        def inject(self, a, **kw):
            for p in self.passes:
                a = p.inject(a, **kw)
            return a
    all_r['R2_cascade5_T1600'] = run_round(2, "级联5宽门 T=1600",
        lambda: Cascade5(), T=1600)

    # ── V48 状态反馈 单独 ──
    all_r['R3_state_fb_kp10_kd1'] = run_round(3, "状态反馈 kp=10 kd=1 str=0.5",
        lambda: StateFeedbackInjection(kp=10.0, kd=1.0, strength=0.5))
    all_r['R4_state_fb_kp5_kd05'] = run_round(4, "状态反馈 kp=5 kd=0.5 str=0.5",
        lambda: StateFeedbackInjection(kp=5.0, kd=0.5, strength=0.5))
    all_r['R5_state_fb_kp20_kd2'] = run_round(5, "状态反馈 kp=20 kd=2 str=0.5",
        lambda: StateFeedbackInjection(kp=20.0, kd=2.0, strength=0.5))

    # ── V50 PD 蓝图追踪 单独 ──
    all_r['R6_pd_tracker_kp5'] = run_round(6, "PD追踪 kp=5 kd=0.5 str=1.0",
        lambda: PDBlueprintTracker(kp=5.0, kd=0.5, strength=1.0))
    all_r['R7_pd_tracker_kp10'] = run_round(7, "PD追踪 kp=10 kd=1 str=1.0",
        lambda: PDBlueprintTracker(kp=10.0, kd=1.0, strength=1.0))
    all_r['R8_pd_tracker_kp20'] = run_round(8, "PD追踪 kp=20 kd=2 str=1.0",
        lambda: PDBlueprintTracker(kp=20.0, kd=2.0, strength=1.0))

    # ── V49 级联 + 状态反馈 ──
    all_r['R9_cascade7_fb_kp10_s03'] = run_round(9, "级联7 + 状态反馈(kp=10 str=0.3)",
        lambda: CascadePlusStateFeedback(n_passes=7, kp=10.0, kd=1.0, fb_strength=0.3))
    all_r['R10_cascade7_fb_kp5_s03'] = run_round(10, "级联7 + 状态反馈(kp=5 str=0.3)",
        lambda: CascadePlusStateFeedback(n_passes=7, kp=5.0, kd=0.5, fb_strength=0.3))
    all_r['R11_cascade7_fb_kp20_s03'] = run_round(11, "级联7 + 状态反馈(kp=20 str=0.3)",
        lambda: CascadePlusStateFeedback(n_passes=7, kp=20.0, kd=2.0, fb_strength=0.3))

    # ── 级联 + 状态反馈 扫 fb_strength ──
    for s in [0.1, 0.2, 0.5, 0.8]:
        all_r[f'R12_cascade7_fb_s{s}'] = run_round(12,
            f"级联7 + 状态反馈(kp=10 str={s})",
            lambda st=s: CascadePlusStateFeedback(n_passes=7, kp=10.0, kd=1.0, fb_strength=st))

    # ── T=1600 级联 + 状态反馈 ──
    all_r['R13_cascade7_fb_T1600'] = run_round(13, "级联7 + 状态反馈 T=1600",
        lambda: CascadePlusStateFeedback(n_passes=7, kp=10.0, kd=1.0, fb_strength=0.3),
        T=1600)

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

    out = os.path.join(os.path.dirname(__file__), "stress_test16.json")
    with open(out, "w") as f:
        json.dump(all_r, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
