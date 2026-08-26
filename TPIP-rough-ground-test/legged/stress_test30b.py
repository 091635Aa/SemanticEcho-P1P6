#!/usr/bin/env python3
"""
stress_test30b.py — 内存优化版：T=2000000, T=2500000 单种子极限

改进：
  - 只存储 q/dq（CI 只需要这两个）
  - 使用 float32 中间存储减少 50% 内存
  - 不存 a/contact/t 全序列（t 只用端点）

目标：
  - 突破 T=1638400 的 +0.6564，尽量上冲至 +0.66+
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, GaitPhaseBlueprint
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion

SEEDS_1 = [42]


def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)


def run_sim_memopt(base, plugins, T, seed, goal=3.0, terrain=0.3, dt=0.01):
    """内存优化版仿真：只用 q_arr/dq_arr float32。"""
    n = base.n_joints
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 0.05, n).astype(np.float64)
    dq = np.zeros(n, dtype=np.float64)
    blueprint = GaitPhaseBlueprint(n)
    # 预分配 float32 数组（节省内存）
    q_arr = np.zeros((T, n), dtype=np.float32)
    dq_arr = np.zeros((T, n), dtype=np.float32)
    for step in range(T):
        t = step * dt
        obs = np.concatenate([q, dq, [goal, terrain]])
        a = base.forward(obs, t)
        bp = blueprint.action(t)
        ct = blueprint.contact(t)
        for plug in plugins:
            a = plug.inject(a, t=t, q=q, dq=dq, blueprint=bp,
                            contact=ct, terrain=terrain,
                            progress=step / T)
        a_clamped = np.clip(a, -0.6, 0.6)
        q_new = q * 0.75 + a_clamped * 0.25
        q_new = np.clip(q_new, -np.pi, np.pi)
        dq = (q_new - q) / dt
        q = q_new
        q_arr[step] = q
        dq_arr[step] = dq
    # CI 计算需要 float64 — 转回来
    return {
        "q": q_arr.astype(np.float64),
        "dq": dq_arr.astype(np.float64),
        "dt": dt,
    }


def eval_plugin_memopt(make_fn, family, seeds, T=800):
    opts = []
    for seed in seeds:
        base = BasePolicy(n_joints=6, family=family, seed=seed)
        t = run_sim_memopt(base, plugins=[], T=T, seed=seed)
        ci_b = compute_coherence(t['q'], t['dq'],
                                 central_diff(t['dq'], t['dt']),
                                 dt=t['dt'])['coherence_index']
        plug = make_fn()
        t2 = run_sim_memopt(base, plugins=[plug], T=T, seed=seed)
        ci_p = compute_coherence(t2['q'], t2['dq'],
                                 central_diff(t2['dq'], t2['dt']),
                                 dt=t2['dt'])['coherence_index']
        opts.append(opt(ci_p, ci_b))
    return float(np.mean(opts)), float(np.std(opts))


def run_round(round_num, name, make_fn, T, seeds=SEEDS_1, all_r=None, out_path=None):
    results = {}
    for fam in ['p2p', 'standard', 'transformer']:
        m, s = eval_plugin_memopt(make_fn, fam, seeds, T=T)
        results[fam] = {'mean': m, 'std': s}
    uni = all(results[f]['mean'] > 0 for f in ['p2p', 'standard', 'transformer'])
    results['universal'] = uni
    tag = "OK" if uni else "--"
    avg = (results['standard']['mean'] + results['transformer']['mean']) / 2
    print(f"  R{round_num}: {name:50s}  std={results['standard']['mean']:+.4f}  trans={results['transformer']['mean']:+.4f}  avg={avg:+.4f}  p2p={results['p2p']['mean']:+.4f}  {tag}", flush=True)
    if all_r is not None:
        all_r[f'R{round_num}_{name.replace(" ", "_").replace("=", "")}'] = results
        if out_path:
            with open(out_path, "w") as f:
                json.dump(all_r, f, ensure_ascii=False, indent=2)
    return results


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
        if j < self.j_act_lo: return 0.0
        if j >= self.j_act_hi: return 1.0
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


class KalmanSmoother(BasePlugin):
    def __init__(self, process_var=0.01, meas_var=0.05, strength=0.7,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5):
        self.process_var = process_var
        self.meas_var = meas_var
        self.strength = strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._x_hat = None
        self._p = None
        self._jerk_hist = []
        self._last_a = None

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        if self._x_hat is None:
            self._x_hat = a.copy()
            self._p = np.ones_like(a) * self.meas_var
            return a
        if self._last_a is None:
            self._last_a = a.copy()
            self._jerk_hist.append(0.0)
        else:
            j = float(np.linalg.norm(a - self._last_a))
            self._last_a = a.copy()
            self._jerk_hist.append(j)
            if len(self._jerk_hist) > 20:
                self._jerk_hist.pop(0)
        j_val = float(np.mean(self._jerk_hist[-3:])) if len(self._jerk_hist) >= 3 else 0.0

        self._p = self._p + self.process_var
        K = self._p / (self._p + self.meas_var)
        self._x_hat = self._x_hat + K * (a - self._x_hat)
        self._p = (1 - K) * self._p

        if j_val < self.j_lo or j_val > self.j_hi:
            gate = 0.0
        elif j_val <= self.j_peak:
            t = (j_val - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)
            gate = float(np.sin(t * np.pi / 2))
        else:
            t = (j_val - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)
            gate = float(np.cos(t * np.pi / 2))

        if gate < 0.01:
            return a
        return a + self.strength * gate * (self._x_hat - a)


class BestCombo(BasePlugin):
    def __init__(self, n_cascade=7, kalman_s=0.7,
                 lpf_decay=0.3, lpf_strength=0.8,
                 j_peak=0.15, j_hi=0.5,
                 j_act_lo=0.3, j_act_hi=1.0):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_cascade)]
        self.kalman = KalmanSmoother(strength=kalman_s)
        self.lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                                j_act_lo=j_act_lo, j_act_hi=j_act_hi)

    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.kalman.inject(a, **kw)
        return self.lpf.inject(a, **kw)


def main():
    out = os.path.join(os.path.dirname(__file__), "stress_test30b.json")
    print("=" * 120)
    print("第三十轮 b：内存优化版 T=2M + T=2.5M (cascade7+K+LPF)")
    print("=" * 120, flush=True)
    all_r = {}

    # 先做 T=2000000
    run_round(0, "cascade7+K+LPF T=2000000", lambda: BestCombo(n_cascade=7),
              T=2000000, seeds=SEEDS_1, all_r=all_r, out_path=out)

    # T=2500000
    run_round(1, "cascade7+K+LPF T=2500000", lambda: BestCombo(n_cascade=7),
              T=2500000, seeds=SEEDS_1, all_r=all_r, out_path=out)

    print(f"\n{'='*120}")
    print("本次结果汇总")
    print(f"{'='*120}")
    sorted_r = sorted(all_r.items(),
                      key=lambda x: (x[1]['standard']['mean'] + x[1]['transformer']['mean']) / 2,
                      reverse=True)
    print(f"  {'配置':48s}  {'std':>8s}  {'trans':>8s}  {'avg':>8s}  {'p2p':>8s}  通用?")
    print("  " + "-" * 90)
    for k, v in sorted_r:
        uni = v.get('universal', False)
        avg = (v['standard']['mean'] + v['transformer']['mean']) / 2
        print(f"  {k:48s}  {v['standard']['mean']:+.4f}  {v['transformer']['mean']:+.4f}  {avg:+.4f}  {v['p2p']['mean']:+.4f}  {'YES' if uni else 'NO'}")

    best = sorted_r[0]
    best_avg = (best[1]['standard']['mean'] + best[1]['transformer']['mean']) / 2
    print(f"\n  本次最优: {best[0]} → avg={best_avg:+.4f}")
    print(f"  历史最优: T=1638400 → avg=+0.6564 (+65.64%)")
    print(f"  结果已写入 {out}")


if __name__ == "__main__":
    main()
