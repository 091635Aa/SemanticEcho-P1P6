#!/usr/bin/env python3
"""
fast_drift_test.py — FastDriftSuppressor 快速验证 (T=204800, 1 seed)

动机：std(标准RL) 基座在主周期步态上叠加了 0.3Hz 慢漂移
      0.03*sin(2π*0.3*t)，干扰步态相图重合度 P_coinc。
      若用长时间常数 EMA 估计并减去慢分量，可专门抬升 std，
      从而拉高 avg=(std+trans)/2，冲击 70%。

组件：FastDriftSuppressor — 挂在 BestCombo(cascade7+K+LPF) 之后。
      对 p2p 用 jerk gate 保护（p2p 重规划跳变是"有意行为"不抑制）。
"""
import sys, os, numpy as np, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, GaitPhaseBlueprint
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion

SEED = 42
T = 204800


def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)


def run_sim_memopt(base, plugins, T, seed, goal=3.0, terrain=0.3, dt=0.01):
    n = base.n_joints
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 0.05, n).astype(np.float64)
    dq = np.zeros(n, dtype=np.float64)
    blueprint = GaitPhaseBlueprint(n)
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
    return {"q": q_arr.astype(np.float64),
            "dq": dq_arr.astype(np.float64), "dt": dt}


class AdaptiveLPF(BasePlugin):
    def __init__(self, decay=0.3, strength=0.8, j_act_lo=0.3, j_act_hi=1.0):
        self.decay, self.strength = decay, strength
        self.j_act_lo, self.j_act_hi = j_act_lo, j_act_hi
        self._hist, self._jerk_hist, self._last_a = [], [], None

    def _measure_jerk(self, a):
        if self._last_a is None:
            self._last_a = a.copy(); return 0.0
        j = float(np.linalg.norm(a - self._last_a))
        self._last_a = a.copy()
        self._jerk_hist.append(j)
        if len(self._jerk_hist) > 20: self._jerk_hist.pop(0)
        if len(self._jerk_hist) < 3: return 0.0
        return float(np.mean(self._jerk_hist))

    def _adaptive_gate(self, j):
        if j < self.j_act_lo: return 0.0
        if j >= self.j_act_hi: return 1.0
        return float((j - self.j_act_lo) / (self.j_act_hi - self.j_act_lo + 1e-9))

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        self._hist.append(a.copy())
        if len(self._hist) > 5: self._hist.pop(0)
        if len(self._hist) < 2: return a
        j = self._measure_jerk(a)
        gate = self._adaptive_gate(j)
        if gate < 0.01: return a
        weights = np.array([(1 - self.decay) ** i for i in range(len(self._hist) - 1, -1, -1)])
        weights = weights / weights.sum()
        smoothed = np.average(np.stack(self._hist), axis=0, weights=weights)
        return a + self.strength * gate * (smoothed - a)


class KalmanSmoother(BasePlugin):
    def __init__(self, process_var=0.01, meas_var=0.05, strength=0.7,
                 j_lo=0.05, j_peak=0.15, j_hi=0.5):
        self.process_var, self.meas_var, self.strength = process_var, meas_var, strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._x_hat, self._p, self._jerk_hist, self._last_a = None, None, [], None

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        if self._x_hat is None:
            self._x_hat = a.copy(); self._p = np.ones_like(a) * self.meas_var
            return a
        if self._last_a is None:
            self._last_a = a.copy(); self._jerk_hist.append(0.0)
        else:
            j = float(np.linalg.norm(a - self._last_a))
            self._last_a = a.copy()
            self._jerk_hist.append(j)
            if len(self._jerk_hist) > 20: self._jerk_hist.pop(0)
        j_val = float(np.mean(self._jerk_hist[-3:])) if len(self._jerk_hist) >= 3 else 0.0
        self._p = self._p + self.process_var
        K = self._p / (self._p + self.meas_var)
        self._x_hat = self._x_hat + K * (a - self._x_hat)
        self._p = (1 - K) * self._p
        if j_val < self.j_lo or j_val > self.j_hi: gate = 0.0
        elif j_val <= self.j_peak:
            gate = float(np.sin(((j_val - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)) * np.pi / 2))
        else:
            gate = float(np.cos(((j_val - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)) * np.pi / 2))
        if gate < 0.01: return a
        return a + self.strength * gate * (self._x_hat - a)


class FastDriftSuppressor(BasePlugin):
    """慢漂移抑制器：EMA 估计慢分量并减去。

    a' = a - strength * gate * (a - ema)?? 不对——
    应减去 ema 本身的低频跟踪（如果 ema 跟踪到慢分量则 a - ema 去掉 DC+慢漂移）
    采用 out = a - strength*gate*slow，其中 slow = ema(α=dt/τ)。
    为保证周期主成分不被误减：ema 时间常数 τ 应 ≈ 慢漂移周期的同量级（3.3s→τ=2~5s），
    此时 1Hz 周期信号经 EMA 后幅值衰减 >95%（几乎只剩慢分量）。
    """
    def __init__(self, tau=2.0, strength=0.5, dt=0.01,
                 j_gate_lo=0.05, j_gate_peak=0.2, j_gate_hi=0.8):
        self.alpha = dt / (tau + dt)  # EMA 系数 α = dt/τ
        self.strength = strength
        self.j_gate_lo, self.j_gate_peak, self.j_gate_hi = j_gate_lo, j_gate_peak, j_gate_hi
        self._ema = None
        self._last_a = None
        self._jerk_hist = []

    def _jerk_gate(self, j):
        """jerk 低→大 probability 有慢漂移需抑制；jerk 极高（p2p 跳变）→ gate 降级保护"""
        if j < self.j_gate_lo: return 1.0
        if j >= self.j_gate_hi: return 0.0
        if j <= self.j_gate_peak:
            return float(np.cos(((j - self.j_gate_lo) / (self.j_gate_peak - self.j_gate_lo + 1e-9)) * np.pi / 2))
        return 0.0

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        if self._ema is None:
            self._ema = a.copy()
            return a
        self._ema = (1 - self.alpha) * self._ema + self.alpha * a
        # jerk 估计
        if self._last_a is None:
            self._last_a = a.copy(); self._jerk_hist.append(0.0)
        else:
            j = float(np.linalg.norm(a - self._last_a))
            self._last_a = a.copy()
            self._jerk_hist.append(j)
            if len(self._jerk_hist) > 20: self._jerk_hist.pop(0)
        j_val = float(np.mean(self._jerk_hist[-3:])) if len(self._jerk_hist) >= 3 else 0.0
        gate = self._jerk_gate(j_val)
        slow = self._ema  # EMA 输出 = 慢分量估计
        if gate < 0.01: return a
        return a - self.strength * gate * slow


class BestCombo(BasePlugin):
    def __init__(self, n_cascade=7, kalman_s=0.7,
                 lpf_decay=0.3, lpf_strength=0.8,
                 j_peak=0.15, j_hi=0.5,
                 j_act_lo=0.3, j_act_hi=1.0,
                 drift_tau=0.0, drift_strength=0.0):
        wide = {'lam': 1.0, 'alpha': 1.0, 'j_peak': j_peak, 'j_hi': j_hi}
        self.cascade = [GoldilocksFusion(**wide) for _ in range(n_cascade)]
        self.kalman = KalmanSmoother(strength=kalman_s)
        self.lpf = AdaptiveLPF(decay=lpf_decay, strength=lpf_strength,
                               j_act_lo=j_act_lo, j_act_hi=j_act_hi)
        self.drift = None
        if drift_tau > 0:
            self.drift = FastDriftSuppressor(tau=drift_tau, strength=drift_strength)

    def inject(self, a, **kw):
        for p in self.cascade:
            a = p.inject(a, **kw)
        a = self.kalman.inject(a, **kw)
        a = self.lpf.inject(a, **kw)
        if self.drift is not None:
            a = self.drift.inject(a, **kw)
        return a


def eval_plugin(make_fn, family, T=T, seed=SEED):
    base = BasePolicy(n_joints=6, family=family, seed=seed)
    t = run_sim_memopt(base, [], T, seed)
    ci_b = compute_coherence(t['q'], t['dq'], central_diff(t['dq'], t['dt']), dt=t['dt'])['coherence_index']
    plug = make_fn()
    t2 = run_sim_memopt(base, [plug], T, seed)
    ci_p = compute_coherence(t2['q'], t2['dq'], central_diff(t2['dq'], t2['dt']), dt=t2['dt'])['coherence_index']
    return opt(ci_p, ci_b)


def run_cfg(name, make_fn):
    results = {}
    for fam in ['p2p', 'standard', 'transformer']:
        results[fam] = eval_plugin(make_fn, fam)
    avg = float((results['standard'] + results['transformer']) / 2)
    uni = bool(bool(results['p2p'] > 0) and bool(results['standard'] > 0) and bool(results['transformer'] > 0))
    print(f"  {name:46s}  std={results['standard']:+.4f}  trans={results['transformer']:+.4f}  avg={avg:+.4f}  p2p={results['p2p']:+.4f}  {'YES' if uni else 'NO'}", flush=True)
    return {'name': name, **results, 'avg': avg, 'universal': uni}


def main():
    out = os.path.join(os.path.dirname(__file__), "fast_drift_test.json")
    print("=" * 110)
    print(f"FastDriftSuppressor 快速验证 (T={T}, seed={SEED}, 1 seed)")
    print("对照：BestCombo baseline @ T=204800 2-seed avg=+0.6503")
    print("=" * 110, flush=True)
    all_r = {}
    configs = [
        ("baseline (no drift)",           lambda: BestCombo()),
        ("drift τ=2.0 s=0.3",             lambda: BestCombo(drift_tau=2.0, drift_strength=0.3)),
        ("drift τ=2.0 s=0.5",             lambda: BestCombo(drift_tau=2.0, drift_strength=0.5)),
        ("drift τ=5.0 s=0.3",             lambda: BestCombo(drift_tau=5.0, drift_strength=0.3)),
        ("drift τ=5.0 s=0.5",             lambda: BestCombo(drift_tau=5.0, drift_strength=0.5)),
        ("drift τ=1.0 s=0.3",             lambda: BestCombo(drift_tau=1.0, drift_strength=0.3)),
    ]
    for i, (name, mk) in enumerate(configs):
        t0 = time.time()
        r = run_cfg(name, mk)
        r['time_s'] = round(time.time() - t0, 1)
        all_r[f'R{i}'] = r
        with open(out, "w") as f:
            json.dump(all_r, f, ensure_ascii=False, indent=2)

    print(f"\n结果已写入 {out}")
    print(f"若 drift 变体 avg 明显高于 baseline 且 std 显著抬升 → 值得继续（冲击 70%）")
    print(f"否则 T-scaling 已近饱和，准备停止并打包交付")


if __name__ == "__main__":
    main()