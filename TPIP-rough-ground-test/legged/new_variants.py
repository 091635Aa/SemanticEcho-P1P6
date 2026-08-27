#!/usr/bin/env python3
"""
new_variants.py — 换思路实证筛选（第 31 轮）

诊断结论：std 瓶颈= P_coinc(0.327→插件0.624)，trans P 轻微被平滑磨掉。
本轮筛选"加相位"而非"加平滑"的新机制：
  V1 DriftPreserve   去慢漂移 + 保留轨道能量（不减 global_scale）
  V2 PhaseCrispening  锐化：+λ*(a - ema_veryfast_fgsm) 恢复快尺度相位细节
  V3 PerJointJerk    按关节 jerk 归一化平滑强度（降最差关节）
对照：BestCombo（同时段同 seed 复发比较）
T=51200, seed=42, 1 seed（快速筛）
"""
import sys, os, numpy as np, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, GaitPhaseBlueprint
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion

T = 51200
SEED = 42


def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)


def run_sim(base, plugins, T, seed, goal=3.0, terrain=0.3, dt=0.01):
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
                            contact=ct, terrain=terrain, progress=step / T)
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


class BestCombo(BasePlugin):
    def __init__(self, n_cascade=7, kalman_s=0.7, lpf_decay=0.3, lpf_strength=0.8,
                 j_peak=0.15, j_hi=0.5, j_act_lo=0.3, j_act_hi=1.0):
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


class DriftPreserve(BasePlugin):
    """去慢漂移 + 保留轨道能量（不缩 global_scale）。"""
    def __init__(self, tau=2.0, strength=0.6, dt=0.01,
                 j_lo=0.05, j_peak=0.2, j_hi=0.8):
        self.alpha = dt / (tau + dt)
        self.strength = strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._slow = None
        self._a_sq = 1.0  # 运行 a 的均方（能量）
        self._d_sq = 1.0
        self._n = 0
        self._last_a = None
        self._jerk_hist = []

    def _jerk_gate(self, j):
        if j < self.j_lo: return 1.0
        if j >= self.j_hi: return 0.0
        if j <= self.j_peak:
            return float(np.cos(((j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)) * np.pi / 2))
        return 0.0

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        if self._slow is None:
            self._slow = a.copy(); return a
        self._slow = (1 - self.alpha) * self._slow + self.alpha * a
        if self._last_a is None:
            self._last_a = a.copy(); self._jerk_hist.append(0.0)
        else:
            j = float(np.linalg.norm(a - self._last_a))
            self._last_a = a.copy()
            self._jerk_hist.append(j)
            if len(self._jerk_hist) > 30: self._jerk_hist.pop(0)
        jv = float(np.mean(self._jerk_hist[-3:])) if len(self._jerk_hist) >= 3 else 0.0
        gate = self._jerk_gate(jv)
        # 去慢：delta = a - slow
        delta = a - self._slow
        # 保幅：按能量比例回缩放，使 ||out|| 与 ||a|| 同量级（不缩 global_scale）
        self._n += 1
        self._a_sq = 0.999 * self._a_sq + 0.001 * float(np.mean(a ** 2))
        self._d_sq = 0.999 * self._d_sq + 0.001 * float(np.mean(delta ** 2)) + 1e-9
        scale = float(np.sqrt(self._a_sq / self._d_sq))
        out = a + self.strength * gate * (scale * delta - a)
        return out


class PhaseCrispening(BasePlugin):
    """锐化：加回快尺度细节 (a - ema_veryfast)，恢复被平滑磨掉的相位细节。"""
    def __init__(self, tau_fast=0.02, strength=0.15, dt=0.01,
                 j_lo=0.03, j_peak=0.15, j_hi=0.6):
        self.alpha = dt / (tau_fast + dt)
        self.strength = strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self._fast = None
        self._last_a = None
        self._jerk_hist = []

    def _jerk_gate(self, j):
        if j < self.j_lo: return 0.0
        if j >= self.j_hi: return 1.0
        if j <= self.j_peak:
            return float(np.sin(((j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)) * np.pi / 2))
        return float(np.cos(((j - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)) * np.pi / 2))

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        if self._fast is None:
            self._fast = a.copy(); return a
        self._fast = (1 - self.alpha) * self._fast + self.alpha * a
        if self._last_a is None:
            self._last_a = a.copy(); self._jerk_hist.append(0.0)
        else:
            j = float(np.linalg.norm(a - self._last_a))
            self._last_a = a.copy()
            self._jerk_hist.append(j)
            if len(self._jerk_hist) > 20: self._jerk_hist.pop(0)
        jv = float(np.mean(self._jerk_hist[-3:])) if len(self._jerk_hist) >= 3 else 0.0
        gate = self._jerk_gate(jv)
        detail = a - self._fast   # 高频细节（含 1Hz 快速沿）
        return a + self.strength * gate * detail


class Chain(BasePlugin):
    """顺序执行多个插件（前一个输出喂给后一个）。"""
    def __init__(self, *plugins):
        self.plugins = plugins

    def inject(self, a, **kw):
        for p in self.plugins:
            a = p.inject(a, **kw)
        return a


def eval_plugin(make_fn, family, T=T, seed=SEED):
    base = BasePolicy(n_joints=6, family=family, seed=seed)
    tb = run_sim(base, [], T, seed)
    ci_b = compute_coherence(tb['q'], tb['dq'], central_diff(tb['dq'], tb['dt']), dt=tb['dt'])['coherence_index']
    tp = run_sim(base, [make_fn()], T, seed)
    ci_p = compute_coherence(tp['q'], tp['dq'], central_diff(tp['dq'], tp['dt']), dt=tp['dt'])['coherence_index']
    return opt(ci_p, ci_b)


def run_cfg(name, make_fn):
    r = {}
    for fam in ['p2p', 'standard', 'transformer']:
        r[fam] = eval_plugin(make_fn, fam)
    avg = float((r['standard'] + r['transformer']) / 2)
    uni = bool(bool(r['p2p'] > 0) and bool(r['standard'] > 0) and bool(r['transformer'] > 0))
    print(f"  {name:44s}  std={r['standard']:+.4f}  trans={r['transformer']:+.4f}  avg={avg:+.4f}  p2p={r['p2p']:+.4f}  {'YES' if uni else 'NO'}", flush=True)
    return {'name': name, **r, 'avg': avg, 'universal': uni}


def main():
    out = os.path.join(os.path.dirname(__file__), "new_variants.json")
    print("=" * 108)
    print(f"换思路筛选 (T={T}, seed={SEED}, 1 seed)  对照 BestCombo @ T=102400 seed42 avg=+0.5815")
    print("=" * 108, flush=True)
    all_r = {}
    configs = [
        ("BestCombo (ref)",               lambda: BestCombo()),
        ("BestCombo+DriftPreserve s=0.5", lambda: Chain(BestCombo(), DriftPreserve(strength=0.5))),
        ("BestCombo+DriftPreserve s=0.3", lambda: Chain(BestCombo(), DriftPreserve(strength=0.3))),
        ("BestCombo+Crispening s=0.15",   lambda: Chain(BestCombo(), PhaseCrispening(strength=0.15))),
        ("BestCombo+Crispening s=0.30",   lambda: Chain(BestCombo(), PhaseCrispening(strength=0.30))),
        ("BestCombo+DP0.3+Cr0.15",        lambda: Chain(BestCombo(), DriftPreserve(strength=0.3), PhaseCrispening(strength=0.15))),
    ]
    for i, (name, mk) in enumerate(configs):
        t0 = time.time()
        r = run_cfg(name, mk)
        r['time_s'] = round(time.time() - t0)
        all_r[f'R{i}'] = r
        with open(out, "w") as f:
            json.dump(all_r, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果已写入 {out}")
    best = max(all_r.items(), key=lambda x: x[1]['avg'])
    print(f"本次最优: {best[1]['name']} avg={best[1]['avg']:+.4f}")


if __name__ == "__main__":
    main()