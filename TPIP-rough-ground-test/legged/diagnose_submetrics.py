#!/usr/bin/env python3
"""
diagnose_submetrics.py — 逐族 CI 子指标诊断（换思路第 1 步）

目标：找出 std 族(瓶颈 +57%)落后 trans(+74%) 的根源：
  是 S_smooth 或 P_coinc 的哪个子指标，以及 rms_jerk(全关节) vs gait_scale(仅代表关节)
  是否存在"低方差高频关节拖累 S_smooth"的结构错配。

跑法：T=102400, 1 seed, baseline vs BestCombo，输出每族两大子指标 + 每关节 jerk/方差。
"""
import sys, os, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, GaitPhaseBlueprint
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion

T = 102400
SEED = 42


def run_sim_memopt(base, plugins, T, seed, goal=3.0, terrain=0.3, dt=0.01):
    n = base.n_joints
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 0.05, n).astype(np.float64)
    dq = np.zeros(n, dtype=np.float64)
    blueprint = GaitPhaseBlueprint(n)
    q_arr = np.zeros((T, n), dtype=np.float32)
    dq_arr = np.zeros((T, n), dtype=np.float32)
    a_arr = np.zeros((T, n), dtype=np.float32)
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
        a_arr[step] = a
    return {"q": q_arr.astype(np.float64),
            "dq": dq_arr.astype(np.float64),
            "a": a_arr.astype(np.float64), "dt": dt}


# ---- BestCombo 组件（复用，省略注释） ----
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


def diagnose(family, plugin_name, make_fn=None):
    base = BasePolicy(n_joints=6, family=family, seed=SEED)
    t_b = run_sim_memopt(base, [], T, SEED)
    if make_fn is not None:
        t_p = run_sim_memopt(base, [make_fn()], T, SEED)
    else:
        t_p = t_b

    def metrics(tr):
        qd = central_diff(tr['dq'], tr['dt'])
        r = compute_coherence(tr['q'], tr['dq'], qd, dt=tr['dt'])
        # 逐关节 jerk / 方差
        jerk = central_diff(central_diff(tr['dq'], tr['dt']), tr['dt'])
        per_jerk = np.sqrt(np.mean(jerk ** 2, axis=0))
        per_var = np.var(tr['q'], axis=0)
        return r, per_jerk, per_var

    rb, jb, vb = metrics(t_b)
    rp, jp, vp = metrics(t_p)
    return rb, rp, jb, jp, vb, vp


def main():
    print("=" * 100)
    print(f"逐族 CI 子指标诊断  (T={T}, seed={SEED}, 1 seed)")
    print("=" * 100, flush=True)
    out = {}
    for fam in ['p2p', 'standard', 'transformer']:
        rb, rp, jb, jp, vb, vp = diagnose(fam, "BestCombo", lambda: BestCombo())
        ci_b, ci_p = rb['coherence_index'], rp['coherence_index']
        opt = (ci_p - ci_b) / (1 - ci_b + 1e-9)
        print(f"\n[{fam}]  基线 CI={ci_b:.4f}  插件 CI={ci_p:.4f}  优化率={opt:+.3f}")
        print(f"  baseline: S_smooth={rb['s_smooth']:.3f} (rms_jerk={rb['rms_jerk']:.4f}, gaitscale={rb['gait_jerk_scale']:.4f})  P_coinc={rb['p_phase_coincidence']:.3f}")
        print(f"  plugin  : S_smooth={rp['s_smooth']:.3f} (rms_jerk={rp['rms_jerk']:.4f}, gaitscale={rp['gait_jerk_scale']:.4f})  P_coinc={rp['p_phase_coincidence']:.3f}")
        print(f"  ΔS_smooth={rp['s_smooth']-rb['s_smooth']:+.3f}  ΔP_coinc={rp['p_phase_coincidence']-rb['p_phase_coincidence']:+.3f}")
        # 逐关节
        print(f"  baseline 每关节 jerk: {np.round(jb,4)}   var: {np.round(vb,4)}")
        print(f"  plugin   每关节 jerk: {np.round(jp,4)}   var: {np.round(vp,4)}")
        print(f"  jerk 降幅(全关节均值): {np.mean(jb)-np.mean(jp):+.4f}  最高方差关节 idx={int(np.argmax(vb))}")
        out[fam] = {
            'ci_b': ci_b, 'ci_p': ci_p, 'opt': opt,
            'S_b': rb['s_smooth'], 'S_p': rp['s_smooth'],
            'P_b': rb['p_phase_coincidence'], 'P_p': rp['p_phase_coincidence'],
            'rms_jerk_b': rb['rms_jerk'], 'rms_jerk_p': rp['rms_jerk'],
            'gait_scale_b': rb['gait_jerk_scale'], 'gait_scale_p': rp['gait_jerk_scale'],
        }
    with open(os.path.join(os.path.dirname(__file__), "diagnose_submetrics.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 diagnose_submetrics.json")


if __name__ == "__main__":
    main()