#!/usr/bin/env python3
"""
verify_architecture.py — 套用架构验证（第 33 轮）

验证目标（对照第 32 轮单种子 ChunkAlign +1.7% 存疑）：
  1. 多种子显著性：ChunkAlign 是否系统性超 BestCombo（avg 更高，非单种子噪声）
  2. 在线步态周期自适应：C 由轨迹实时估计（period//4）而非硬编码 25，验证自适应逻辑成立且稳健
  3. 轻量 WMA 预测补偿（第 5 级）：超前一阶登场补偿，是否额外抬标准 P_coinc

跑法：T=51200, seeds=[42,7,99]，输出每配置 avg 的 mean±std。
"""
import sys, os, numpy as np, json, time
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, GaitPhaseBlueprint
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion

T = 51200
SEEDS = [42, 7, 99]


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


# ---- BestCombo ----
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


class Chain(BasePlugin):
    def __init__(self, *plugins): self.plugins = plugins

    def inject(self, a, **kw):
        for p in self.plugins:
            a = p.inject(a, **kw)
        return a


# ---- 第 4 级：块对齐滑窗（硬编码 C） ----
class ChunkAlign(BasePlugin):
    def __init__(self, C=25, strength=0.6, j_lo=0.03, j_peak=0.2, j_hi=0.8):
        self.C, self.strength = C, strength
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self.buf = deque()
        self._last_a = None
        self._jerk_hist = []

    def _gate(self, j):
        if j < self.j_lo: return 1.0
        if j >= self.j_hi: return 0.0
        if j <= self.j_peak:
            return float(np.cos(((j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)) * np.pi / 2))
        return 0.0

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        if self._last_a is None:
            self._last_a = a.copy(); self._jerk_hist.append(0.0)
        else:
            j = float(np.linalg.norm(a - self._last_a))
            self._last_a = a.copy()
            self._jerk_hist.append(j)
            if len(self._jerk_hist) > 20: self._jerk_hist.pop(0)
        jv = float(np.mean(self._jerk_hist[-3:])) if len(self._jerk_hist) >= 3 else 0.0
        self.buf.append(a.copy())
        if len(self.buf) > self.C: self.buf.popleft()
        if len(self.buf) < 3: return a
        chunk_mean = np.mean(np.stack(self.buf), axis=0)
        gate = self._gate(jv)
        return a + self.strength * gate * (chunk_mean - a)


# ---- 第 4 级自适应：在线估计步态周期，C = period//4 ----
class AdaptiveChunkAlign(BasePlugin):
    def __init__(self, strength=0.6, max_period=600, min_period=20, frac=0.25,
                 j_lo=0.03, j_peak=0.2, j_hi=0.8):
        self.strength = strength
        self.max_period, self.min_period = max_period, min_period
        self.frac = frac
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self.sig = deque(maxlen=1200)  # 去 DC 通道（最大方差关节）
        self.C = 25
        self.t = 0
        self._last_a = None
        self._jerk_hist = []
        self.est_period = None

    def _estimate_period(self):
        s = np.array(self.sig)
        if len(s) < 200: return None
        s = s - s.mean()
        # 找正过零点
        zc = np.nonzero((s[:-1] <= 0) & (s[1:] > 0))[0]
        if len(zc) < 3: return None
        diffs = np.diff(zc)
        diffs = diffs[diffs > self.min_period]
        if len(diffs) == 0: return None
        p = int(np.median(diffs))
        return int(np.clip(p, self.min_period, self.max_period))

    def _gate(self, j):
        if j < self.j_lo: return 1.0
        if j >= self.j_hi: return 0.0
        if j <= self.j_peak:
            return float(np.cos(((j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)) * np.pi / 2))
        return 0.0

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        self.t += 1
        # 选最大方差关节做周期估计（用全局缓冲近似：取 track a 的展度）
        if self._last_a is None:
            self._last_a = a.copy(); self._jerk_hist.append(0.0)
        else:
            j = float(np.linalg.norm(a - self._last_a))
            self._last_a = a.copy()
            self._jerk_hist.append(j)
            if len(self._jerk_hist) > 20: self._jerk_hist.pop(0)
        jv = float(np.mean(self._jerk_hist[-3:])) if len(self._jerk_hist) >= 3 else 0.0
        # 周期估计用本步动作范数（保留周期信息够用）
        ch = int(np.argmax(np.var(a, axis=0))) if a.size > 1 else 0
        self.sig.append(float(a[ch]))
        # 周期慢更新（每 ~2s 一次）
        if self.t % 200 == 0:
            p = self._estimate_period()
            if p is not None:
                self.est_period = p
                self.C = max(self.min_period // 4, int(p * self.frac))
        # 应用块对齐滑窗
        buf = list(self.sig)[-(self.C):] if self.C <= len(self.sig) else list(self.sig)
        if len(buf) < 3: return a
        chunk_mean = np.mean(buf, axis=0) * 0.0 / a.size  # 占位
        # 实际用近 C 步动作均值：需要存动作 buffer，这里用 sig 不行。改用内部动作 buffer
        return a  # 由子类逻辑替换（见下方 inject 重建）

    # 实际实现放在这里（上面忽略，正确版本见下方）
    def _real_inject(self, a):
        return a


class AdaptiveChunkAlignV2(BasePlugin):
    """正确实现：用近 C 步动作滑窗均值，C 由在线周期估计决定。"""
    def __init__(self, strength=0.6, min_period=20, frac=0.25,
                 j_lo=0.03, j_peak=0.2, j_hi=0.8):
        self.strength, self.min_period, self.frac = strength, min_period, frac
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self.sig = deque(maxlen=1500)
        self.buf = deque()
        self.C = 25
        self.t = 0
        self._last_a = None
        self._jerk_hist = []
        self.est_period = None

    def _estimate_period(self):
        s = np.array(list(self.sig))
        if len(s) < 200: return None
        s = s - s.mean()
        zc = np.nonzero((s[:-1] <= 0) & (s[1:] > 0))[0]
        if len(zc) < 3: return None
        diffs = np.diff(zc)
        diffs = diffs[diffs > self.min_period]
        if len(diffs) == 0: return None
        return int(np.clip(int(np.median(diffs)), self.min_period, 1000))

    def _gate(self, j):
        if j < self.j_lo: return 1.0
        if j >= self.j_hi: return 0.0
        if j <= self.j_peak:
            return float(np.cos(((j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)) * np.pi / 2))
        return 0.0

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        self.t += 1
        if self._last_a is None:
            self._last_a = a.copy(); self._jerk_hist.append(0.0)
        else:
            j = float(np.linalg.norm(a - self._last_a))
            self._last_a = a.copy()
            self._jerk_hist.append(j)
            if len(self._jerk_hist) > 20: self._jerk_hist.pop(0)
        jv = float(np.mean(self._jerk_hist[-3:])) if len(self._jerk_hist) >= 3 else 0.0
        ch = int(np.argmax(np.var(a, axis=0))) if a.size > 1 else 0
        self.sig.append(float(a[ch]))
        if self.t % 200 == 0:
            p = self._estimate_period()
            if p is not None:
                self.est_period = p
                self.C = max(self.min_period // 4, int(p * self.frac))
        self.buf.append(a.copy())
        if len(self.buf) > self.C: self.buf.popleft()
        if len(self.buf) < 3: return a
        chunk_mean = np.mean(np.stack(self.buf), axis=0)
        gate = self._gate(jv)
        return a + self.strength * gate * (chunk_mean - a)


# ---- 第 5 级：轻量 WMA 预测补偿（超前一阶） ----
class WMAForwardPredict(BasePlugin):
    def __init__(self, beta=0.08, j_lo=0.03, j_peak=0.2, j_hi=0.8):
        self.beta = beta
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self.prev = None
        self._jerk_hist = []
        self._last_a = None

    def _gate(self, j):
        if j < self.j_lo: return 0.0
        if j >= self.j_hi: return 1.0
        if j <= self.j_peak:
            return float(np.sin(((j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)) * np.pi / 2))
        return float(np.cos(((j - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)) * np.pi / 2))

    def inject(self, a, **kw):
        a = np.asarray(a, dtype=float)
        if self._last_a is None:
            self._last_a = a.copy(); self._jerk_hist.append(0.0)
        else:
            j = float(np.linalg.norm(a - self._last_a))
            self._last_a = a.copy()
            self._jerk_hist.append(j)
            if len(self._jerk_hist) > 20: self._jerk_hist.pop(0)
        jv = float(np.mean(self._jerk_hist[-3:])) if len(self._jerk_hist) >= 3 else 0.0
        if self.prev is None:
            self.prev = a.copy(); return a
        delta = a - self.prev
        self.prev = a.copy()
        gate = self._gate(jv)
        return a + self.beta * gate * delta


def eval_family(family, make_fn, seed, T=T):
    base = BasePolicy(n_joints=6, family=family, seed=seed)
    tb = run_sim(base, [], T, seed)
    ci_b = compute_coherence(tb['q'], tb['dq'], central_diff(tb['dq'], tb['dt']), dt=tb['dt'])['coherence_index']
    tp = run_sim(base, [make_fn()], T, seed)
    ci_p = compute_coherence(tp['q'], tp['dq'], central_diff(tp['dq'], tp['dt']), dt=tp['dt'])['coherence_index']
    return opt(ci_p, ci_b)


def run_cfg(name, make_fn):
    std = [eval_family('standard', make_fn, s) for s in SEEDS]
    trans = [eval_family('transformer', make_fn, s) for s in SEEDS]
    p2p = [eval_family('p2p', make_fn, s) for s in SEEDS]
    avgs = [(std[i] + trans[i]) / 2 for i in range(len(SEEDS))]
    m = float(np.mean(avgs)); sd = float(np.std(avgs))
    uni = all(all(float(x) > 0 for x in xs) for xs in [std, trans, p2p])
    print(f"  {name:40s}  avg={m:+.4f}±{sd:.4f}  std=({np.mean(std):+.3f}) trans=({np.mean(trans):+.3f}) p2p=({np.mean(p2p):+.3f})  {m:+.4f}  {'YES' if uni else 'NO'}", flush=True)
    return {'name': name, 'avg_mean': m, 'avg_std': sd, 'avg_each': avgs,
            'std': float(np.mean(std)), 'trans': float(np.mean(trans)), 'p2p': float(np.mean(p2p)),
            'universal': uni}


def main():
    out = os.path.join(os.path.dirname(__file__), "verify_architecture.json")
    print("=" * 112)
    print(f"套用架构验证 (T={T}, seeds={SEEDS}) — 多种子显著性与在线自适应")
    print("=" * 112, flush=True)
    all_r = {}
    configs = [
        ("BestCombo (ref)",                 lambda: BestCombo()),
        ("ref+ChunkAlign C=25 s=0.6",       lambda: Chain(BestCombo(), ChunkAlign(C=25, strength=0.6))),
        ("ref+AdaptChunk (在线周期 C=P//4)", lambda: Chain(BestCombo(), AdaptiveChunkAlignV2(strength=0.6))),
        ("ref+ChunkAlign+WMA fwd",          lambda: Chain(BestCombo(), ChunkAlign(C=25, strength=0.6), WMAForwardPredict(beta=0.08))),
    ]
    for i, (name, mk) in enumerate(configs):
        t0 = time.time()
        r = run_cfg(name, mk)
        r['time_s'] = round(time.time() - t0)
        all_r[f'R{i}'] = r
        with open(out, "w") as f:
            json.dump(all_r, f, ensure_ascii=False, indent=2, default=float)
    print("\n— 显著性判定（ref 为基准，平均差 > 0.01 且跨 seed 一致才认'显著'）—")
    ref = all_r['R0']['avg_mean']
    for k, v in all_r.items():
        d = v['avg_mean'] - ref
        tag = '显著↑' if d > 0.01 and v['avg_std'] <= 0.03 else ('持平' if abs(d) <= 0.01 else '显著↓')
        print(f"  {v['name']:40s}  Δavg={d:+.4f}  {tag}")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()