#!/usr/bin/env python3
"""
unitree_actionchunk_test.py — 套用宇树 Action Chunking 思想的推理期对照（第 32 轮）

宇树 UnifoLM-VLA 靠 Action Chunking（动作分块预测）实现"顺滑自然"。
本项目约束=零权重/不重训/推理期注入，故把"分块连贯"平移为推理期电路：
  V1 ChunkAlignSmooth  块内滑窗均值低通（C 步对齐步态相位分段）
  V2 ChunkPlayback     阶段式块播放（每 C 步重算块参考，边界线性混合）——更贴"先预测一段再行动"
  V3 ChunkAlignSmooth  C=50（半周期，宽块）
重点验证：能否在保持 P_coinc（第31轮教训：过度平滑伤 P）的同时抬 S_smooth。
对照：BestCombo 同配置复发（T=51200, seed=42 可比）。
"""
import sys, os, numpy as np, json, time
from collections import deque
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


# ---- BestCombo（复用） ----
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


# ---- Action Chunking 电路 ----
class ActionChunkAlign(BasePlugin):
    """V1/V3: 分块对齐滑窗均值低通。C=块长（步）；滑窗=近 C 步均值作为'块参考'。"""
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


class ActionChunkPlayback(BasePlugin):
    """V2: 阶段式块播放——每 C 步重算一次块参考并保持，块内平滑到参考，块边界线性混合
    （更贴合 VLA '先预测一段再执行'）。块内动作一致→jerk 大降；边界混合→不跳变。"""
    def __init__(self, C=25, strength=0.6, blend=0.3, j_lo=0.03, j_peak=0.2, j_hi=0.8):
        self.C, self.strength, self.blend = C, strength, blend
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self.buf = deque()
        self.ref = None
        self.count = 0
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
        gate = self._gate(jv)
        # 收集块内样本
        self.buf.append(a.copy())
        self.count += 1
        # 每 C 步重算块参考
        if self.count % self.C == 0 and self.ref is not None:
            # 平滑切换到新参考：旧/新混合，避免跳变
            newref = np.mean(np.stack(list(self.buf)), axis=0)
            self.ref = (1 - self.blend) * self.ref + self.blend * newref
            self.buf.clear()
        if self.count < self.C:
            self.ref = np.mean(np.stack(list(self.buf)), axis=0)
        if self.ref is None or self.count < 1:
            return a
        return a + self.strength * gate * (self.ref - a)


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
    print(f"  {name:42s}  std={r['standard']:+.4f}  trans={r['transformer']:+.4f}  avg={avg:+.4f}  p2p={r['p2p']:+.4f}  {'YES' if uni else 'NO'}", flush=True)
    return {'name': name, **r, 'avg': avg, 'universal': uni}


def main():
    out = os.path.join(os.path.dirname(__file__), "unitree_actionchunk_test.json")
    print("=" * 108)
    print(f"套用 Action Chunking 推理期对照 (T={T}, seed={SEED})")
    print("对照基准: new_variants BestCombo @ T=51200 avg=+0.5324  上诉：看能否抬 avg 且不伤 P")
    print("=" * 108, flush=True)
    all_r = {}
    configs = [
        ("BestCombo (ref)",             lambda: BestCombo()),
        ("ref+ChunkAlign C=25 s=0.6",   lambda: Chain(BestCombo(), ActionChunkAlign(C=25, strength=0.6))),
        ("ref+ChunkPlayback C=25 s=0.6",lambda: Chain(BestCombo(), ActionChunkPlayback(C=25, strength=0.6))),
        ("ref+ChunkAlign C=50 s=0.6",   lambda: Chain(BestCombo(), ActionChunkAlign(C=50, strength=0.6))),
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