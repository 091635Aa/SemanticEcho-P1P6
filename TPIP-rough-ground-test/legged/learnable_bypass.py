#!/usr/bin/env python3
"""
learnable_bypass.py — 即插即用引导向量注入器 (ULI) 验证 (第 35 轮)

满足约束澄清后的全部要求：
  - 基座冻结、不重训任何策略模型；
  - 只注入"引导向量/推理期低秩权重"（可学习 W_out，LoRA 式）；
  - **单一插件参数**：只在一个"锚定基座"(standard) 离线训一次；
  - 之后同一组权重零样本迁移到 p2p / transformer —— 真正"即插即用"，
    而非"每个方案重训一个模型"。

对照：无参 BestCombo（第 30 轮 +65.88% 的出处之一），同 T 同 seeds 对比。

核心：随机特征(ReLU)岭回归 → 只解输出层 W_out(可学习)，闭式、轻量、稳定。
"""
import sys, os, json, time
from collections import deque
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics.coherence_index import compute_coherence, central_diff
from legged_env import BasePolicy, GaitPhaseBlueprint
from plugins import BasePlugin
from plugins_v8 import GoldilocksFusion
from verify_architecture import BestCombo, AdaptiveLPF  # 复用无参基准

T = 102400          # 评估长度（足够恢复 CI）
T_TRAIN = 120000    # 锚定基座训练长度
SEEDS = [42, 7, 99]
N_J = 6
N_JOINT = 6
HID = 48            # 随机特征宽度


def opt(ci_p, ci_b): return (ci_p - ci_b) / (1 - ci_b + 1e-9)


def run_sim(base, plugins, T, seed, goal=3.0, terrain=0.3, dt=0.01):
    n = base.n_joints
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 0.05, n).astype(np.float64)
    dq = np.zeros(n, dtype=np.float64)
    bp_gen = GaitPhaseBlueprint(n)
    q_arr = np.zeros((T, n), dtype=np.float32)
    dq_arr = np.zeros((T, n), dtype=np.float32)
    for step in range(T):
        t = step * dt
        obs = np.concatenate([q, dq, [goal, terrain]])
        a = base.forward(obs, t)
        bp = bp_gen.action(t)
        ct = bp_gen.contact(t)
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
    return {"q": q_arr.astype(np.float64), "dq": dq_arr.astype(np.float64), "dt": dt}


# --------------------------------------------------------------------------- #
# 即插即用引导向量注入器：基座冻结，仅推理期注入低秩权重 W_out 引导向量       #
# --------------------------------------------------------------------------- #
class ULI(BasePlugin):
    """Universal Learnable Injector。fit 后带可学习权重；冷启动时恒等。"""
    def __init__(self, strength=1.0, W=40, lam=5e-3,
                 j_lo=0.03, j_peak=0.2, j_hi=0.8, g_scale=1.0):
        self.strength = strength
        self.W, self.lam = W, lam
        self.j_lo, self.j_peak, self.j_hi = j_lo, j_peak, j_hi
        self.g_scale = g_scale
        self.rng = np.random.default_rng(0)
        self.R1 = None            # 随机投影 (D_total x HID)
        self.W_out = None         # 可学习低秩输出权重 (HID+N -> n)
        self.fitted = False
        self.buf = deque(maxlen=2000)
        self._jerk_hist = []
        self._last_a = None
        self._t = 0

    def reset(self):
        """清运行时状态（保留已训练权重 W_out/R1），供跨 seed 复用同一插件参数。"""
        self.buf.clear()
        self._jerk_hist = []
        self._last_a = None
        self._t = 0

    def _feat(self, a, bp, hist_mean, t):
        phase = 2 * np.pi * 1.0 * t
        x = np.concatenate([a, bp, hist_mean,
                            [np.sin(phase), np.cos(phase), 1.0]])
        return x.astype(np.float64)

    def _feat_map(self, X):
        """随机特征: [X, ReLU(XR)] -> (N, D+HID)。固定投影、只学输出层 = LoRA 式。"""
        if self.R1 is None:
            self.R1 = self.rng.normal(0, 0.3, (X.shape[1], HID)).astype(np.float64)
        nonlin = np.maximum(0.0, X @ self.R1)
        return np.hstack([X, nonlin])

    def collect_trajectory(self, base, seed, t_len, dt=0.01):
        """在锚定基座上收集"特征 -> 理想连贯目标"训练语料。"""
        rng = np.random.default_rng(seed)
        n = base.n_joints
        q = rng.normal(0, 0.05, n).astype(np.float64)
        dq = np.zeros(n, dtype=np.float64)
        bp_gen = GaitPhaseBlueprint(n)
        raw = []          # 原始动作
        tarr = []         # 时间
        bp_hist = []      # 蓝图
        for step in range(t_len):
            t = step * dt
            obs = np.concatenate([q, dq, [3.0, 0.3]])
            a = base.forward(obs, t)
            bp = bp_gen.action(t)
            a_clamp = np.clip(a, -0.6, 0.6)
            q = np.clip(q * 0.75 + a_clamp * 0.25, -np.pi, np.pi)
            raw.append(a.copy()); tarr.append(t); bp_hist.append(bp.copy())
        raw = np.asarray(raw); bp_hist = np.asarray(bp_hist); tarr = np.asarray(tarr)
        # hist_mean: 每点前 W 步动作窗口均值
        N = len(raw)
        hist_mean = np.zeros_like(raw)
        cs = raw.copy()
        for i in range(N):
            lo = max(0, i - self.W)
            hist_mean[i] = raw[lo:i + 1].mean(axis=0) if (i > 0) else raw[0]
        # 理想连贯目标 = 蓝图对齐强平滑（窗口均值 + 与蓝图方向一致的调制）
        a_target = 0.7 * hist_mean + 0.3 * raw
        return {"raw": raw, "bp": bp_hist, "t": tarr, "target": a_target,
                "hist_mean": hist_mean}

    def fit_from(self, data, stride=2):
        raw, bp, tarr, target, hm = (data["raw"], data["bp"], data["t"],
                                     data["target"], data["hist_mean"])
        idx = np.arange(0, len(raw), stride)
        X = np.stack([self._feat(raw[i], bp[i], hm[i], tarr[i]) for i in idx])
        Y = target[idx]
        self._D = X.shape[1]
        Phi = self._feat_map(X)
        nf = Phi.shape[1]
        W = np.linalg.solve(Phi.T @ Phi + self.lam * np.eye(nf),
                            Phi.T @ Y)
        self.W_out = W
        self.fitted = True
        # 用训练集上的"度"估算幅度缩放，防止注入过冲（自标定，非重训）
        pred = Phi @ W
        denom = np.linalg.norm(pred - raw[idx], axis=1).mean() + 1e-9
        numer = np.linalg.norm(target - raw[idx], axis=1).mean() + 1e-9
        self._amp = float(np.clip(numer / denom, 0.5, 2.0))
        return self._amp

    def _gate(self, j):
        if j < self.j_lo: return 0.0
        if j >= self.j_hi: return 1.0
        if j <= self.j_peak:
            return float(np.sin(((j - self.j_lo) / (self.j_peak - self.j_lo + 1e-9)) * np.pi / 2))
        return float(np.cos(((j - self.j_peak) / (self.j_hi - self.j_peak + 1e-9)) * np.pi / 2))

    def inject(self, a, **kw):
        t = kw.get("t", 0.0)
        bp = kw.get("blueprint", np.zeros(a.shape))
        if self._last_a is None:
            self._last_a = a.copy(); self._jerk_hist.append(0.0)
        else:
            j = float(np.linalg.norm(a - self._last_a)); self._last_a = a.copy()
            self._jerk_hist.append(j)
            if len(self._jerk_hist) > 20: self._jerk_hist.pop(0)
        jv = float(np.mean(self._jerk_hist[-3:])) if len(self._jerk_hist) >= 3 else 0.0
        # 维护动作窗口均值（历史引导）
        self.buf.append(np.asarray(a).copy())
        hm = np.mean(np.stack(list(self.buf))[-self.W:], axis=0) if len(self.buf) else a
        if not self.fitted or self.W_out is None:
            return a
        x = self._feat(np.asarray(a), np.asarray(bp), hm, t)
        phi = self._feat_map(x[None, :])[0]
        guide = phi @ self.W_out
        gate = self._gate(jv)
        return np.asarray(a) + gate * self.strength * self._amp * (guide - np.asarray(a))


def eval_family(family, make_plugin_fn, seed, T=T, dt=0.01):
    base = BasePolicy(n_joints=6, family=family, seed=seed)
    tb = run_sim(base, [], T, seed, dt=dt)
    ci_b = compute_coherence(tb["q"], tb["dq"], central_diff(tb["dq"], tb["dt"]),
                             dt=tb["dt"])["coherence_index"]
    p = make_plugin_fn()
    if hasattr(p, "reset"):
        p.reset()
    tp = run_sim(base, [p], T, seed, dt=dt)
    ci_p = compute_coherence(tp["q"], tp["dq"], central_diff(tp["dq"], tp["dt"]),
                             dt=tp["dt"])["coherence_index"]
    return opt(ci_p, ci_b)


def main():
    out = os.path.join(os.path.dirname(__file__), "learnable_bypass.json")
    print("=" * 108)
    print(f"即插即用引导向量注入器 ULI  验证 (T={T}, seeds={SEEDS}, trained on 'standard' only)")
    print("=" * 108, flush=True)

    # ---- 0) 无参 BestCombo 基准 (同 T) ----
    print("\n[基准] 无参 BestCombo:", flush=True)
    base_res = {}
    for fam in ["standard", "transformer", "p2p"]:
        v = [eval_family(fam, BestCombo, s) for s in SEEDS]
        base_res[fam] = v
        print(f"  {fam:12s} {np.mean(v):+.4f}±{np.std(v):.4f}", flush=True)

    # ---- 1) 在锚定基座 standard 上离线训练一次性插件权重 ----
    print("\n[训练] 锚定基座=standard , 离线圈 t_len=%d" % T_TRAIN, flush=True)
    anchor = BasePolicy(n_joints=6, family="standard", seed=42)
    plug = ULI()
    data = plug.collect_trajectory(anchor, seed=42, t_len=T_TRAIN)
    amp = plug.fit_from(data)
    print(f"  W_out 形状={plug.W_out.shape}, 自标定幅度 amp={amp:.3f}", flush=True)

    # ---- 2) 同一权重零样本迁移到三族 ----
    print("\n[迁移] 单一 ULI 参数 -> 三族 (零样本)", flush=True)
    uli_res = {}
    for fam in ["standard", "transformer", "p2p"]:
        v = [eval_family(fam, lambda: plug, s) for s in SEEDS]
        uli_res[fam] = v
        print(f"  {fam:12s} {np.mean(v):+.4f}±{np.std(v):.4f}", flush=True)

    # ---- 汇总 ----
    def avg_of(r): return [(r["standard"][i] + r["transformer"][i]) / 2 for i in range(3)]
    bb = avg_of(base_res); ub = avg_of(uli_res)
    bb_m, bb_s = np.mean(bb), np.std(bb)
    ub_m, ub_s = np.mean(ub), np.std(ub)
    uni_b = all(float(x) > 0 for rr in base_res.values() for x in rr)
    uni_u = all(float(x) > 0 for rr in uli_res.values() for x in rr)
    res = {
        "T": T, "trained_on": "standard", "seeds": SEEDS,
        "bestcombo": {"avg": round(bb_m, 4), "std": round(bb_s, 4),
                      "families": {k: round(float(np.mean(v)), 4) for k, v in base_res.items()},
                      "universal": uni_b},
        "ULI_one_plugin_zeroshot": {"avg": round(ub_m, 4), "std": round(ub_s, 4),
                                    "W_out": list(plug.W_out.shape), "amp": round(amp, 3),
                                    "families": {k: round(float(np.mean(v)), 4) for k, v in uli_res.items()},
                                    "universal": uni_u},
        "avg_delta_ULI_minus_BestCombo": round(ub_m - bb_m, 4),
    }
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)

    print("\n" + "=" * 108)
    print(f"BestCombo    avg={bb_m:+.4f}±{bb_s:.4f}  universal={'YES' if uni_b else 'NO'}")
    print(f"ULI(单插件/零样本) avg={ub_m:+.4f}±{ub_s:.4f}  universal={'YES' if uni_u else 'NO'}")
    print(f"Δ = {ub_m - bb_m:+.4f}")
    print(f"结果 -> {out}")


if __name__ == "__main__":
    main()