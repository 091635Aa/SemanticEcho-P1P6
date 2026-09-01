#!/usr/bin/env python3
"""
plugins_robust.py — 噪声鲁棒方案 (Iteration B / 第38轮)

动机（noise_suite 实测）：温和传感器噪声即令优化率从 +0.53 崩塌到 +0.14；
且 ULI 在噪声下几乎无贡献（因只在干净仿真训练）。

修复思路：在平滑底座之外加"去噪前置"，使到达 ULI 的信号恢复近清洁状态，
  从而让已训练的干净 ULI 重新有效 —— 保持"单次训练、即插即用"，不追加重训。

组件：
  1. SpikeGuard      : 中位数/MAD 离群点抑制（杀偶发 votr 噪点）
  2. FastNoiseLPF    : 以"高频能量比"(非绝对值 jerk) 驱动开关的指数平滑，
                       专门捕捉光滑族在观测噪声下被激起的快速颤动
  3. RobustCombo     : BestCombo -> SpikeGuard -> FastNoiseLPF -> ULI(干净训练)
"""
import numpy as np
from collections import deque
from plugins import BasePlugin


class SpikeGuard(BasePlugin):
    """基于中位数 + MAD 的离群点抑制：识别并平坦死点（跳越），不伤步态。"""

    def __init__(self, K=11, k=4.0, floor=1e-3):
        self.K, self.k, self.floor = K, k, floor
        self.buf = deque(maxlen=K)

    def reset(self):
        self.buf.clear()

    def inject(self, a, **kw):
        a = np.asarray(a).copy()
        self.buf.append(a)
        if len(self.buf) < self.K:
            return a
        arr = np.stack(list(self.buf))           # (K,n)
        med = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - med), axis=0)
        sigma = 1.4826 * mad
        sigma = np.maximum(sigma, self.floor)
        mask = np.abs(a - med) > self.k * sigma   # 单步离群
        a[mask] = med[mask]
        return a


class FastNoiseLPF(BasePlugin):
    """以高频能量比驱动开关的指数平滑：高噪声→强平滑，低噪声→几乎不动。"""

    def __init__(self, Win=9, thr=0.12, alpha_min=0.08, alpha_base=0.5, g_scale=1.0):
        self.Win, self.thr = Win, thr
        self.alpha_min, self.alpha_base = alpha_min, alpha_base
        self.g_scale = g_scale
        self.past = deque(maxlen=Win)
        self._last = None
        self._sm = None

    def reset(self):
        self.past.clear(); self._last = None; self._sm = None

    def _highfreq_ratio(self, a):
        if self._last is None:
            return 0.0
        d = np.abs(a - self._last)
        s = np.abs(a).mean() + 1e-6
        r = float(d.mean() / s)
        self.past.append(r)
        return float(np.mean(list(self.past))) if self.past else r

    def inject(self, a, **kw):
        a = np.asarray(a).copy()
        r = self._highfreq_ratio(a)
        if self._sm is None:
            self._sm = a.copy(); self._last = a.copy(); return a
        # 高噪声→alpha 越小→越依赖过去→越平滑
        alpha = float(np.clip(self.alpha_base / (1.0 + r / self.thr),
                              self.alpha_min, 1.0))
        gate = float(np.clip(r / self.thr, 0.0, 1.0)) ** 0.7 * self.g_scale
        new_sm = self._sm + alpha * (a - self._sm)
        self._sm = new_sm
        self._last = a.copy()
        return (a + gate * (new_sm - a)).astype(float)


class RobustCombo(BasePlugin):
    """BestCombo(无参底座) -> SpikeGuard(杀噪点) -> FastNoiseLPF(抗高频噪声) -> ULI(干净引导)。

    注：第38轮实测本设计在 clean 下过度平滑导致崩塌 (avg -0.51)，
        已由 PhaseMatchedSmoother (PMS, 第39轮) 取代为 RobustComboV2。保留仅供对照。
    """

    def __init__(self, combo, uli):
        self.combo = combo
        self.spike = SpikeGuard()
        self.lpf = FastNoiseLPF()
        self.uli = uli

    def reset(self):
        self.spike.reset(); self.lpf.reset()
        if hasattr(self.uli, "reset"):
            self.uli.reset()

    def inject(self, a, **kw):
        a = self.combo.inject(a, **kw)
        a = self.spike.inject(a, **kw)
        a = self.lpf.inject(a, **kw)
        return self.uli.inject(a, **kw)


# --------------------------------------------------------------------------- #
# 第39轮 — PhaseMatchedSmoother (PMS)：相位匹配去噪器                          #
# --------------------------------------------------------------------------- #
class PhaseMatchedSmoother(BasePlugin):
    """
    相位匹配去噪器：在步态基频上的滑窗傅里叶(最小二乘)拟合。

    原理：
      - 基座动作的主导成分为"相干周期步态"(周期已知，频率 freq)。
      - 在滑窗内用基函数 [1, sin(φ), cos(φ)] 做 LS 回归 → 得到该时刻的
        "相干估计" fitted。噪声(非相干高频)与离群噪点落在该低维子空间之外，
        经窗口内平均被大幅抑制。
      - 输出 = raw + gate * (fitted - raw)，gate 由"当前样本残差/动作幅值"驱动。

    关键设计：clean(无噪声)时动作基本落在周期子空间内 → 残差小 → gate≈0
      → 近似恒等，不损伤基座；噪声/噪点 → 残差大 → gate 升高 → 拉向相干估计，
      使到达后续 ULI 的信号恢复近清洁状态（保持单次训练的 ULI 有效）。
    """

    def __init__(self, Win=31, freq=1.0, dt=0.01, g_max=0.9,
                 beta=0.15, beta_max=None, p=0.7, ema_win=0.970):
        self.Win = Win
        self.freq = freq
        self.dt = dt
        self.g_max = g_max
        self.beta = beta
        self.beta_max = beta_max or beta
        self.p = p
        self.ema_win = ema_win            # 残差比率的慢 EMA(抑制瞬时误判)
        k = np.arange(Win) * dt
        phi = 2 * np.pi * freq * k
        self.B = np.stack([np.ones(Win), np.sin(phi), np.cos(phi)], axis=1)  # (Win,3)
        self.BtB_inv = np.linalg.inv(self.B.T @ self.B)
        self.buf = deque(maxlen=Win)
        self._t = 0.0
        self._ema = None

    def reset(self):
        self.buf.clear(); self._t = 0.0; self._ema = None

    def inject(self, a, **kw):
        a = np.asarray(a).copy()
        self.buf.append(a)
        self._t += self.dt
        if len(self.buf) < self.Win:
            return a
        A = np.stack(list(self.buf))                  # (Win,n)
        c = self.BtB_inv @ (self.B.T @ A)             # (3,n) LS 系数
        lk = (self.Win - 1) * self.dt
        phi_c = 2 * np.pi * self.freq * lk
        basis_c = np.array([1.0, np.sin(phi_c), np.cos(phi_c)])
        fitted = basis_c @ c                          # (n,)
        res = a - fitted
        ratio = float(np.linalg.norm(res) / (np.linalg.norm(a) + 1e-9))
        # 慢 EMA：clean 阶段 ratio 低而稳定 → gate 趋近 0；持续噪声才抬 gate
        if self._ema is None:
            self._ema = ratio
        else:
            self._ema = self.ema_win * self._ema + (1 - self.ema_win) * ratio
        r = self._ema
        gate = float(np.clip((r - self.beta) / (self.beta_max - self.beta + 1e-9),
                             0.0, 1.0)) ** self.p * self.g_max
        return (a + gate * (fitted - a)).astype(float)


class RobustComboV2(BasePlugin):
    """第39轮：BestCombo -> PMS(相位匹配去噪) -> ULI(干净训练)。"""

    def __init__(self, combo, uli, **pms_kw):
        self.combo = combo
        self.pms = PhaseMatchedSmoother(**pms_kw)
        self.uli = uli

    def reset(self):
        self.pms.reset()
        if hasattr(self.uli, "reset"):
            self.uli.reset()

    def inject(self, a, **kw):
        a = self.combo.inject(a, **kw)
        a = self.pms.inject(a, **kw)
        return self.uli.inject(a, **kw)


# --------------------------------------------------------------------------- #
# 第40轮 — PMS2：以相位拟合残差的高频能量驱动注入                              #
# --------------------------------------------------------------------------- #
class PhaseMatchSmoother2(BasePlugin):
    """
    PMS2：同上做滑窗相位(LS)拟合得 fitted，但注入门控改为
      "残差 r_t = a - fitted 的逐样本高频能量( |Δr| 的慢 EMA )"。

    动机（第39轮实测）：把 gate 挂在 |残差|/|a| 上会误伤高 jerk 的 p2p
      （其零星重规划跳变 → 残差大 → 常开），而对平滑族在传感器小噪声下
      因残差被相位拟合吸收 → 常关。改为衡量"残差的白噪声持续性"：
      - 白色传感器噪声：每一拍残差都在抖 → |Δr| 持续高 → gate 高 → 注入拟合；
      - p2p 零星重规划：跳变是孤立的，慢 EMA 会把它平均掉 → gate≈0 → 不动。
    输出 = a + gate*(fitted-a)；clean(平滑族) 残差≈0 → 恒等。
    """

    def __init__(self, Win=31, freq=1.0, dt=0.01, g_max=0.6,
                 thr_hf=0.02, thr_shape=1.2, ref=-1, p=1.0, ema_win=0.985):
        self.Win = Win
        self.freq = freq
        self.dt = dt
        self.g_max = g_max
        self.thr_hf = thr_hf
        self.thr_shape = thr_shape
        self.ref = ref
        self.p = p
        self.ema_win = ema_win
        k = np.arange(Win) * dt
        phi = 2 * np.pi * freq * k
        self.B = np.stack([np.ones(Win), np.sin(phi), np.cos(phi)], axis=1)
        self.BtB_inv = np.linalg.inv(self.B.T @ self.B)
        self.buf = deque(maxlen=Win)
        self._t = 0.0
        self._prev_r = None
        self._hf_ema = None

    def reset(self):
        self.buf.clear(); self._t = 0.0; self._prev_r = None; self._hf_ema = None

    def inject(self, a, **kw):
        a = np.asarray(a).copy()
        self.buf.append(a)
        self._t += self.dt
        if len(self.buf) < self.Win:
            return a
        A = np.stack(list(self.buf))
        c = self.BtB_inv @ (self.B.T @ A)
        lk = (self.Win - 1) * self.dt
        phi_c = 2 * np.pi * self.freq * lk
        basis_c = np.array([1.0, np.sin(phi_c), np.cos(phi_c)])
        fitted = basis_c @ c
        r = a - fitted
        rn = np.linalg.norm(r) / (np.linalg.norm(a) + 1e-9)
        # 相位形状相似性：残差是否多为"高频白噪声"（相位匹配良好但幅值抖）
        shape = float(np.clip(rn * self.thr_shape, 0.0, 2.0) ** self.p)
        if self._prev_r is None:
            self._prev_r = r.copy()
            self._hf = 0.0
        else:
            d = np.linalg.norm(r - self._prev_r) / (np.linalg.norm(a) + 1e-9)
            if self._hf_ema is None:
                self._hf_ema = d
            else:
                self._hf_ema = self.ema_win * self._hf_ema + (1 - self.ema_win) * d
            self._prev_r = r.copy()
        hf = self._hf_ema if self._hf_ema is not None else 0.0
        # 白噪声：HF 持续性高；p2p 孤立跳变：HF 低（EMA 拉平）
        n_gate = float(np.clip(hf / self.thr_hf, 0.0, 1.0)) ** self.p
        gate = n_gate * shape * self.g_max
        return (a + gate * (fitted - a)).astype(float)


class RobustComboV3(BasePlugin):
    """第40轮：BestCombo -> PMS2(HF残差门控) -> ULI(干净训练)。"""

    def __init__(self, combo, uli, **pms_kw):
        self.combo = combo
        self.pms2 = PhaseMatchSmoother2(**pms_kw)
        self.uli = uli

    def reset(self):
        self.pms2.reset()
        if hasattr(self.uli, "reset"):
            self.uli.reset()

    def inject(self, a, **kw):
        a = self.combo.inject(a, **kw)
        a = self.pms2.inject(a, **kw)
        return self.uli.inject(a, **kw)


# --------------------------------------------------------------------------- #
# 第42轮 — StateHFBoost (SHPL)：状态高频门控的相位锁定，专克执行噪声可信度衰减 #
# --------------------------------------------------------------------------- #
class StateHFBoost(BasePlugin):
    """
    State-HF-Gated Phase Lock。

    动机（第41轮全T实测）：执行噪声(act_noise)施加在插件之后，插件动作上不可见，
      但它会污染物理状态：dq=(q_new-q)/dt 使其按 1/dt 放大成每拍白噪声 → dq的高频
      持续能量极高。这是唯一能可靠感知执行噪声的旁路信号。
      - p2p 的重规划跳变是孤立、低频的 → 慢 EMA 拉平 → 检测器不触发 → 不动。
      - clean 平滑族 dq 平滑 → 不触发 → 恒等。
      - 执行噪声下平滑族 dq 白噪 → 触发 → 提高"相位锁定注入"强度：把命令拉向
        (与其当前相位/幅值匹配的) 拟合轨迹，用确定性的周期结构重建相干性。

    输出 = a + gate*lam*(fitted - a)；fitted 取自滑窗相位(LS)拟合，与当前幅值匹配。
    """
    def __init__(self, Win=31, freq=1.0, dt=0.01, g_max=1.0, lam=0.5,
                 thr_hf=0.20, p=1.0, ema_win=0.990, vfloor=1e-3):
        self.Win = Win
        self.freq = freq
        self.dt = dt
        self.g_max = g_max
        self.lam = lam
        self.thr_hf = thr_hf
        self.p = p
        self.ema_win = ema_win
        self.vfloor = vfloor
        k = np.arange(Win) * dt
        phi = 2 * np.pi * freq * k
        self.B = np.stack([np.ones(Win), np.sin(phi), np.cos(phi)], axis=1)
        self.BtB_inv = np.linalg.inv(self.B.T @ self.B)
        self.act_buf = deque(maxlen=Win)
        self.dq_buf = deque(maxlen=Win)
        self._prev_dq = None
        self._hf_ema = None
        self._t = 0.0

    def reset(self):
        self.act_buf.clear(); self.dq_buf.clear()
        self._prev_dq = None; self._hf_ema = None; self._t = 0.0

    def _fit_current(self):
        A = np.stack(list(self.act_buf))               # (Win,n)
        c = self.BtB_inv @ (self.B.T @ A)              # (3,n)
        lk = (self.Win - 1) * self.dt
        phi_c = 2 * np.pi * self.freq * lk
        basis_c = np.array([1.0, np.sin(phi_c), np.cos(phi_c)])
        return basis_c @ c

    def inject(self, a, **kw):
        a = np.asarray(a).copy()
        dqv = kw.get("dq", None)
        self.act_buf.append(a)
        self._t += self.dt
        # --- 状态高频(白噪)检测 ------------------------------------------------------------------
        if dqv is not None:
            self.dq_buf.append(np.asarray(dqv).copy())
            if self._prev_dq is None:
                self._prev_dq = np.asarray(dqv).copy()
            else:
                d = np.linalg.norm(np.asarray(dqv) - self._prev_dq) / \
                    (np.linalg.norm(np.asarray(dqv)) + self.vfloor)
                if self._hf_ema is None:
                    self._hf_ema = d
                else:
                    self._hf_ema = self.ema_win * self._hf_ema + (1 - self.ema_win) * d
                self._prev_dq = np.asarray(dqv).copy()
        hf = self._hf_ema if self._hf_ema is not None else 0.0
        # --- 相位锁定注入 ------------------------------------------------------------------------
        if len(self.act_buf) < self.Win or hf <= 0.0:
            return a
        gate = float(np.clip(hf / self.thr_hf, 0.0, 1.0)) ** self.p * self.g_max
        if gate < 1e-3:
            return a
        fitted = self._fit_current()
        return (a + (gate * self.lam) * (fitted - a)).astype(float)


class RobustComboV4(BasePlugin):
    """第42轮：BestCombo -> ULI(干净) -> StateHFBoost(状态白噪检测+相位锁定)。"""

    def __init__(self, combo, uli, **boost_kw):
        self.combo = combo
        self.uli = uli
        self.boost = StateHFBoost(**boost_kw)

    def reset(self):
        self.boost.reset()
        if hasattr(self.uli, "reset"):
            self.uli.reset()

    def inject(self, a, **kw):
        a = self.combo.inject(a, **kw)
        a = self.uli.inject(a, **kw)
        return self.boost.inject(a, **kw)


# --------------------------------------------------------------------------- #
# 第43轮 — DecoupledNoiseBoost (DNB)：信噪比(SNR)机制                         #
# --------------------------------------------------------------------------- #
class DecoupledNoiseBoost(BasePlugin):
    """
    DNB：针对"执行噪声 = 固定方差 σ² 叠加在最终指令上"这条不可逆物理路径，
      不走"把噪声滤掉"(做不到)，而是**抬升相干(周期)能量的信噪比**：
      把确定性步态的相干幅度放大，使它在固定 σ² 噪声地板之上站得更高。

    流程：
      1) 状态白噪检测：dq 的逐样本跳变(∝1/dt 放大的执行噪声) 慢 EMA → hf。
         平滑族在执行噪声下 hf 持续高(白噪)；p2p 重规划跳变是孤立低频 → 低。
      2) gate = clip(hf/thr_hf,0,1)^p * g_max。
      3) 注入：取滑窗相位拟合 fitted(与当前相位/幅值匹配的相干分量)，
         放大 γ = 1 + gammax*gate 倍，向它混合：
            out = a + gate*lam*(fitted*γ - a)
         提升 q 中周期分量振幅 / 噪声方差 = 更高的相干 SNR → 更高 CI。
    """
    def __init__(self, Win=31, freq=1.0, dt=0.01, g_max=1.0, gammax=1.0, lam=0.6,
                 thr_hf=0.06, p=1.0, ema_win=0.980, vfloor=1e-3):
        self.Win = Win
        self.freq = freq
        self.dt = dt
        self.g_max = g_max
        self.gammax = gammax
        self.lam = lam
        self.thr_hf = thr_hf
        self.p = p
        self.ema_win = ema_win
        self.vfloor = vfloor
        k = np.arange(Win) * dt
        phi = 2 * np.pi * freq * k
        self.B = np.stack([np.ones(Win), np.sin(phi), np.cos(phi)], axis=1)
        self.BtB_inv = np.linalg.inv(self.B.T @ self.B)
        self.act_buf = deque(maxlen=Win)
        self._prev_dq = None
        self._hf_ema = None
        self._t = 0.0
        self.gate_sum = 0.0
        self.gate_n = 0

    def reset(self):
        self.act_buf.clear(); self._prev_dq = None; self._hf_ema = None
        self._t = 0.0; self.gate_sum = 0.0; self.gate_n = 0

    @property
    def mean_gate(self):
        return (self.gate_sum / self.gate_n) if self.gate_n else 0.0

    def _fit_current(self):
        A = np.stack(list(self.act_buf))
        c = self.BtB_inv @ (self.B.T @ A)
        lk = (self.Win - 1) * self.dt
        phi_c = 2 * np.pi * self.freq * lk
        basis_c = np.array([1.0, np.sin(phi_c), np.cos(phi_c)])
        return basis_c @ c

    def inject(self, a, **kw):
        a = np.asarray(a).copy()
        dqv = kw.get("dq", None)
        self.act_buf.append(a)
        self._t += self.dt
        if dqv is not None:
            dvv = np.asarray(dqv)
            if self._prev_dq is None:
                self._prev_dq = dvv.copy()
            else:
                d = np.linalg.norm(dvv - self._prev_dq) / (np.linalg.norm(dvv) + self.vfloor)
                if self._hf_ema is None:
                    self._hf_ema = d
                else:
                    self._hf_ema = self.ema_win * self._hf_ema + (1 - self.ema_win) * d
                self._prev_dq = dvv.copy()
        hf = self._hf_ema if self._hf_ema is not None else 0.0
        if len(self.act_buf) < self.Win or hf <= 0.0:
            return a
        gate = float(np.clip(hf / self.thr_hf, 0.0, 1.0)) ** self.p * self.g_max
        self.gate_sum += gate; self.gate_n += 1
        if gate < 1e-3:
            return a
        fitted = self._fit_current()
        gamma = 1.0 + self.gammax * gate
        target = fitted * gamma
        return (a + (gate * self.lam) * (target - a)).astype(float)


class RobustComboV5(BasePlugin):
    """第43轮：BestCombo -> ULI(干净) -> DNB(状态白噪检测 + 相干幅度放大)。"""

    def __init__(self, combo, uli, **dnb_kw):
        self.combo = combo
        self.uli = uli
        self.dnb = DecoupledNoiseBoost(**dnb_kw)

    def reset(self):
        self.dnb.reset()
        if hasattr(self.uli, "reset"):
            self.uli.reset()

    def inject(self, a, **kw):
        a = self.combo.inject(a, **kw)
        a = self.uli.inject(a, **kw)
        return self.dnb.inject(a, **kw)


# --------------------------------------------------------------------------- #
# 第44轮 — DNB2 (DecoupledNoiseBoost v2)：相干形状门控，保护 p2p               #
# --------------------------------------------------------------------------- #
class DecoupledNoiseBoost2(BasePlugin):
    """
    DNB2：把 DNB 的"信噪比放大"只施加到已是"相干步态"的平滑族，从而保护
      p2p(点对点/阶跃体式)。第43轮实测：
        - mild 下 DNB-g2t06 把平滑族增益从 champ +0.11 抬到 +0.25(≈2.2×)，
          信噪比放大机制确实有效；
        - 但 gate(白噪检测)对 p2p 也常开 → 放大动作里不匹配的"拟合正弦"，
          使 p2p 增益从 +0.42 掉到 +0.17，破坏 Universal。
    修复：在已有 hf(状态白噪)门控之外，再加一个"相干形状门"：
        shape_gate(平滑族高 / p2p 低)，由"动作对低阶相位基的残差比"驱动：
          res_ratio = |a - fitted| / |a|
          - 平滑族：动作基本落在大频相位子空间 → res_ratio 小 → shape_gate→1(放大)
          - p2p   ：阶跃/重规划不在此子空间   → res_ratio 大 → shape_gate→0(不放大)
      最终 effective_gate = hf_gate * shape_gate，平滑族保留 SNR 放大收益，
      同时 p2p 退化为"不放大"(只保留 BestCombo+ULI)，恢复 Universal。
    """

    def __init__(self, Win=31, freq=1.0, dt=0.01, g_max=1.0, gammax=1.0, lam=0.6,
                 thr_hf=0.06, p=1.0, ema_win=0.980, vfloor=1e-3,
                 sh_cut=0.5, sh_pow=1.0, sh_ema=0.990, sh_floor=0.0):
        self.Win = Win
        self.freq = freq
        self.dt = dt
        self.g_max = g_max
        self.gammax = gammax
        self.lam = lam
        self.thr_hf = thr_hf
        self.p = p
        self.ema_win = ema_win
        self.vfloor = vfloor
        self.sh_cut = sh_cut        # 残差比 > sh_cut → shape_gate→0 (p2p 不放大)
        self.sh_pow = sh_pow
        self.sh_ema = sh_ema        # 残差比慢 EMA，抗传感器噪声瞬态
        self.sh_floor = sh_floor    # p2p 保留增强下限
        k = np.arange(Win) * dt
        phi = 2 * np.pi * freq * k
        self.B = np.stack([np.ones(Win), np.sin(phi), np.cos(phi)], axis=1)
        self.BtB_inv = np.linalg.inv(self.B.T @ self.B)
        self.act_buf = deque(maxlen=Win)
        self._prev_dq = None
        self._hf_ema = None
        self._res_ema = None
        self._t = 0.0
        self.gate_sum = 0.0
        self.gate_n = 0
        self.gate_shape_sum = 0.0

    def reset(self):
        self.act_buf.clear(); self._prev_dq = None; self._hf_ema = None
        self._res_ema = None
        self._t = 0.0; self.gate_sum = 0.0; self.gate_n = 0
        self.gate_shape_sum = 0.0

    @property
    def mean_gate(self):
        return (self.gate_sum / self.gate_n) if self.gate_n else 0.0

    @property
    def mean_shape_gate(self):
        return (self.gate_shape_sum / self.gate_n) if self.gate_n else 0.0

    def _fit_current(self):
        A = np.stack(list(self.act_buf))
        c = self.BtB_inv @ (self.B.T @ A)
        lk = (self.Win - 1) * self.dt
        phi_c = 2 * np.pi * self.freq * lk
        basis_c = np.array([1.0, np.sin(phi_c), np.cos(phi_c)])
        return basis_c @ c

    def inject(self, a, **kw):
        a = np.asarray(a).copy()
        dqv = kw.get("dq", None)
        self.act_buf.append(a)
        self._t += self.dt
        if dqv is not None:
            dvv = np.asarray(dqv)
            if self._prev_dq is None:
                self._prev_dq = dvv.copy()
            else:
                d = np.linalg.norm(dvv - self._prev_dq) / (np.linalg.norm(dvv) + self.vfloor)
                if self._hf_ema is None:
                    self._hf_ema = d
                else:
                    self._hf_ema = self.ema_win * self._hf_ema + (1 - self.ema_win) * d
                self._prev_dq = dvv.copy()
        hf = self._hf_ema if self._hf_ema is not None else 0.0
        if len(self.act_buf) < self.Win or hf <= 0.0:
            return a
        fitted = self._fit_current()
        res_ratio = float(np.linalg.norm(a - fitted) / (np.linalg.norm(a) + self.vfloor))
        # 残差比慢 EMA：抗传感器噪声瞬态
        if self._res_ema is None:
            self._res_ema = res_ratio
        else:
            self._res_ema = self.sh_ema * self._res_ema + (1 - self.sh_ema) * res_ratio
        rr = self._res_ema
        # 相干形状门(硬阈值)：平滑族 rr 小→高；p2p 阶跃/重规划 rr 大→低
        shape = (1.0 - min(rr / self.sh_cut, 1.0)) ** self.sh_pow
        shape_gate = self.sh_floor + (1.0 - self.sh_floor) * shape
        hf_gate = float(np.clip(hf / self.thr_hf, 0.0, 1.0)) ** self.p * self.g_max
        gate = hf_gate * shape_gate
        self.gate_sum += gate; self.gate_n += 1
        self.gate_shape_sum += shape_gate
        if gate < 1e-3:
            return a
        gamma = 1.0 + self.gammax * gate
        target = fitted * gamma
        return (a + (gate * self.lam) * (target - a)).astype(float)


class RobustComboV6(BasePlugin):
    """第44轮：BestCombo -> ULI(干净) -> DNB2(状态白噪 × 相干形状 双门控 SNR 放大)。"""

    def __init__(self, combo, uli, **dnb_kw):
        self.combo = combo
        self.uli = uli
        self.dnb = DecoupledNoiseBoost2(**dnb_kw)

    def reset(self):
        self.dnb.reset()
        if hasattr(self.uli, "reset"):
            self.uli.reset()

    def inject(self, a, **kw):
        a = self.combo.inject(a, **kw)
        a = self.uli.inject(a, **kw)
        return self.dnb.inject(a, **kw)


# --------------------------------------------------------------------------- #
# 第46轮 — BlueprintAnchor (BAE)：跨关节步态蓝图参照增强                       #
# --------------------------------------------------------------------------- #
class BlueprintAnchor(BasePlugin):
    """
    BAE：修复"逐关节独立拟合平滑"破坏 参照性/联动性 的问题。

    第45轮三维诊断显示：V6(逐关节独立 DNB2 拟合) 大幅抬升单通道 CI，
      却让 参照性 R_ref(逐时刻关节姿态对全局步态蓝图的Pearson相关) 暴跌
      (-0.6~-0.8)，联动性 L_link 也几乎无改观 —— 即"更平滑但不再参照蓝图"。
    根因：每关节被平滑到"自己拟合"的正弦，拟合相位逐关节独立漂移，
      跨关节相位偏移偏离共享蓝图(phase_offsets=linspace(0,π))，交联姿态失序。

    BAE 在窗口内做逐关节 LS 相位拟合判断"相干族"(平滑std/transformer)，再用
    蓝图方向对动作做"交联重投影"，把关节姿态锁回共享蓝图相位结构：
      - d = a - mean; bvn = 蓝图去均值单位方向。
      - align = d·bvn        # 沿蓝图方向的模式幅值
      - target = a.mean() + align*bvn   # 蓝图成形(保留DC与蓝图模幅)
      - out = a + lam*gate*(target - a)
    只有"相干族"才开 gate(逐关节低阶拟合残差小)；p2p 残差大 → gate≈0 → 不动，
    保护 p2p 与噪声增益不复失。
    """
    def __init__(self, Win=31, freq=1.0, dt=0.01, lam=0.4,
                 thr_rr=0.5, p=1.0, sh_ema=0.990, vfloor=1e-3):
        self.Win = Win
        self.freq = freq
        self.dt = dt
        self.lam = lam
        self.thr_rr = thr_rr
        self.p = p
        self.sh_ema = sh_ema
        self.vfloor = vfloor
        # 蓝图相位偏移(与 GaitPhaseBlueprint 一致)
        self.offs = np.linspace(0, np.pi, 6)
        k = np.arange(Win) * dt
        self.phi = 2 * np.pi * freq * k                 # 全局步态相位(蓝图/共享)
        # 蓝图相位基：B = [1, sin(Φ+off_j), cos(Φ+off_j)] per joint
        self.B = np.stack([np.ones(Win),
                           np.sin(self.phi[:, None] + self.offs[None, :]),
                           np.cos(self.phi[:, None] + self.offs[None, :])], axis=0)  # (3,Win,n)
        self.act_buf = deque(maxlen=Win)
        self._res_ema = None
        self._t = 0.0

    def reset(self):
        self.act_buf.clear(); self._res_ema = None; self._t = 0.0

    def inject(self, a, **kw):
        a = np.asarray(a).copy()
        self.act_buf.append(a)
        self._t += self.dt
        if len(self.act_buf) < self.Win:
            return a
        # --- 相干族门：逐关节自由LS相位拟合残差小 → 平滑族 → 高门 ---
        A = np.stack(list(self.act_buf))                # (Win,n)
        Bf = np.stack([np.ones(self.Win),
                       np.sin(self.phi),
                       np.cos(self.phi)], axis=1)       # (Win,3) 自由相位基
        Bf_inv = np.linalg.pinv(Bf)
        c = Bf_inv @ A                                  # (3,n)
        lk = (self.Win - 1) * self.dt
        phi_c = 2 * np.pi * self.freq * lk
        basis_c = np.array([1.0, np.sin(phi_c), np.cos(phi_c)])
        fitted = basis_c @ c                            # (n,)
        rr = float(np.linalg.norm(A[-1] - fitted) / (np.linalg.norm(A[-1]) + self.vfloor))
        if self._res_ema is None:
            self._res_ema = rr
        else:
            self._res_ema = self.sh_ema * self._res_ema + (1 - self.sh_ema) * rr
        coh = float(np.clip(1.0 - min(self._res_ema / self.thr_rr, 1.0), 0.0, 1.0)) ** self.p
        if coh < 1e-3:
            return a
        # --- 逐关节蓝图相位投影(GPS)：把每关节锁定到共享蓝图相位偏移结构 ---
        # 每关节只调"蓝图相位分量幅值"，全局相位 Φ=2πft 共享 → 跨关节偏移保持蓝图
        n = a.shape[0]
        campl = np.zeros(n)
        for j in range(n):
            sb = np.sin(self.phi + self.offs[j])            # (Win,) 该关节蓝图相位正弦
            cb = np.cos(self.phi + self.offs[j])
            xj = A[:, j] - A[:, j].mean()
            # 回归 amplit = (2/Win)·Σ x·sin(Φ+off)；再用 cos 去 DC 相位污染最小化
            c_s = 2.0 / self.Win * (xj @ sb)
            c_c = 2.0 / self.Win * (xj @ cb)
            campl[j] = c_s  # 与 sin 对齐的蓝图幅值(蓝图以 sin 相位定义)
        # 当前时刻蓝图相位分量
        phi_cur = 2 * np.pi * self.freq * (self._t - self.dt)
        target = a.mean(axis=0) if a.ndim > 1 else a.copy()
        target = np.asarray(target, dtype=float)
        if target.ndim == 0:
            target = a.copy()
        blu = campl * np.sin(phi_cur + self.offs)           # 蓝图相位重构
        target = a - a.mean() + blu                         # 保留 DC(a)
        out = a + (self.lam * coh) * (target - a)
        return np.asarray(out, dtype=float)


class RobustComboV7(BasePlugin):
    """第46轮：BestCombo -> ULI(干净) -> DNB2(SNR放大) -> BlueprintAnchor(蓝图参照)。

    在保留 V6 全部噪声增益(平滑族 SNR 放大 + p2p 保护)之上，追加分段蓝图相位
    锁定，回收独立平滑丢失的 参照性/联动性，且蓝图本身是平滑相干信号，不损 CI。
    """

    def __init__(self, combo, uli, bae_lam=0.4, **dnb_kw):
        self.combo = combo
        self.uli = uli
        self.dnb = DecoupledNoiseBoost2(**dnb_kw)
        self.bae = BlueprintAnchor(lam=bae_lam)

    def reset(self):
        self.dnb.reset()
        if hasattr(self.uli, "reset"):
            self.uli.reset()

    def inject(self, a, **kw):
        a = self.combo.inject(a, **kw)
        a = self.uli.inject(a, **kw)
        a = self.dnb.inject(a, **kw)
        return self.bae.inject(a, **kw)


# --------------------------------------------------------------------------- #
# 第47轮 — BlueprintLockedBoost (BLB)：蓝图相位锁定 SNR 放大                    #
# --------------------------------------------------------------------------- #
class BlueprintLockedBoost(BasePlugin):
    """
    BLB：把 DNB2 的"自由相位"相干放大改成"蓝图相位锁定"放大。

    第45/46轮三维诊断的根因：DNB2/PMS 用自由共享相位基 [1,sinφ,cosφ] 重建+放大，
      → 每个关节的 sin/cos 系数比(相位)逐关节漂移，跨关节相位偏移偏离蓝图
      (off_j=linspace(0,π))，导致 参照性 R_ref 从基线 0.93 暴跌到 0.15~0.32。

    修复：相干重建目标永远投影到"蓝图相位基" [sin(Φ+off_j)] 上，全局相位 Φ 共享、
      每关节偏移 off_j 固定(蓝图)。这样：
        - 重建/放大后的信号任何时刻都落在蓝图子空间 → 跨关节偏移保持蓝图 →
          R_ref 保住甚至趋近1，L_link(全关节共享全局相位)也同步抬升；
        - 蓝图正弦本身平滑相干 → 不损 CI；
        - 逐关节残差比门控(平滑族低/p2p高) 保护 p2p；
        - 状态白噪(hf)门控 决定何时放大(执行噪声)。
    输出 = a + lam*gate*( (dc + gamma·campl·sin(Φ_cur+off_j)) - a )
    """
    def __init__(self, Win=31, freq=1.0, dt=0.01, gammax=1.0, lam=0.6,
                 thr_hf=0.06, p=1.0, ema_win=0.980, vfloor=1e-3,
                 rrcut=0.5, rr_pow=1.0, rr_ema=0.990, rr_floor=0.0,
                 gain=1.0):
        self.Win = Win
        self.freq = freq
        self.dt = dt
        self.gammax = gammax
        self.lam = lam
        self.thr_hf = thr_hf
        self.p = p
        self.ema_win = ema_win
        self.vfloor = vfloor
        self.rrcut = rrcut
        self.rr_pow = rr_pow
        self.rr_ema = rr_ema
        self.rr_floor = rr_floor
        self.gain = gain          # 蓝图投影/重建的耦合强度(＝蓝图幅值缩放，正比SNR)
        # 蓝图相位偏移(与 GaitPhaseBlueprint / cdiag 一致)
        self.offs = np.linspace(0, np.pi, 6)
        k = np.arange(Win) * dt
        self.phi = 2 * np.pi * freq * k               # 全局步态相位(共享)
        self.act_buf = deque(maxlen=Win)
        self._prev_dq = None
        self._hf_ema = None
        self._rr_ema = None
        self._t = 0.0
        self.gate_sum = 0.0
        self.gate_n = 0

    def reset(self):
        self.act_buf.clear(); self._prev_dq = None; self._hf_ema = None
        self._rr_ema = None; self._t = 0.0
        self.gate_sum = 0.0; self.gate_n = 0

    @property
    def mean_gate(self):
        return (self.gate_sum / self.gate_n) if self.gate_n else 0.0

    def inject(self, a, **kw):
        a = np.asarray(a).copy()
        dqv = kw.get("dq", None)
        self.act_buf.append(a)
        self._t += self.dt
        # --- 状态白噪(hf)检测：执行噪声∝1/dt放大成逐拍白噪 → 慢EMA高 ---
        if dqv is not None:
            dvv = np.asarray(dqv)
            if self._prev_dq is None:
                self._prev_dq = dvv.copy()
            else:
                d = np.linalg.norm(dvv - self._prev_dq) / (np.linalg.norm(dvv) + self.vfloor)
                if self._hf_ema is None:
                    self._hf_ema = d
                else:
                    self._hf_ema = self.ema_win * self._hf_ema + (1 - self.ema_win) * d
                self._prev_dq = dvv.copy()
        hf = self._hf_ema if self._hf_ema is not None else 0.0
        if len(self.act_buf) < self.Win or hf <= 0.0:
            return a
        A = np.stack(list(self.act_buf))              # (Win,n)
        n = a.shape[0]
        # --- 蓝图锁定相干投影：每关节重建到蓝图子空间 [sin(Φ+off_j)] ---
        campl = np.zeros(n)
        rr_tot = 0.0
        for j in range(n):
            sb = np.sin(self.phi + self.offs[j])       # 该关节蓝图相位正弦 (Win,)
            cb = np.cos(self.phi + self.offs[j])
            xj = A[:, j] - A[:, j].mean()
            c_s = 2.0 / self.Win * (xj @ sb)
            c_c = 2.0 / self.Win * (xj @ cb)
            # 残差比：该关节相对蓝图子空间的重建残差(平滑族低/p2p高)
            blu = c_s * sb                              # 蓝图重建分量(不含DC/正交分量)
            rj2 = np.linalg.norm(xj - blu) / (np.linalg.norm(xj) + self.vfloor)
            rj2 = min(rj2, 9.0)
            rr_tot += rj2 / n
            campl[j] = c_s
        # 残差比慢EMA → 相干形状门(硬阈值)
        if self._rr_ema is None:
            self._rr_ema = rr_tot
        else:
            self._rr_ema = self.rr_ema * self._rr_ema + (1 - self.rr_ema) * rr_tot
        rr = self._rr_ema
        coh = (1.0 - min(rr / self.rrcut, 1.0)) ** self.rr_pow
        coh = self.rr_floor + (1.0 - self.rr_floor) * coh
        hf_gate = float(np.clip(hf / self.thr_hf, 0.0, 1.0)) ** self.p
        gate = hf_gate * coh * self.gammax             # 放大强度
        self.gate_sum += gate; self.gate_n += 1
        if gate < 1e-3:
            return a
        # --- 蓝图相位重构：dc + (1+gate)·campl·sin(Φ_cur+off_j) ---
        phi_cur = 2 * np.pi * self.freq * (self._t - self.dt)
        blu_cur = self.gain * np.sin(phi_cur + self.offs)      # 蓝图方向(逐关节偏移)
        dc = A[-1] - campl * np.sin(phi_cur + self.offs)       # DC ≈ a - 蓝图相位分量
        blue_sig = campl * np.sin(phi_cur + self.offs)         # 当前蓝图相位分量
        # 蓝图SNR放大：把相干(蓝图)分量放大 (1+gate)，DC不动
        target = dc + (1.0 + gate) * blue_sig * self.gain
        # 蓝图锁定：向蓝图子空间同时混合(保留幅值放大收益)
        out = a + (self.lam * coh) * (target - a)
        return np.asarray(out, dtype=float)


class RobustComboV8(BasePlugin):
    """第47轮：BestCombo -> ULI(干净) -> BlueprintLockedBoost(蓝图锁定SNR放大)。

    用蓝图相位锁定替换 DNB2 的自由相位放大：重建目标永远落蓝图子空间，
    同时拿到 执行噪声下的 CI/SNR 增益 和 参照性R_ref+联动性L_link(跨关节共享全局
    相位) 的保留/提升，杜绝 前几轮 CI↑ 但 R_ref↓0.6~0.8 的净损失。
    """

    def __init__(self, combo, uli, **blb_kw):
        self.combo = combo
        self.uli = uli
        self.blb = BlueprintLockedBoost(**blb_kw)

    def reset(self):
        self.blb.reset()
        if hasattr(self.uli, "reset"):
            self.uli.reset()

    def inject(self, a, **kw):
        a = self.combo.inject(a, **kw)
        a = self.uli.inject(a, **kw)
        return self.blb.inject(a, **kw)


# --------------------------------------------------------------------------- #
# 第48轮 — BlueprintRefLock (BRL)：零相位蓝图参照锁定(不含 BestCombo)           #
# --------------------------------------------------------------------------- #
class BlueprintRefLock(BasePlugin):
    """
    BRL：解决 R_ref 被 BestCombo 因果平滑(=相位滞后)砸坏的根因。

    第47轮阶段扫描确认：仅 BestCombo 就让 smooth 族 R_ref 从 0.938 掉到 0.316
    (clean 也一样) —— 因果 Kalman/LPF 逐拍平滑把 q 相对蓝图引入了相位滞后，
    逐时刻跨关节相关立刻崩塌。而 R 在基线上已高达 0.93~0.96，本不该被毁。

    设计(零相位、蓝图参照锁定)：
      - inject 收到 t(时间) → 全局步态相位 Φ=2πft 完全可知(频率已知=1.0Hz)。
      - 每关节在滑窗内对蓝图相位基 [1, sin(Φ+off_j), cos(Φ+off_j)] 做 LS 拟合，
        得到该关节在"蓝图频率"上的(幅值,相位)。由于相位是解析已知、逐样本求值，
        **不存在平滑滤波的相位滞后**。
      - 重建：out_j(t) = dc_j(t) + A_j·sin(Φ(t)+off_j) + B_j·cos(Φ(t)+off_j)
        —— 恒落在蓝图子空间 → R_ref 由构造保住(≈基线甚至更高)。
      - 门控：逐关节重建残差比 rr(平滑族低/p2p高)。平滑族 → 锁定到蓝图周期轨迹
        (C 一致性、L 联动性同步抬升)；p2p 残差大 → 不锁 → 保护。
      - 不发散：只混合 lam·gate 比例，保留 DC 与正交内容，clean 下对平滑族
        (本就周期) 趋近恒等、不大改。

    可选用已有 ULI(干净训练引导) 前置；但**不再需要 BestCombo 的滞后平滑底座**，
    因为那正是毁掉参照性的元凶。
    """
    def __init__(self, Win=31, freq=1.0, dt=0.01, lam=0.6,
                 rrcut=0.35, rr_pow=1.0, rr_ema=0.990, dc_ema=0.995,
                 amp_ema=0.990, vfloor=1e-3, clean_decay=0.990,
                 gammax=0.0, thr_hf=0.06, hf_p=1.0, hf_ema=0.980, lk_lam=0.4):
        self.Win, self.freq, self.dt = Win, freq, dt
        self.lam = lam
        self.rrcut, self.rr_pow, self.rr_ema = rrcut, rr_pow, rr_ema
        self.dc_ema, self.amp_ema = dc_ema, amp_ema
        self.vfloor = vfloor
        self.gammax = gammax
        self.thr_hf = thr_hf
        self.hf_p = hf_p
        self.hf_ema = hf_ema
        self.lk_lam = lk_lam          # 蓝图锁定强度(保R/C/L)
        self.offs = np.linspace(0, np.pi, 6)
        k = np.arange(Win) * dt
        self.phi = 2 * np.pi * freq * k                # (Win,) 全局相位
        self.sb = np.sin(self.phi[:, None] + self.offs[None, :])   # (Win,6)
        self.cb = np.cos(self.phi[:, None] + self.offs[None, :])
        # 每关节基 B_j = [1, sb_j, cb_j] → 预计算 (Win,3) 与伪逆
        self.Binv = {}
        for j in range(6):
            B = np.stack([np.ones(Win), self.sb[:, j], self.cb[:, j]], axis=1)
            self.Binv[j] = np.linalg.pinv(B)
        self.act_buf = deque(maxlen=Win)
        self._dc = None
        self._amp = None
        self._rr_ema = None
        self._prev_dq = None
        self._hf_ema = None
        self._t = 0.0

    def reset(self):
        self.act_buf.clear(); self._dc = None; self._amp = None
        self._rr_ema = None; self._prev_dq = None; self._hf_ema = None
        self._t = 0.0

    def inject(self, a, **kw):
        a = np.asarray(a).copy()
        dqv = kw.get("dq", None)
        self.act_buf.append(a)
        self._t += self.dt
        if len(self.act_buf) < self.Win:
            return a
        A = np.stack(list(self.act_buf))               # (Win,n)
        n = a.shape[0]
        phi_cur = 2 * np.pi * self.freq * (self._t - self.dt)
        # --- 状态白噪(hf)：执行噪声→逐拍白噪∝1/dt；用于 SNR 放大门(噪声时才放大) ---
        hf = 0.0
        if dqv is not None:
            dvv = np.asarray(dqv)
            if self._prev_dq is None:
                self._prev_dq = dvv.copy()
            else:
                d = np.linalg.norm(dvv - self._prev_dq) / (np.linalg.norm(dvv) + self.vfloor)
                if self._hf_ema is None:
                    self._hf_ema = d
                else:
                    self._hf_ema = self.hf_ema * self._hf_ema + (1 - self.hf_ema) * d
                self._prev_dq = dvv.copy()
        hf = self._hf_ema if self._hf_ema is not None else 0.0
        n_gate = float(np.clip(hf / self.thr_hf, 0.0, 1.0)) ** self.hf_p if self.gammax > 0 else 0.0
        out = a.copy()
        rr_tot = 0.0
        for j in range(n):
            xj = A[:, j]
            c = self.Binv[j] @ xj                      # (3,) = [dc, A_s, B_c]
            dcj, As, Bc = c[0], c[1], c[2]
            phj = phi_cur + self.offs[j]
            recon = dcj + As * np.sin(phj) + Bc * np.cos(phj)
            res = np.linalg.norm(xj - (dcj + As * self.sb[:, j] + Bc * self.cb[:, j]))
            rrj = res / (np.linalg.norm(xj) + self.vfloor)
            rrj = min(rrj, 9.0)
            rr_tot += rrj
            # 蓝图 SNR 放大：放大蓝图频率分量幅值((1+γ·n_gate))，相位/DC不变
            if self.gammax > 0:
                amp = np.sqrt(As * As + Bc * Bc)
                gr = 1.0 + n_gate * self.gammax
                As2, Bc2 = As * gr, Bc * gr
                recon = dcj + As2 * np.sin(phj) + Bc2 * np.cos(phj)
            out[j] = recon
        rr_tot = rr_tot / n
        # 残差比慢EMA → 蓝图锁定门(平滑族低→高锁, p2p高→不锁)
        if self._rr_ema is None:
            self._rr_ema = rr_tot
        else:
            self._rr_ema = self.rr_ema * self._rr_ema + (1 - self.rr_ema) * rr_tot
        rr = self._rr_ema
        coh = (1.0 - min(rr / self.rrcut, 1.0)) ** self.rr_pow
        if coh < 1e-3:
            return a
        out = a + (self.lam * coh) * (out - a)
        return np.asarray(out, dtype=float)


class RobustComboV9(BasePlugin):
    """第48轮：ULI(干净引导) -> BlueprintRefLock(蓝图参照零相位锁定)。
    # 移除 BestCombo → 从根上不再因果滞后 → R_ref 保住；蓝图锁定同时给 C/L/CI。
    """
    def __init__(self, uli, **brl_kw):
        self.uli = uli
        self.brl = BlueprintRefLock(**brl_kw)

    def reset(self):
        self.brl.reset()
        if hasattr(self.uli, "reset"):
            self.uli.reset()

    def inject(self, a, **kw):
        if self.uli is not None:
            a = self.uli.inject(a, **kw)
        return self.brl.inject(a, **kw)


# --------------------------------------------------------------------------- #
# 第50轮 — PhaseRecomb (V11)：零净损失的相位相干周期去抖动                    #
# --------------------------------------------------------------------------- #
class PhaseRecomb(BasePlugin):
    """
    PhaseRecomb：在"零净损失(不降C/L/R)"前提下同时抬升 C 一致性 / CI 连贯性。

    诊断（第49轮数据，blb_screen_v9）：
      - 平滑族基座 R 已近饱和(0.94~0.96)，L 处蓝图操作点(~0.55) —— 几乎没有可提
        空间，关键是不能丢。真正有空间的是 C(standard 0.22/transformer 0.60) 与
        CI(0.116)。
      - BRL 失败根因：把关节信号**强制替换**成"拟合基波正弦"，系数随滑窗抖动 +
        DC 漂移，致重建形状比基座"自然跟随"更偏离蓝图 → R 不升反降(0.938→0.855)，
        C/L 也因砍谐波而略降。
      - DNB/BLB 等"SNR 放大"同样会对平滑族改造形状 → 净损失。

    设计 —— 不做任何"形状替换"，只做"相位同步的周期去抖动"：
      1) 全局相位 Φ=2πft(频率=1Hz已知) → 周期边界完全确定(零相位、无滞后)。
      2) 每关节维护"历史周期模板" T_j[M]，= 已结束周期的相位分箱EMA。
         **模板保留该关节真实的谐波形状(非强制正弦)** → 反映"平均周期形状"，
         与蓝图贴近 → R(跨关节蓝图相关) 保住；去均值标准化后与相位滞后无关。
      3) 输出：out_j = a_j + lam·coh_j·(T_j[bin] - a_j)。
         把当前样本向"同相位的历史均值形状"收拢 → 抑制逐周期相位/形状抖动
         → C(逐周期一致) 与 CI(平滑连贯) 抬升。
      4) coh_j 逐关节门：模板与当前周期形状残差比 —— 平滑族低→开门去抖；
         p2p 非周期/阶跃 → 残差大 → coh_j≈0 → 直通保护(Universal)。
      5) 只去抖动不改均值 → R 与 L 不降，甚至因抖动去除而微升。
    """

    def __init__(self, M=100, dt=0.01, freq=1.0, lam=0.5, alpha=0.05,
                 rrcut=0.35, rr_pow=1.0, rr_ema=0.990, vfloor=1e-3,
                 min_cyc=2, r2_floor=0.20, r2_ceil=0.55, r2_pow=1.0,
                 adapt=False, lam_ref=0.45, lam_min=0.05,
                 blend=True, r2b_lo=0.35, r2b_hi=0.75, blend_pow=1.0,
                 rr_stand=False, rr_lo=0.030, rr_hi=0.045, rr_stand_pow=1.0,
                 gbl=0.0, gbl_alpha=0.05, gbl_floor=0.02,
                 tpl_gate=False, tpl_r2lo=0.30, tpl_r2hi=0.60, tpl_r2p=1.0,
                 r2_stand=False, r2_stand_lo=0.85, r2_stand_hi=0.95, r2_stand_p=1.0):
        self.tpl_gate = tpl_gate      # r2门控模板更新: 低1Hz相干周期少贡献(抗噪声污染), 默认关保V14
        self.tpl_r2lo = tpl_r2lo
        self.tpl_r2hi = tpl_r2hi
        self.tpl_r2p = tpl_r2p
        self.r2_stand = r2_stand      # r2高值 stand-down: 已饱和+超高相干(干净transformer)→lam归零不动(护净损失)
        self.r2_stand_lo = r2_stand_lo
        self.r2_stand_hi = r2_stand_hi
        self.r2_stand_p = r2_stand_p
        self.M, self.dt, self.freq = M, dt, freq
        self.lam = lam
        self.alpha = alpha
        self.rrcut = rrcut
        self.rr_pow = rr_pow
        self.rr_ema = rr_ema
        self.vfloor = vfloor
        self.min_cyc = min_cyc
        self.r2_floor = r2_floor    # 蓝图1Hz基R²下限(低于→门关, 护p2p)
        self.r2_ceil = r2_ceil      # R²上限(高于→门全开)
        self.r2_pow = r2_pow
        self.r2_ema_ = rr_ema       # R²慢EMA系数(复用rr_ema)
        self.adapt = adapt          # 自适应lam：关节越一致(transformer)→lam越小
        self.lam_ref = lam_ref      # rr=lam_ref → lam 满额
        self.lam_min = lam_min      # 自适应lam下限(最小干预)
        self.blend = blend          # 高1Hz相干R²→目标=干净1Hz骨干(护R/升C)；低R²→模板(保谐波)
        self.r2b_lo = r2b_lo
        self.r2b_hi = r2b_hi
        self.blend_pow = blend_pow
        self.rr_stand = rr_stand    # 低rr stand-down：关节已经足够自洽(rr很小)→门关(不做无谓改造)
        self.rr_lo = rr_lo          # rr低于该值→门随rr下降而关闭(护"已饱和"的transformer-clean)
        self.rr_hi = rr_hi          # rr高于该值→stand-down门全开
        self.rr_stand_pow = rr_stand_pow
        self.gbl = gbl                # 蓝图相位联动强度(0=关)：把关节向全局蓝图锁相(抬 L 联动 / R 参照)
        self.gbl_alpha = gbl_alpha    # 逐关节幅值比 s_j 的慢EMA系数
        self.gbl_floor = gbl_floor
        self._gle = None              # 逐关节幅值/A 模板比例 EMA (n,)
        self.tpl = None          # (M,n) 历史周期模板
        self.cyc = None          # (M,n) 当前周期累加
        self.B1 = None           # (M,3) 蓝图1Hz基[1,sin,cos]
        self.B1_inv = None
        self.bb = None           # (M,n) 上一完整周期的1Hz骨干(去抖目标)
        self._prev_bin = -1
        self._ncyc = 0           # 完成的完整周期数
        self._rr_ema = None      # (n,) 残差比慢EMA
        self._r2_ema = None      # (n,) 蓝图1Hz相干R²慢EMA

    def reset(self):
        self.tpl = None; self.cyc = None
        self.B1 = None; self.B1_inv = None
        self.bb = None
        self._prev_bin = -1; self._ncyc = 0
        self._rr_ema = None; self._r2_ema = None
        self._gle = None

    def inject(self, a, **kw):
        a = np.asarray(a).copy()
        n = a.shape[0]
        t = kw.get("t", None)
        if t is None:
            raise ValueError("PhaseRecomb requires t (time) in inject kwargs")
        if self.cyc is None:
            self.cyc = np.full((self.M, n), np.nan)
        if self.B1 is None:
            phi1 = 2 * np.pi * self.freq * (np.arange(self.M) * self.dt)
            self.B1 = np.stack([np.ones(self.M), np.sin(phi1), np.cos(phi1)], axis=1)
            self.B1_inv = np.linalg.pinv(self.B1)
        bin_ = int((self.freq * t * self.M) % self.M)
        wrap = self._prev_bin >= 0 and bin_ < self._prev_bin
        self._prev_bin = bin_
        self.cyc[bin_] = a
        if wrap:
            # 上一周期完整：所有M个bins都已填充(闭环周期)
            valid = ~np.isnan(self.cyc)
            if np.all(valid):
                cyc_ok = self.cyc.copy()
                # --- 蓝图1Hz相干R²(频率门控)：p2p非1Hz→R²低→关 ---
                c1 = self.B1_inv @ cyc_ok                    # (3,n)
                fit1 = self.B1 @ c1                           # (M,n)
                var_tot = np.var(cyc_ok, axis=0) + self.vfloor
                var_exp = np.var(fit1, axis=0)
                r2 = np.clip(var_exp / var_tot, 0.0, 1.0)
                if self._r2_ema is None:
                    self._r2_ema = r2
                else:
                    self._r2_ema = self.r2_ema_ * self._r2_ema + (1 - self.r2_ema_) * r2
                if self.blend:
                    self.bb = fit1          # (M,n) 本完整周期的1Hz骨干(去抖干净目标)
                if self.tpl_gate:
                    # r2门控: 本周期逐关节1Hz相干R²越低→更新权重越小(抗噪声污染模板)
                    u = np.clip((r2 - self.tpl_r2lo) / (self.tpl_r2hi - self.tpl_r2lo + 1e-9),
                                0.0, 1.0) ** self.tpl_r2p
                else:
                    u = np.ones(n)
                if self.tpl is None:
                    self.tpl = cyc_ok.copy()
                else:
                    self.tpl = (1 - self.alpha * u) * self.tpl + self.alpha * u * cyc_ok
                self._ncyc += 1
                # 逐关节残差比：本周期 vs 模板(去均值标准化抗相位/幅值)
                cm = cyc_ok - cyc_ok.mean(axis=0, keepdims=True)
                tm = self.tpl - self.tpl.mean(axis=0, keepdims=True)
                cstd = cm.std(axis=0) + self.vfloor
                tstd = tm.std(axis=0) + self.vfloor
                rr = np.mean(np.abs(cm / cstd - tm / tstd), axis=0)  # (n,)
                rr = np.clip(rr, 0.0, 9.0)
                if self._rr_ema is None:
                    self._rr_ema = rr
                else:
                    self._rr_ema = self.rr_ema * self._rr_ema + (1 - self.rr_ema) * rr
        if self.tpl is None or self._ncyc < self.min_cyc:
            return a
        e = self._rr_ema if self._rr_ema is not None else np.zeros(n)
        coh = (1.0 - np.minimum(e / self.rrcut, 1.0)) ** self.rr_pow  # (n,)
        if self.rr_stand:
            # 低rr stand-down：关节已足够自洽(rr→很小)→该关节无需去抖→门关
            # (护"已饱和"的 clean/transformer：基座本就每周期高度一致, 强去抖反而致损)
            stand = np.clip((e - self.rr_lo) / (self.rr_hi - self.rr_lo + 1e-9),
                            0.0, 1.0) ** self.rr_stand_pow
            coh = coh * stand
        # --- 蓝图1Hz相干门：平滑族R²高→开；p2p 非1Hz→R²低→关(护Universal) ---
        r2e = self._r2_ema if self._r2_ema is not None else np.zeros(n)
        fac = np.clip((r2e - self.r2_floor) / (self.r2_ceil - self.r2_floor + 1e-9),
                      0.0, 1.0) ** self.r2_pow
        coh = coh * fac
        # --- r2高值 stand-down：关节已超高相干(干净饱和transformer, r2→~0.95)→无需干预→门关
        #     噪声下 r2 回落→门开→保留噪声去抖增益。与 rr_stand(伤害standard)不同, 用r2判别不误伤.
        if self.r2_stand:
            st = np.clip((self.r2_stand_hi - r2e) / (self.r2_stand_hi - self.r2_stand_lo + 1e-9),
                         0.0, 1.0) ** self.r2_stand_p
            coh = coh * st
        # --- 自适应lam：关节自身越一致(transformer rr小)→lam越小→近乎恒等 ---
        if self.adapt:
            lamj = self.lam * np.clip(e / self.lam_ref, 0.0, 1.0)
            lamj = np.maximum(lamj, self.lam_min)
        else:
            lamj = np.full(n, self.lam)
        coh = coh * lamj
        coh = np.where(coh < 1e-3, 0.0, coh)
        nz = coh > 1e-3
        out = a.copy()
        if np.any(nz):
            target = self.tpl[bin_, :].copy()      # 默认：全谐波模板(保R/形状)
            if self.blend and self.bb is not None:
                # 高1Hz相干R²→去抖目标=干净1Hz骨干(贴合蓝图, 升C护R)；
                # 低R²(有真实谐波)→模板(保形状)。β逐关节 R² 线性过渡。
                r2e = self._r2_ema if self._r2_ema is not None else np.zeros(n)
                beta = np.clip((r2e - self.r2b_lo) / (self.r2b_hi - self.r2b_lo + 1e-9),
                               0.0, 1.0) ** self.blend_pow
                target = (1.0 - beta) * self.tpl[bin_, :] + beta * self.bb[bin_, :]
            out[nz] = a[nz] + coh[nz] * (target[nz] - a[nz])
        # --- GBL: 全局蓝图相位联动(只在协同门打开时, 逐关节缩放, 抬 L/R 不动 C) ---
        if self.gbl > 0.0:
            phi = 2 * np.pi * self.freq * t
            offs = np.linspace(0, np.pi, n)
            bp = 0.25 * np.sin(phi + offs)               # (n,) 固定蓝图(无系数抖动)
            s_prev = self._gle if self._gle is not None else np.ones(n)
            # 逐关节 A/蓝图 幅值比：用估计幅值 |A|(慢EMA) / 蓝图幅值基线
            amp_a = np.abs(a)
            s_new = (amp_a / (np.abs(bp) + self.gbl_floor))
            s_new = np.maximum(s_new, 0.1)
            self._gle = (1 - self.gbl_alpha) * s_prev + self.gbl_alpha * s_new
            gle = self._gle
            gtarget = gle * bp
            # gble 门沿用 coh(已含r2保护p2p)；但 clean/trans 已饱和时不宜强拉蓝图→用低G
            gcoh = coh  # (n,)
            out = out + gcoh * self.gbl * (gtarget - out)
        return np.asarray(out, dtype=float)


class RobustComboV11(BasePlugin):
    """第50轮：ULI(干净引导) -> PhaseRecomb(相位相干周期去抖, 零净损失保R/L升C/CI)。

    相比 V9(BRL 强制正弦替换)：PhaseRecomb 保留关节真实谐波形状、只抑制周期抖动，
    因此 R/L 不降而 C/CI 抬升，满足"无净损失"目标。
    """

    def __init__(self, uli, **pr_kw):
        self.uli = uli
        self.pr = PhaseRecomb(**pr_kw)

    def reset(self):
        self.pr.reset()
        if hasattr(self.uli, "reset"):
            self.uli.reset()

    def inject(self, a, **kw):
        if self.uli is not None:
            a = self.uli.inject(a, **kw)
        return self.pr.inject(a, **kw)


class RobustComboV10(BasePlugin):
    """第49轮：BestCombo(CI/SNR增益) -> ULI -> BlueprintRefLock(修复被BestCombo滞后砸掉的R)。
    关键实验：V6 用 BestCombo 拿到 CI+0.4 但把 R 砸到0.32(因果滞后)；BRL 是零相位蓝图
    锁定。若 BRL 排在 BestCombo 之后，能否"先拿CI增益、再通过蓝图锁定把R拉回基线"
    —— 即 C/L/R 与 CI 同时提升、杜绝净损失。
    """
    def __init__(self, combo, uli, **brl_kw):
        self.combo = combo
        self.uli = uli
        self.brl = BlueprintRefLock(**brl_kw)

    def reset(self):
        self.brl.reset()
        if hasattr(self.uli, "reset"):
            self.uli.reset()

    def inject(self, a, **kw):
        a = self.combo.inject(a, **kw)
        if self.uli is not None:
            a = self.uli.inject(a, **kw)
        return self.brl.inject(a, **kw)