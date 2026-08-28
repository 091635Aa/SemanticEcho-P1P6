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