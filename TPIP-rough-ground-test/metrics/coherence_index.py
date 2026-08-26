#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coherence_index.py
==================
计算"连贯性指数 (Coherence Index, CI)"。

衡量一条关节角/质心轨迹是"点对点式(P2P, 僵硬)"还是"连贯式(Coherent, 平滑惯性连贯)"。

定义 (两块子指标，各归一化到 [0,1]，加权求和):
  CI = w_jerk * S_smooth + w_phase * P_coinc

  [1] S_smooth —— 加速度变化率导数(加加速度, jerk)的均方根倒数的归一化
      jerk(t) = d(q_ddot)/dt
      rms_jerk = sqrt(mean(jerk^2))
      gait_scale = A * (2*pi*f0)^3   (理想步态加加速度标度, f0=主频, A=幅值)
      S_smooth = 1 / (1 + rms_jerk/gait_scale / jitter_mult)
      高加加速度(震颤/尖峰) -> rms_jerk 相对标度剧增 -> S_smooth 低 -> 差评(P2P 特征)

  [2] P_coinc —— 步态相图重合度 (Gait Phase-Portrait Coincidence)
      用主频基波解析相位把轨迹切成相位对齐的完整步态周期，
      将每周期相空间轨迹 (q, dq) 重采样到统一相角[0,1)，
      计算"逐相角离散度"(同相位下各周期点的散布)：
        P_coinc = 1 / (1 + phase_disp / global_scale / map_factor)
      步态稳定重复 -> 各周期贴合 -> 低离散 -> P 高(连贯)。
      相位漂移/抖振/中途重规划 -> 周期互相错开 -> 高离散 -> P 低(P2P/重规划式)。

CI ∈ [0, 1]，更高 = 更连贯。

用法:
  python coherence_index.py --data run.npz                 # 单条实验
  python coherence_index.py --data base.npz --data tpip.npz # 对比基线 vs TPIP
  python coherence_index.py --selftest                       # 合成数据自检(演示分辨力)
  python coherence_index.py --help

--data 接受的格式 (npz/csv/json 均可, 参考 build 演示):
  必需维度: q: (T, n_joints) 关节角 [rad] (或质心轨迹任一维度)
  可选维度: dq: (T, n_joints), q_ddot: (T, n_joints), accel: (T, n_dof)
            zmp: (T, 2) 零力矩点 [m], foot_contact: (T,) 布尔 每次脚触地=1
  若未给 dq/q_ddot，脚本用中心差分自动求导。

配套自检 --selftest 生成两条合成轨迹:
  base : 方波/平台式扭矩 -> 关节角呈 PWM 抖动(加加速度大), 相图发散  -> 低 CI (P2P)
  tpip : 光滑摆线/正弦过渡, 相图画圆重合                      -> 高 CI (连贯)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# 自检/演示用合成轨迹生成器（仅用于演示工具分辨力，不是真实仿真数据）          #
# --------------------------------------------------------------------------- #


def _synth_p2p(T: int = 800, dt: float = 0.01) -> np.ndarray:
    """合成 P2P(点对点) 关节角轨迹：阶跃式控制 + 高频 PWM 抖振 + 中途重规划。

    本质上是"分段平台->阶跃"的控制：相图发散、加加速度高，
    并在 t≈0.35s 处发生一次"重规划"相位跳变(代表面对扰动重新规划)。
    """
    t = np.arange(T) * dt
    period = 0.5  # s
    q = np.zeros(T)
    for i, tt in enumerate(t):
        base = 0.8 * math.sin(2 * math.pi * tt / period)
        # 平台式量化：把目标离散成台阶，制造加速度不连续
        quant = 0.06 * np.round(math.sin(2 * math.pi * tt / period) / 0.06)
        # 目标附近高频 PID 震颤 (非整数倍步频 -> 相位随周期漂移, 相图发散)
        jitter = 0.04 * np.sign(math.sin(2 * math.pi * 25.0 * tt))
        # 遭遇微小凸起后"重新规划"导致的相位跳变
        if 0.34 < tt < 0.40:
            base = base + 0.45 * math.sin(2 * math.pi * (tt - 0.34) / 0.02)
        q[i] = base + quant + jitter
    return q


def _synth_coherent(T: int = 800, dt: float = 0.01) -> np.ndarray:
    """合成连贯式 关节角轨迹：光滑摆线/正弦，相图画圆稳定、低加加速度。"""
    t = np.arange(T) * dt
    period = 0.5  # s
    ph = 2 * math.pi * t / period
    # 光滑周期信号：三阶导数连续
    return 0.8 * np.sin(ph) + 0.15 * np.sin(2 * ph)


# --------------------------------------------------------------------------- #
# 差分与子指标实现                                                             #
# --------------------------------------------------------------------------- #


def central_diff(x: np.ndarray, dt: float, axis: int = 0) -> np.ndarray:
    """中心差分一阶导数，两端单向差分。"""
    d = np.empty_like(x, dtype=float)
    d[0] = (x[1] - x[0]) / dt
    d[-1] = (x[-1] - x[-2]) / dt
    d[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    return d


def _summarize(sig: np.ndarray) -> np.ndarray:
    """多维信号 -> 逐时间步的包络(用于 rms 类统计 / 周期切分信号)。"""
    if sig.ndim == 1:
        return np.asarray(sig, dtype=float)
    return np.linalg.norm(sig, axis=-1)


def gait_jerk_scale(q: np.ndarray, dt: float) -> float:
    """理想步态加加速度标度 = A * (2π·f0)^3。

    用主频 f0(FFT 峰值)与幅值 A(std·√2) 估计"纯正弦行走"的理论加加速度量级，
    用于把实测 jerk 无量纲化，使 S_smooth 跨实验可比。
    """
    q = np.asarray(q, dtype=float).ravel()
    fft = np.abs(np.fft.rfft(q - np.mean(q)))
    freqs = np.fft.rfftfreq(len(q), d=dt)
    if len(freqs) > 1:
        f0 = freqs[1:][int(np.argmax(fft[1:]))] if np.any(fft[1:] > 0) else 1.0
    else:
        f0 = 1.0
    f0 = max(f0, 1e-9)
    amp = float(np.std(q)) * math.sqrt(2.0) + 1e-9
    return amp * ((2 * math.pi * f0) ** 3)


def smoothness_score(jerk_rms: float, gait_scale: float,
                     jitter_mult: float = 4.0) -> float:
    """S_smooth: jerk 相对理想步态标度的比值越小越连贯。0->1 单调。

    ratio = jerk_rms / gait_scale
    S_smooth = 1 / (1 + ratio / jitter_mult)
    纯基频正弦 ~ ratio≈0.707 -> 中高值；高频抖振后 ratio 剧增 -> S 压低。
    """
    ratio = jerk_rms / (gait_scale + 1e-9)
    return 1.0 / (1.0 + ratio / jitter_mult)


def _gait_phase(q: np.ndarray, dt: float) -> np.ndarray:
    """主频基波累积相位 (rad, 单调递增)：用主导 FFT 分量提取，用于周期切分。"""
    x = q - np.mean(q)
    n = len(x)
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=dt)
    if n > 2 and np.any(np.abs(fft[1:]) > 0):
        k = 1 + int(np.argmax(np.abs(fft[1:])))
    else:
        k = 1
    f0 = max(freqs[k], 1e-9)
    ph0 = np.angle(fft[k])
    return 2 * np.pi * f0 * np.arange(n) * dt + ph0  # 累积相位, 每 +2π = 一个步态周期


def _phase_coincidence(q: np.ndarray, dq: np.ndarray,
                       foot_contact=None, dt: float = 0.01,
                       n_bins: int = 32, map_factor: float = 0.18) -> float:
    """
    P_coinc: 步态相图重合度。

    用主频基波解析相位把轨迹切成相位对齐的完整步态周期，将每周期相空间
    轨迹 (q, dq) 重采样到统一相角[0,1)，计算"逐相角离散度"：
      P_coinc = 1 / (1 + mean_phase_dispersion / global_scale / map_factor)
    稳定重复的步态 -> 各周期贴合 -> 低离散 -> P 高(连贯)。
    相位漂移/抖振/中途重规划 -> 周期互相错开 -> 高离散 -> P 低(P2P/重新规划式)。
    """
    if q.ndim > 1:
        q = q[:, 0]
    if dq.ndim > 1:
        dq = dq[:, 0]
    if q.ndim > 1 or len(q) < 64:
        return 0.0

    # 相位对齐的完整周期切分 (主频基波每 2π 一个周期)
    phi = _gait_phase(q, dt)
    cycle_idx = np.floor((phi + np.pi) / (2 * np.pi)).astype(int)
    starts = np.where(np.diff(cycle_idx, prepend=cycle_idx[0]) != 0)[0]
    starts = np.concatenate([starts, [len(q)]])
    if len(starts) < 3:
        return 0.0

    # 相空间归一化
    qn = (q - q.mean()) / (q.std() + 1e-9)
    dqn = (dq - dq.mean()) / (dq.std() + 1e-9)

    ph = np.linspace(0.0, 1.0 - 1e-9, n_bins)
    cycles = []
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < n_bins:
            continue
        seg = np.linspace(0.0, 1.0, b - a)
        cyc = np.stack([np.interp(ph, seg, qn[a:b]),
                        np.interp(ph, seg, dqn[a:b])], axis=-1)  # (n_bins,2)
        cycles.append(cyc)
    if len(cycles) < 2:
        return 0.0
    C = np.stack(cycles, axis=0)                      # (n,n_bins,2)
    global_scale = float(np.std(C)) + 1e-9
    phase_disp = float(np.mean(np.linalg.norm(C - C.mean(axis=0), axis=-1)))
    return float(np.clip(1.0 / (1.0 + phase_disp / global_scale / map_factor),
                         0.0, 1.0))


def _zero_crossings(sig: np.ndarray):
    s = np.sign(sig)
    return np.where(np.diff(s, prepend=0) != 0)[0]


# --------------------------------------------------------------------------- #
# 主入口                                                                       #
# --------------------------------------------------------------------------- #


def load_trajectory(path: str, dt: float = 0.01):
    """解析 npz/csv/json -> (q, dq, q_ddot, zmp, foot_contact)。"""
    p = Path(path)
    q = dq = q_ddot = None
    zmp = None
    foot_contact = None
    if p.suffix == ".npz":
        data = np.load(p)
        q = data["q"] if "q" in data else None
        dq = data.get("dq")
        q_ddot = data.get("q_ddot")
        zmp = data.get("zmp")
        foot_contact = data.get("foot_contact")
    elif p.suffix in (".csv", ".txt"):
        arr = np.genfromtxt(p, delimiter=",", skip_header=1)
        q = arr[:, 1] if arr.ndim == 2 else arr
    elif p.suffix == ".json":
        d = json.loads(p.read_text())
        q = np.asarray(d["q"])
        dq = np.asarray(d.get("dq", [])) or None
        q_ddot = np.asarray(d.get("q_ddot", [])) or None
        foot_contact = np.asarray(d.get("foot_contact", [])) or None
    else:
        raise ValueError(f"unsupported format: {path}")

    if q is None:
        raise ValueError("no q (joint-angle) field in trajectory")
    q = np.asarray(q, dtype=float)
    if dq is None:
        dq = central_diff(q, dt)
    dq = np.asarray(dq, dtype=float)
    if q_ddot is None:
        q_ddot = central_diff(dq, dt)
    return q, dq, q_ddot, zmp, foot_contact


def compute_coherence(q, dq, q_ddot, foot_contact=None,
                      w_jerk=0.5, w_phase=0.5, dt=0.01, jitter_mult=4.0) -> dict:
    q = np.asarray(q, dtype=float)
    # 加加速度 = 加速度的时间导数 (对所有关节/维度取整体 RMS, 形状无关)
    jerk = central_diff(np.asarray(q_ddot, dtype=float), dt)
    rms_jerk = float(np.sqrt(np.mean(jerk ** 2)))
    # 代表性 1D 信号(方差最大的通道) 供步态标度/相图使用, 保证 1D/多通道一致
    rep = q if q.ndim == 1 else q[:, int(np.argmax(np.var(q, axis=0)))]
    gait_scale = gait_jerk_scale(rep, dt)
    s_smooth = smoothness_score(rms_jerk, gait_scale, jitter_mult)
    p_coinc = _phase_coincidence(rep, dq, foot_contact, dt=dt)
    ci = w_jerk * s_smooth + w_phase * p_coinc
    return {
        "coherence_index": ci,
        "s_smooth": s_smooth,
        "rms_jerk": rms_jerk,
        "gait_jerk_scale": gait_scale,
        "p_phase_coincidence": p_coinc,
        "w_jerk": w_jerk,
        "w_phase": w_phase,
    }


def _fmt(r: dict, name: str) -> str:
    return (f"[{name}] CI={r['coherence_index']:.4f} | "
            f"S_smooth={r['s_smooth']:.3f} (rms_jerk={r['rms_jerk']:.4f}) | "
            f"P_coinc={r['p_phase_coincidence']:.3f}")


def run_selftest():
    """合成演示：验证工具能分辨 P2P(差) 与 连贯(好)。"""
    q_p2p = _synth_p2p()
    q_coh = _synth_coherent()
    dt = 0.01
    r_p2p = compute_coherence(q_p2p, central_diff(q_p2p, dt),
                              central_diff(central_diff(q_p2p, dt), dt), dt=dt)
    r_coh = compute_coherence(q_coh, central_diff(q_coh, dt),
                              central_diff(central_diff(q_coh, dt), dt), dt=dt)
    print("合成自检 (仅演示工具分辨力，非真实仿真):")
    print("  " + _fmt(r_p2p, "P2P(点对点体式, 应低)"))
    print("  " + _fmt(r_coh, "Coherent(连贯摆线, 应高)"))
    delta = r_coh["coherence_index"] - r_p2p["coherence_index"]
    print(f"  差量 dCI = {delta:+.4f}  (期望 > 0.10)")
    ok = delta > 0.10
    print("  自检 " + ("通过 ✓" if ok else "未通过 ✗"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="TPIP 连贯性指数")
    ap.add_argument("--data", action="append", help="轨迹文件(可多个对比)")
    ap.add_argument("--labels", nargs="*", help="对应标签, 默认用文件名")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--w-jerk", type=float, default=0.5)
    ap.add_argument("--w-phase", type=float, default=0.5)
    ap.add_argument("--jitter-mult", type=float, default=4.0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return run_selftest()
    if not args.data:
        ap.error("需提供 --data 或 --selftest")

    labels = args.labels or [Path(d).stem for d in args.data]
    results = []
    for d in args.data:
        q, dq, qd, zmp, fc = load_trajectory(d, args.dt)
        r = compute_coherence(q, dq, qd, fc,
                              w_jerk=args.w_jerk, w_phase=args.w_phase,
                              dt=args.dt, jitter_mult=args.jitter_mult)
        results.append((labels[len(results)], r))
        print("  " + _fmt(r, labels[len(results) - 1]))

    if len(results) == 2:
        (ln, rn), (lx, rx) = results
        d = rx["coherence_index"] - rn["coherence_index"]
        print(f"  ΔCI({lx} - {ln}) = {d:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())