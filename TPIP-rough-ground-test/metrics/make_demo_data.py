#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成演示用轨迹 npz (base=基线P2P, tpip=扩展连贯) 供 --data 对比模式使用。

用法:
  python3 make_demo_data.py ./demo
生成 ./demo/base.npz 与 ./demo/tpip.npz (q, dq, q_ddot, zmp, foot_contact)。

注意: 此为合成演示数据, 仅用于演示度量工具, 非 Isaac Gym 真实仿真采集。
"""
from pathlib import Path
import numpy as np
import coherence_index as m

_OUT = ["q", "dq", "q_ddot", "zmp", "foot_contact"]


def _zmp(contact):
    """合成 ZMP 落点样本: 支撑相连续, 摆动相短暂过渡(演示用)。"""
    z = np.full(len(contact), 0.0)
    in_swing = np.ones(len(contact), dtype=bool)
    # 简单模拟: 每 0.1s 在 ±0.03m 间切换(连贯式更平滑)
    for i in range(len(contact)):
        z[i] = 0.02 * np.sin(i * 0.5)
    return z


def _contacts(T, freq_hz=2.0, dt=0.01):
    """双足触地标志(演示): 频率支撑/摆动切换。"""
    phase = np.arange(T) * dt * freq_hz % 1.0
    return (phase < 0.6).astype(int)  # 60% 支撑


def main(out_dir: str = "demo"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dt = 0.01
    T = 800

    specs = {"base.npz": m._synth_p2p(T, dt), "tpip.npz": m._synth_coherent(T, dt)}
    for name, q in specs.items():
        dq = m.central_diff(q, dt)
        qd = m.central_diff(dq, dt)
        fc = _contacts(T)
        np.savez(out / name,
                 q=q[:, None], dq=dq[:, None], q_ddot=qd[:, None],
                 zmp=_zmp(fc), foot_contact=fc)
        print("wrote", out / name)
    print("done. 下一步:\n python3 coherence_index.py --data demo/base.npz --data demo/tpip.npz --labels baseline tpip")


if __name__ == "__main__":
    raise SystemExit(main())