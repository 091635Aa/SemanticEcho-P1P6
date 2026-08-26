#!/usr/bin/env python3
"""
arm_env.py — 服务型机器人臂操作微仿真

6-DOF 机械臂执行"取物-放置"往返任务。
三种基座策略：
  p2p_arm   — 阶跃式点到点运动（无中间规划，最短路径直线，速度跳变）
  pid_arm   — PID 位置控制（有过冲和振荡）
  learned   — 学习型策略（有轨迹规划但残留小幅抖动）

测试 P1~P6 / V1~V6 能否平滑臂端轨迹，提升操作连贯性。
"""

from __future__ import annotations
import numpy as np


class ArmBasePolicy:
    """6-DOF 臂基座策略。"""
    def __init__(self, n_joints=6, family="p2p_arm", seed=42):
        self.n_joints = n_joints
        self.family = family
        self.rng = np.random.default_rng(seed)
        # 两个目标点（往返取放）
        self.target_a = np.array([0.5, 0.3, -0.2, 0.1, 0.4, -0.3])
        self.target_b = np.array([-0.3, -0.4, 0.3, -0.2, -0.1, 0.5])
        self.switch_period = 200  # 每 200 步切换目标（2秒）

    def _target(self, t):
        cycle = int(t * 100) // self.switch_period
        return self.target_a if cycle % 2 == 0 else self.target_b

    def forward(self, q, dq, t):
        target = self._target(t)
        error = target - q
        if self.family == "p2p_arm":
            # 阶跃式：直接大步冲向目标，速度跳变
            a = np.sign(error) * np.minimum(np.abs(error) * 3.0, 0.8)
            # 量化（关节离散控制）
            a = np.round(a / 0.05) * 0.05
        elif self.family == "pid_arm":
            # PID：有比例增益 + 残留振荡
            a = error * 2.0 + dq * 0.05
            osc = 0.08 * np.sin(2 * np.pi * 4.0 * t + np.arange(self.n_joints))
            a = a + osc
        else:  # learned
            # 学习型：轨迹规划 + 残留抖动
            progress = (t * 100 % self.switch_period) / self.switch_period
            smooth = 3 * progress**2 - 2 * progress**3  # smoothstep
            a = error * (1.0 + 0.5 * smooth)
            jitter = 0.03 * np.sin(2 * np.pi * 12.0 * t + np.arange(self.n_joints))
            a = a + jitter
        return a.astype(float)


class ArmMicroSim:
    """臂动力学：a = 期望关节速度，一阶跟踪。"""
    def __init__(self, base, plugins=None, T=800, dt=0.01, seed=42):
        self.base = base
        self.plugins = plugins or []
        self.T = T
        self.dt = dt
        self.rng = np.random.default_rng(seed)

    def run(self):
        n = self.base.n_joints
        q = np.zeros(n)
        dq = np.zeros(n)
        traj = {"q": [], "dq": [], "a": [], "t": []}
        for step in range(self.T):
            t = step * self.dt
            a = self.base.forward(q, dq, t)
            # 生成平滑蓝图：smoothstep 插值到当前目标
            target = self.base._target(t)
            progress = (t * 100 % self.base.switch_period) / self.base.switch_period
            smooth = 3 * progress**2 - 2 * progress**3
            blueprint = target * smooth + q * (1 - smooth) * 0.1
            for plug in self.plugins:
                a = plug.inject(a, t=t, q=q, dq=dq, blueprint=blueprint,
                                contact=None, terrain=0.0,
                                progress=step / self.T)
            # 一阶位置跟踪
            a_clamped = np.clip(a, -1.0, 1.0)
            q_new = q * 0.8 + (q + a_clamped * self.dt) * 0.2
            dq = (q_new - q) / self.dt
            q = q_new
            traj["q"].append(q.copy())
            traj["dq"].append(dq.copy())
            traj["a"].append(a.copy())
            traj["t"].append(t)
        for k in ("q", "dq", "a"):
            traj[k] = np.stack(traj[k], axis=0)
        traj["t"] = np.array(traj["t"])
        return traj
