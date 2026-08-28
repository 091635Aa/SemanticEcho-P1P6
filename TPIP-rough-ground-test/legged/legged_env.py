#!/usr/bin/env python3
"""
legged_env.py — CPU 微仿真：简化的多关节足式机器人代理模型

特征：
- 6 关节（每条腿 3 关节 × 2 腿）
- 自回归动作生成：a(t) -> dq(t+1) -> q(t+1)
- 三种基座"模型族"：p2p(点对点僵硬) / standard(标准 RL) / transformer(类 Transformer)
- 统一观测：o = [q, dq, goal_dist, terrain_roughness]
"""

from __future__ import annotations
import numpy as np


class BasePolicy:
    """基座策略（只读，模拟三种不同"模型族"）。"""

    def __init__(self, n_joints: int = 6, family: str = "p2p", seed: int = 42):
        assert family in ("p2p", "standard", "transformer")
        self.n_joints = n_joints
        self.family = family
        rng = np.random.default_rng(seed)
        # 线性映射 obs(2*n_joints+2) -> action(n_joints)
        self.W = rng.normal(0, 0.12, (n_joints, 2 * n_joints + 2))
        self.b = rng.normal(0, 0.02, n_joints)

    def forward(self, obs: np.ndarray, t: float = 0.0) -> np.ndarray:
        """返回原始动作（无插件）。obs shape: (2*n_joints+2,)"""
        a = self.W @ obs + self.b
        if self.family == "p2p":
            # 点对点特征：阶跃量化 + 高频 PID 震颤 + 随机重规划
            a = np.round(a / 0.08) * 0.08
            jitter = np.sin(2 * np.pi * 23.7 * t) * 0.05
            a = a + jitter
            # 偶尔"重规划"跳变
            if int(t * 100) % 137 == 0:
                a += np.random.default_rng(int(t * 1000)).normal(0, 0.2, self.n_joints)
        elif self.family == "standard":
            # 标准 RL：已学会周期步态，但相位有轻微漂移 + 小幅噪声
            ph = 2 * np.pi * 1.0 * t + np.linspace(0, np.pi, self.n_joints)
            periodic = 0.35 * np.sin(ph)                       # 周期步态主成分
            drift = 0.03 * np.sin(2 * np.pi * 0.3 * t)          # 慢漂移
            noise = np.random.default_rng(int(t*1000)).normal(0, 0.02, self.n_joints)
            a = np.tanh(a) * 0.2 + periodic + drift + noise
        else:  # transformer：周期步态更精准，但有小幅高频纹波
            ph = 2 * np.pi * 1.0 * t + np.linspace(0, np.pi, self.n_joints)
            periodic = 0.40 * np.sin(ph)
            ripple = 0.02 * np.sin(2 * np.pi * 15.0 * t)        # 高频纹波
            a = np.clip(a, -0.1, 0.1) + periodic + ripple
        return a.astype(float)

    @property
    def obs_dim(self) -> int:
        return 2 * self.n_joints + 2


class GaitPhaseBlueprint:
    """共享全局步态相位蓝图（电路B 的平移）。"""

    def __init__(self, n_joints: int = 6, freq: float = 1.0):
        self.n_joints = n_joints
        self.freq = freq
        # 各关节相位偏移，制造交替步态
        self.phase_offsets = np.linspace(0, np.pi, n_joints)

    def action(self, t: float) -> np.ndarray:
        phi = 2 * np.pi * self.freq * t
        return 0.25 * np.sin(phi + self.phase_offsets)

    def contact(self, t: float) -> np.ndarray:
        """足底触地标志（布尔）。"""
        phi = (2 * np.pi * self.freq * t + self.phase_offsets) % (2 * np.pi)
        return (phi < np.pi).astype(float)  # 0=摆动, 1=支撑


class LeggedMicroSim:
    """简化的自回归动力学：a -> dq += a*dt, q += dq*dt。"""

    def __init__(self, base: BasePolicy, plugins=None,
                 T: int = 800, dt: float = 0.01, seed: int = 42,
                 obs_noise: float = 0.0, act_noise: float = 0.0,
                 obs_spike_p: float = 0.0, obs_spike_amp: float = 0.0,
                 act_spike_p: float = 0.0, act_spike_amp: float = 0.0):
        self.base = base
        self.plugins = plugins or []
        self.T = T
        self.dt = dt
        self.obs_noise = obs_noise      # 传感器观测噪声 σ（插件可感知：进入策略→动作）
        self.act_noise = act_noise      # 执行噪声 σ（插件不可感知：注入后叠加）
        self.obs_spike_p = obs_spike_p  # 传感器异常读数离群概率（插件可感知）
        self.obs_spike_amp = obs_spike_amp
        self.act_spike_p = act_spike_p  # 执行侧异常概率（插件不可感知）
        self.act_spike_amp = act_spike_amp
        self.rng = np.random.default_rng(seed)
        self.blueprint = GaitPhaseBlueprint(base.n_joints)

    def _obs(self, q, dq, goal, terrain):
        # 传感器噪声：只污染"读数"，不污染"真值状态"（插件可感知）
        obs = np.concatenate([q, dq, [goal, terrain]]).copy()
        if self.obs_noise > 0:
            obs[:2 * self.base.n_joints] += self.rng.normal(
                0, self.obs_noise, 2 * self.base.n_joints)
        # 传感器"异常测量"离群点（插件可感知，模拟异常读数）
        if self.obs_spike_p > 0:
            n = 2 * self.base.n_joints
            sp = self.rng.random(n) < self.obs_spike_p
            obs[:n] += self.rng.choice([-1.0, 1.0], n) * self.obs_spike_amp * sp
        return obs

    def run(self, goal: float = 3.0, terrain: float = 0.3) -> dict:
        n = self.base.n_joints
        q = self.rng.normal(0, 0.05, n)
        dq = np.zeros(n)
        traj = {"q": [], "dq": [], "a": [], "t": [], "contact": []}

        for step in range(self.T):
            t = step * self.dt
            obs = self._obs(q, dq, goal, terrain)
            a = self.base.forward(obs, t)
            bp = self.blueprint.action(t)
            ct = self.blueprint.contact(t)

            # P 插件注入
            for plug in self.plugins:
                a = plug.inject(a, t=t, q=q, dq=dq, blueprint=bp,
                                contact=ct, terrain=terrain,
                                progress=step / self.T)

            # 执行噪声 + 执行侧异常（真实执行器不完美；插件不可感知）
            if self.act_noise > 0:
                a = a + self.rng.normal(0, self.act_noise, n)
            if self.act_spike_p > 0:
                spike = self.rng.random(n) < self.act_spike_p
                a = a + self.rng.choice([-1.0, 1.0], n) * self.act_spike_amp * spike

            # 动力学更新：位置混合型（a=期望关节角，q 一阶跟踪，无正反馈）
            a_clamped = np.clip(a, -0.6, 0.6)
            q_new = q * 0.75 + a_clamped * 0.25
            q_new = np.clip(q_new, -np.pi, np.pi)
            dq = (q_new - q) / self.dt
            q = q_new

            traj["q"].append(q.copy())
            traj["dq"].append(dq.copy())
            traj["a"].append(a.copy())
            traj["t"].append(t)
            traj["contact"].append(ct.copy())

        for k in ("q", "dq", "a", "contact"):
            traj[k] = np.stack(traj[k], axis=0)
        traj["t"] = np.array(traj["t"])
        return traj
