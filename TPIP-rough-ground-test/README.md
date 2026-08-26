# TPIP-rough-ground-test

**推理期三元传播注入协议（TPIP）**在 **崎岖地面 (rough-ground，含 3cm 微型凸起)** 场景下的验证工程。

三路并行电路在推理时协作，仅在主模型输出第一个动作 token 的 **前一刻（P1.5）** 做一次门控融合注入 —— **不修改基座权重、不梯度回传、不重训**。

## 目录

```
.
├── inference_hooks/
│   └── tpip_hooks.py          # 三路电路拼接 + P1.5 门控融合注入伪代码(基座冻结)
├── metrics/
│   ├── coherence_index.py     # 连贯性指数: jerk平滑度 + 步态相图重合度
│   ├── make_demo_data.py      # 生成演示对比轨迹 (demo/*.npz)
│   └── demo/
│       ├── base.npz           # 基线(点对点体式) 合成示例轨迹
│       └── tpip.npz           # 扩展(连贯摆线) 合成示例轨迹
└── report/
    └── conclusion.md          # 三问答结论 (Q1/Q2/Q3)
```

## 快速自检

```bash
# 度量工具自检：证明能分辨"点对点"vs"连贯"
python3 metrics/coherence_index.py --selftest

# 生成并对比演示轨迹
python3 metrics/make_demo_data.py demo
python3 metrics/coherence_index.py --data demo/base.npz --data demo/tpip.npz --labels baseline tpip
```

## 三问答（详见 report/conclusion.md）

1. **是否显著降低"点对点"特征？** 条件性肯定 —— 机制上通过门控融合+A/B/C 预补偿压低震颤/尖峰/非必要重规划；本库给出可分辨工具证据，真实仿真定量待 GPU 环境补跑。
2. **是否需要重训基座？** **否**。基座全冻结、梯度零回传。
3. **是否具备真实控制器实时性潜力？** 有条件满足 —— 注入为每周期单次小型 MLP 开销，满足实时前提，需目标硬件单帧延迟实测。

## 复现/取证（真实 Isaac Gym 对照）

见 `report/conclusion.md` 第 4 节；核心理念：**同起始状态、唯一变量 = 是否挂载 TPIP**，用连贯性指数 ΔCI、加加速度 rms、ZMP 跳变、重规划事件数四项判据在 P<0.05 下对比。

## 研究参数空间（非强制，供探索）

- 压缩向量维度 `latent_dim ∈ {64, 128, 256, 512}`（默认 256）
- 门控融合比率 `α:β:γ`（主线保留 / 全局目标 / 瞬时感知）
- 铁律：**绝不触碰基座任何一层权重**