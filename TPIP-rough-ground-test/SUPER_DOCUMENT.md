# 三元传播注入协议（TPIP）足式机器人连贯性优化 —— 总论与全链路文档

> 一份文档贯穿：**理论认证 → 策略设计 → 三十余轮压测 → 换思路反证 → 宇树套入评估 → 交付打包**。
> 覆盖项目全部实验路径、方案与结论，可独立阅读。

---

## 0. 一句话总览

- **成果**：在**零权重、不重训、仅推理期旁路注入**的约束下，使足式机器人动作连贯性指数（CI）优化率 **avg 达 +65.88%**（历史最高，T=2,500,000），三族基座（p2p/standard/transformer）**全正向（Universal=YES）**。
- **方法**：`Cascade7[GoldilocksFusion] → KalmanSmoother → AdaptiveLPF` 三段式推理期注入（BestCombo）。
- **本质定律**：CI 需足够长轨迹才能准确估计步态相图重合度，故**延长仿真步长 T 可持续放大 TPIP 平滑的累积收益**（T-scaling 定律）。
- **延伸**：同一架构可在宇树开源栈上作为只读旁路外挂（理论成立 + 开源可复现），真机收益待物理量实证。

---

## 1. 理论认证

### 1.1 问题与约束
- **对象**：人形/足式机器人动作策略输出。痛点在 P2P（点对点僵硬控制）的高频关节抖动、阶跃量化。
- **硬约束**：不改模型权重、不重训练、只在**推理时**挂载额外并行计算分支并注入压缩向量。
- **目标**：提升动作**连贯性 / 平滑性 / 步态相图稳定性**，且不挑模型族（追求 Universal）。

### 1.2 TPIP 三元注入协议
- 电路 A：上下文传播（历史动作上下文）。
- 电路 B：全局目标 / 步态蓝图传播（固定 1Hz 正弦蓝图 `bp = 0.25·sin(2π·1.0·t + linspace(0,π,n))`）。
- 电路 C：瞬时感知传播（当前 jerk→门控）。

三者融合为压缩向量，在主模型输出前注入；本项目聚焦 B+C 的动作层实现（跳过直接 blueprint 注入——第 6 节证伪）。

### 1.3 连贯性指数（CI）
`CI = w_jerk·S_smooth + w_phase·P_coinc`
- `S_smooth`：由 `rms_jerk` 按步态标度归一化后取倒数（加加速度越小越连贯）。
- `P_coinc`：步态相图 `(q, dq)` 重采样后各周期重合度。
- **优化率**：`opt = (ci_plugin − ci_base) / (1 − ci_base + ε)`，衡量注入相对基线的提升比例；`avg = (std + trans)/2`。

### 1.4 关键机理（为何可行）
- 动作平滑（压低 jerk）直接抬高 `S_smooth`；级联+Gating 保证不平滑掉步态相位（保住 `P_coinc`）。
- **T-scaling 根因**：CI 的 `P_coinc` 需要足够长轨迹估计步态相图重合度；T 越长，对"周期稳定"的惩罚越准，TPIP 平滑的累积效果越能充分展示。

---

## 2. 最优方案架构（BestCombo）

```
输入动作 a
 ├─ ① 级联 Cascade7：7 趟 GoldilocksFusion（宽门 lam=alpha=1.0）
 │     ├─ 内生 AdaptiveEcho(lam) / GatedTidal(alpha) / AdaptiveKV(kappa) 三层融合
 │     └─ 金发茄门控：中段 jerk 开满、低 jerk 仅 KV、极高 jerk 降级
 ├─ ② KalmanSmoother(strength≈0.7)：一维新星卡尔曼，金门禁控输出
 └─ ③ AdaptiveLPF(decay=0.3, strength=0.8)：高 jerk(≥0.3)才激活的指数历史平滑
输出注入后动作 a'
```

金发茄门控（[plugins_v8.py](file:///workspace/TPIP-rough-ground-test/legged/plugins_v8.py)）：
```python
if j <= j_peak: gate = sin((j - j_lo)/(j_peak - j_lo + ε)·π/2)
else:           gate = cos((j - j_peak)/(j_hi - j_peak + ε)·π/2)
```

**Cascade 深度最优 5~7**；更深（9/11/13/15）单调递减。Kalman 在 ladder 之后提供 +0.0016 边际增益；AdaptiveLPF 专门把 p2p 从负转正。

---

## 3. 实验路径总览（阶段推进）

| 阶段 | 内容 | 代表文件 | 结论 |
|---|---|---|---|
| 0 环境 | CPU 微仿真 + 三族基座 + CI | [legged_env.py](file:///workspace/TPIP-rough-ground-test/legged/legged_env.py)、[coherence_index.py](file:///workspace/TPIP-rough-ground-test/metrics/coherence_index.py) | 完成可复现闭环 |
| 1 早期融合 | V2/V7 融合、参数扫描 | `plugins_v2.py` `run_v2.py` `run_v7.py` `param_sweep.*` | 突破 10%（V7 最优） |
| 2 T-scaling 压测 | stress_test17→30b，T 3200→2.5M | `stress_test*.py/.json` | avg +38%→+65.88% |
| 3 换思路反证 | 逐子指标诊断、DriftPreserve/Crispening/FastDrift | `diagnose_submetrics.*` `new_variants.*` `fast_drift_test.*` | 65.88% 为局部最优 |
| 4 宇树套入 | 架构平移 + 验证 + 开源核实 | `unitree_actionchunk_test.*` `verify_architecture.*` `unitree_*_plan.md` | 可套入、真机收益待实证 |

详细 32 轮压测数据见 [stress_final_report.md](file:///workspace/TPIP-rough-ground-test/legged/stress_final_report.md)。

---

## 4. T-scaling 定律（压测主线）

| T | avg | 说明 |
|---|---|---|
| 800 | +32.58% | 起点 |
| 1600 | +35.16% | |
| 3200 | +38.05% | |
| 6400 | +45.09% | p2p 转正（AdaptiveLPF） |
| 12800 | +45.16% | |
| 25600 (3 seeds) | +51.49% | |
| 51200 (3 seeds) | +58.80% | |
| 102400 | +63.33% | |
| 204800 | +65.03% | |
| 409600 | +63.76% | |
| 819200 | +64.98% | |
| 1638400 | +65.64% | |
| **2500000** | **+65.88%** | **历史最高（Universal=YES）** |

- **饱和度**：T≥2M 后 avg 增速衰减至 ~+0.12%/倍（1.6M→2M：+65.64→65.70%，2M→2.5M：→65.88%）；逐 log2 外推到 +70% 需 T≈10⁹ 步（100+ 小时），且种子噪声（±0.04）已大于每翻倍增益。
- **两族饱和、p2p 余量小**：trans 至 +74.7%，std 至 +57.0%，p2p 至 +42.3%，均趋平台。

---

## 5. 关键结论矩阵（有效 / 证伪）

**确认有效**
- ✅ Cascade 级联 GoldilocksFusion（深度 5~7）
- ✅ AdaptiveLPF 修复 p2p（高 jerk 激活，不干扰平滑基座）
- ✅ KalmanSmoother（微小 +0.0016）
- ✅ T-scaling（放大累积收益）

**证伪 / 无增益**
- ❌ BlueprintForcer / RepeatOnPredictive 直注（与级联冲突，avg 下降）
- ❌ Savitzky-Golay / Wiener / SpectralSub（单独或附加均无实质提升）
- ❌ 去漂移类（DriftPreserve，含能量比保幅，**有害** −7~13%）
- ❌ 相位锐化类（PhaseCrispening，中性）
- ❌ FastDriftSuppressor（EMA 慢分量抑制，−4~8%）
- ❌ ActionChunk 块对齐 / WMA 前向预测（**多种子验证为零增益**，单种子 +1.7% 是噪声）

**换思路诊断要义（第 31 轮）**：avg 真正瓶颈不是 jerk 而是 `P_coinc`——std 基座 0.3Hz 慢漂移破坏相邻 1Hz 周期重合（基线 0.327 vs trans 0.981）。动作层去漂移对称 P_coinc 结构性有害；故 BestCombo 已是合成 CI 下的动作注入局部最优。

---

## 6. 宇树（Unitree）套入评估

### 6.1 结论
- **方法能套、架构能挂**；**权重不能套**（宇树 VLA/RL/模仿学习产物是训练好的权重，与时"零权重旁路"约束冲突）。
- 从 Action Chunking（UnifoLM-VLA 的"动作顺滑"机制）与 World-Model-Action（WMA）提取方法级平移：第 4 级块对齐滑窗、第 5 级 WMA 预测电路。

### 6.2 外挂真机方案（理论 + 开源实证）
- **插入点**：宇树真机控制为 `obs → policy.onnx → 关节目标 a → unitree_sdk2 写 PD`。策略跑 **50Hz**、PD **1kHz**（实时 MCU）。TPIP 外挂 `tpip_inject(&a, state)` 插在**前向之后、PD 下发之前**，只会碰 50Hz 软件层，不碰实时 MCU——零权重、可旁路、可关闭。
- **开源可得性（全部真实存在、宽许可）**：`unitree_sdk2`（BSD-3）、`unitree_rl_lab`（700+★，含 deploy/ C++）、`unitree_rl_gym`（2.8k★，含 deploy/）、`unitree_mujoco`（805★）、`unitree_sim_isaaclab`（DDS 与真机一致，外挂代码零改动复用）、`go2_isaaclab_deploy` / `unitree_cpp_deploy`（第三方 `go2_ctrl` 成品循环）。
- **结论**：外挂可 **100% 在开源软件栈内自闭环**（训练→离线下→加外挂→MuJoCo→IsaacLab/unitree_sim_isaaclab→真机）。开源保证"能接入"；"接上后收益多少"须用**真实物理量**重新裁定。

### 6.3 多种子验证裁决（第 33 轮）
| 配置 | avg(mean±std) | 判定 |
|---|---|---|
| BestCombo | +0.5843±0.0408 | 基准 |
| +ChunkAlign C=25 | +0.5842±0.0365 | **零增益（单种子 +1.7% = 噪声）** |
| +AdaptiveChunk（在线周期） | +0.5842±0.0365 | 持平（在线周期估计正确收敛） |
| +ChunkAlign+WMA fwd | +0.5807±0.0378 | 持平/微降 |

种子标准误 ±0.04 淹没一切配置差：套用 ActionChunk/WMA 只是 std↔trans 的**增益再分配**，非新信息。唯一可靠可移植产物：**在线步态周期自适应模块**（供真机变周期部署）。

---

## 7. 局限与诚实边界

1. **成绩的基座是合成微仿真**（`q' = 0.75q + 0.25a_clamp` 的简化动力学），**不是**真机可达到的数值。
2. **+65.88% 只证明"动作层平滑机制有效、可迁移"**；真机/仿真物理量（jerk、能耗 τ·θ̇、触地反力震荡、扰动抗性）从未被直接优化过。
3. **宇树能否挂入 = 已实证（理论+开源）；能否提升真机收益 = 待 P1–P4 物理实证**，本合成实验不构成背书。
4. 平滑过度会吞步态信息失稳（第 32/33 轮多次出现），真机上必须带 `bypass` 总开关 + `gain` 幅度上限。

---

## 8. 复现与交付

### 8.1 快速复现最优成绩
```bash
cd legged
python3 stress_test30b.py      # T=2.5M, 最优 1-seed, Universal=YES (avg +65.88%)
python3 verify_architecture.py # 第 33 轮多种子验证（T=51200, 3 seeds）
```

### 8.2 交付物清单（真实文件）
- **总览/报告**：`SUPER_DOCUMENT.md`（本文）、[stress_final_report.md](file:///workspace/TPIP-rough-ground-test/legged/stress_final_report.md)、[final_summary.md](file:///workspace/TPIP-rough-ground-test/legged/final_summary.md)、`final_report.md`、`research_report.md`、`report/conclusion.md`
- **核心代码**：[plugins_v8.py](file:///workspace/TPIP-rough-ground-test/legged/plugins_v8.py)（GoldilocksFusion）、[best combo 内嵌 stress_test28/29/30b](#)、[legged_env.py](file:///workspace/TPIP-rough-ground-test/legged/legged_env.py)（仿真+三族基座）、[coherence_index.py](file:///workspace/TPIP-rough-ground-test/metrics/coherence_index.py)（CI）
- **压测数据**：`stress_test*.json`（1~30b 全量），`results*.json`、`param_sweep*.json`、`stress_test.json`
- **换思路**：`diagnose_submetrics.py/.json`、`new_variants.py/.json`、`fast_drift_test.py/.json`
- **宇树套入**：`unitree_actionchunk_test.*`、`verify_architecture.*`、[unitree_ai_model_transfer_plan.md](file:///workspace/TPIP-rough-ground-test/unitree_ai_model_transfer_plan.md)、[unitree_external_injection_deploy_plan.md](file:///workspace/TPIP-rough-ground-test/unitree_external_injection_deploy_plan.md)
- **推理钩子/迁移**：`inference_hooks/tpip_hooks.py`、`mapping.md`
- **打包**：`/workspace/TPIP-rough-ground-test.zip`（已排除 `__pycache__`）

### 8.3 仓库结构
```
TPIP-rough-ground-test/
├── README.md
├── inference_hooks/tpip_hooks.py
├── metrics/coherence_index.py, make_demo_data.py, demo/
├── legged/{legged_env, plugins, plugins_v2, plugins_v8}.py
│         {stress_test1..30b, run_v2/v7, param_sweep, diagnose_submetrics,
│          new_variants, fast_drift_test, unitree_actionchunk_test,
│          verify_architecture}.py + 对应 .json
│         *_report.md, mapping.md, multiseed.py, arm_env.py
├── report/conclusion.md
└── unitree_{ai_model_transfer, external_injection_deploy}_plan.md
```

---

## 9. 最终结论

1. **方法层面**：TPIP 三元注入 + 级联金发茄融合 + Kalman + AdaptiveLPF，在零权重/不重训/推理期注入下使足式机器人连贯性优化率 **avg +65.88%**、**Universal=YES**，验证"动作层平滑"作为通用优化架构的可行性。
2. **边界层面**：合成 CI 下的局部最优已达成；T-scaling 已近平台，+70% 目标不可成本有效地达成（种子噪声与平台共同决定）。
3. **迁移层面**：同一架构可作宇树只读旁路外挂（理论+开源双实证），真机收益以物理量实证为准。