# 宇树（Unitree）AI 模型向 TPIP 足式机器人连贯性优化的移植方案

> 日期：2026-08-26
> 来源项目：TPIP-rough-ground-test（6 关节双足 CPU 微仿真，BestCombo 已达 avg=+65.88%）
> 目的：评估宇树开源 AI 模型能否套用——**能套的是方法，不能套的是权重**。

---

## 一、结论速览

| 维度 | 判定 |
|---|---|
| **权重级套用（VLA/RL/模仿学习产物）** | ❌ 不可行（与"零权重、不重训、推理期挂载"核心约束冲突） |
| **方法级套用 — Action Chunking** | ✅ **可行，已实测微增** avg +1.7%（C=1/4 周期对齐滑窗） |
| **方法级套用 — World Model 预测电路** | 🔶 理论可行，需更大 T/种子验证（架构已给出） |
| 对现有最优 avg 的影响 | 追加级（+1.7% 单种子，落在噪声带内），不构成突破声明 |

---

## 二、宇树模型盘点与套用判定

| 模型/框架 | 类型 | 是否可套 | 理由 |
|---|---|---|---|
| UnifoLM-VLA-0 | VLA 大模型 | 权重❌/方法✅ | 靠 **Action Chunking（动作分块预测）** 达到"顺滑自然"，机制可平移到推理期电路 |
| UnifoLM-WMA-0 | World-Model-Action | 方法✅ | 世界模型"预测→动作"，对应本项目步态蓝图前向预测的强化版 |
| WVLA 2.0 | 世界模型 VLA | 权重❌ | 偏长时序任务/视觉语言，非底层连贯 |
| OmniXtreme | flow 策略+残差 RL | 思路✅ | **预训练+轻量残差后训练**的两阶段解耦思想，可作"注入幅度自调"启发 |
| unitree_rl_lab/mjlab/gym | RL 运动控制 | ❌ | 需完全重训策略网络 |
| unitree_IL_lerobot | 模仿学习（DP/ACT） | ❌ | 训练式，冲突 |

**结论**：宇树能平移的是"动作块相干决策"（Action Chunking）与"预测驱动注入"（World Model）两个方法级思想。

---

## 三、套用架构（推理期，不改权重/不重训）

在原 BestCombo Pipeline 后追加两级，形成 **4 级推理期注入架构**：

```
输入 a ──→ ①Cascade7[GoldilocksFusion×7] ──→ ②KalmanSmoother ──→ ③AdaptiveLPF
   ──→ ④ ActionChunk 块对齐滑窗（新增）──→ ⑤ 轻量 WMA 预测微调（新增/可选）
```

**④ ActionChunk 块对齐滑窗**（已实现并实测 = `ActionChunkAlign`）
```
维护近 C 步动作 buffer；chunk_mean = 滑窗均值（块参考）
out = a + strength·gate(jerk)·(chunk_mean − a)
约束：C < 步态周期一半（实测 C=25≈1/4 周期有效；C=50 吞相位失效）
```
- 语义：近似 VLA 的"一个 chunk 是连贯动作片断"，用**重叠滑窗**保证块间无跳变；
- 不僵化：区别于开环块播放 → 保护 P_coinc（实测证明）。

**⑤ 轻量 WMA 预测微调**（可选，未实测，理论设计）
```
用最近 k 步轨迹在线拟合下一个步态周期的相位/幅值，生成小幅预测 Δ_pred；
out += β·gate·Δ_pred（β 很小，仅微调相位超前量）
```
- 思想来自 WMA-0，但**不引入网络权重**，仅在线最小二乘拟合，保持零重训约束。

---

## 四、实测数据（第 32 轮，T=51200, seed=42）

| 配置 | std | trans | **avg** | p2p | Universal |
|---|---|---|---|---|---|
| BestCombo（ref） | +0.5039 | +0.5608 | **+0.5324** | +0.4433 | YES |
| ref + ChunkAlign C=25 s=0.6 | +0.4955 | **+0.5878** | **+0.5417** ✅ | +0.4433 | YES |
| ref + ChunkPlayback C=25 s=0.6 | +0.1176 | +0.0562 | +0.0869 ❌ | +0.4433 | YES |
| ref + ChunkAlign C=50 s=0.6 | +0.3668 | +0.4402 | +0.4035 ❌ | +0.4433 | YES |

- **ChunkAlign C=25 是首个能超 ref 的追加机制**（第 31 轮 DriftPreserve/Crispening 全失效后的正向信号）；
- 增益集中在 **trans**（+0.5608→+0.5878，抬 2.7pp），std 微降（−0.8pp），净 +1.7%；
- 单种子且落在 ±1% 噪声带内，**尚不足以宣称突破 70%**。

---

## 五、理论论证

**Q1：为什么 action chunking（分块相干）在推理期能提升连贯性？**
- 步态是 1Hz 周期信号，`CI = w·S_smooth + w·P_coinc`。块内动作来自同一"平滑参考"，块内高频抖动被均值化 → **rms_jerk↓ → S_smooth↑**；
- 采用重叠滑窗（非硬分块）→ 相邻瞬时不产生跳变，步态相位相干不被破坏 → **P_coinc 保持**；
- 所以 C=25 净增益 = S_smooth 抬升 − P_coinc 少量损耗。

**Q2：为什么 C=50 和 ChunkPlayback 失败？**
- C=50=半周期：滑窗宽度吞掉半个 1Hz 周期 → 把步态本身的相位结构也抹掉 → **P_coinc 崩**（std 0.49→0.37）。这是"宽低通磨相位"（第 31 轮）在块域的复现；
- ChunkPlayback：机械保持块参考不变 = **开环僵化**，VLA 在训练期学到的 block 表征，这里硬搬为恒定参考 → 相位错位 → P_coinc 雪崩。**证明不能机械平移 VLA 的开环 chunk 播放，只能平移"块内相干"的低通近似**。

**Q3：为什么权重级模型不可套？**
- 三种瓶颈模型均为训练产物：换装机即更换基座，不再满足"不动模型、推理期挂载压缩向量"的 TPIP 定义；tensor 大小/计算量也与 CPU 微仿真不匹配。

---

## 六、局限与建议下一步

**局限**：合成微仿真 + 单种子 + 增量落在噪声带；指标为合成 CI，需 Isaac/Lab 复核；C、strength、插入位置未做网格搜索。

**下一步（按性价比排序）**：
1. ChunkAlign 在**最佳 T（102400）下用 3 seeds × C∈{20,25,30}** 确认 +1.7% 是否显著；
2. 若显著，把 C 对齐到**实测步态周期**（而非硬编码 25），自适应更大 T；
3. 实现轻量 WMA 预测微调电路，验证相位超前注入是否抬 std P_coinc。
4. 任一确认后重跑 T=2500000 挑战 +65.88% 记录。

---

## 七、结论

宇树 Action Chunking 的"块内动作相干"思想**可平移为推理期第 4 级电路**并已实测出首个正向增量（+1.7%）；但其实现必须用**重叠对齐滑窗**且块长 < 步态周期一半，**不可**机械地套用 VLA 的开环块播放。World Model 预测电路为可选第 5 级，理论可行未实测。权重级宇树模型一律不可套用。

**测试脚本**：[unitree_actionchunk_test.py](file:///workspace/TPIP-rough-ground-test/legged/unitree_actionchunk_test.py)
**原始数据**：[unitree_actionchunk_test.json](file:///workspace/TPIP-rough-ground-test/legged/unitree_actionchunk_test.json)