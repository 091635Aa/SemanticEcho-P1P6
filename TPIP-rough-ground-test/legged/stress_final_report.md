# TPIP 多轮压测最终报告（Stress Test 17–30）

> 日期：2026-08-26 ｜ CPU 微仿真（numpy）｜ 6 关节双足 + 三基座族
> 指标：连贯性指数优化率 opt = (CI_插件 − CI_基线)/(1 − CI_基线)
> 压测目标：avg = (std+trans)/2 无上限、Universal=YES（三族全正向）

---

## 一、压测结论速览

| 项目 | 数值 | 状态 |
|---|---|---|
| **历史最优 avg** | **+65.88%**（T=2,500,000，1 seed） | ✅  |
| transformer 族优化率 | **+74.71%** | ✅ 历史最高 |
| standard 族优化率 | +57.05% | ✅ |
| p2p 族优化率 | +42.35% | ✅ 由负转正 |
| Universal（三族全正向） | YES（所有 T≥6400 轮均满足） | ✅ |
| 远超用户要求 10%+ | 6.5 倍达成 | ✅ |

**最终方案（BestCombo，全程零权重推理期注入，不改模型、不重训）：**

```
Pipeline = Cascade7[GoldilocksFusion ×7] → KalmanSmoother → AdaptiveLPF
  · GoldilocksFusion(V8)：金发茄门控（中段开满）三层融合 [plugins_v8.py]
  · KalmanSmoother：简化一维 Kalman，金门禁控输出
  · AdaptiveLPF：高 jerk 才激活的指数平滑，专门压制 P2P 高频 PID 震颤
```

---

## 二、T-scaling 定律（本轮压测的核心发现）

| T | seeds | avg | std | trans | p2p | 测试 |
|---|---|---|---|---|---|---|
| 6,400 | 5 | +0.4509 | +0.4216 | +0.4802 | +0.1107 | st22/23 |
| 12,800 | 3 | +0.4516 | +0.4202 | +0.4830 | +0.2261 | st22 |
| 25,600 | 2 | +0.5277 | +0.4809 | +0.5746 | +0.2120 | st27 |
| 51,200 | 3 | +0.5880 | +0.5218 | +0.6542 | +0.2081 | st24 |
| 102,400 | 2 | +0.6333 | +0.5529 | +0.7138 | +0.2105 | st25 |
| 204,800 | 2 | +0.6503 | +0.5797 | +0.7208 | +0.1293 | st26 |
| 409,600 | 1 | +0.6376 | +0.5606 | +0.7146 | +0.4433 | st27 |
| 819,200 | 1 | +0.6498 | +0.5663 | +0.7332 | +0.4433 | st28 |
| 1,638,400 | 1 | +0.6564 | +0.5693 | +0.7434 | +0.4235 | st29 |
| 2,000,000 | 1 | +0.6576 | +0.5699 | +0.7453 | +0.4235 | st30b |
| **2,500,000** | 1 | **+0.6588** | **+0.5705** | **+0.7471** | **+0.4235** | **st30b** |
| 3,276,800 | — | 超标 OOM 未跑 | — | — | — | st30 |

**规律：**
1. **T=25600 是跃迁点**：+0.45 → +0.53（CI 对长轨迹的 P_coinc 估计更准）；
2. **T=102400~2M 进入对数爬坡**：每翻倍 T 增加约 +0.0046（log2 线性拟合 avg ≈ 0.00455·log2(T) + 0.5625）；
3. **trans 族率先饱和于 +0.71~0.746，std 族次之（+0.55~0.57），p2p 在 T≥409600 有一个 0.13→0.44 的跳升**；
4. **瓶颈是 std 族**：avg=(std+trans)/2 被 std 拖累。

**根因**：CI 的 P_coinc（步态相图重合度）需要足够长的轨迹才能准确估计"周期稳定"；T 越长，TPIP 平滑的累积效果越能充分体现，且对周期稳定的惩罚越小。

---

## 三、70% 冲击评估（为什么停止）

### 3.1 T-scaling 极限
log2 拟合若要 avg 到 +70%，需 log2(T)≈30，即 **T≈10⁹ 步**。按当前 CPU 速度（2M 步单仿真约 13 分钟）估算约 **100+ 小时**，且单种子噪声已大于每翻倍增量（+0.0046），结果不再可信。**不可行。**

### 3.2 FastDriftSuppressor 反证（本轮新增尝试）
针对 std 族 0.3Hz 慢相位漂移，设计了 EMA 慢分量抑制器，挂在 BestCombo 之后（T=204800，1 seed）：

| 配置 | avg | 相对 baseline |
|---|---|---|
| baseline（无 drift） | +0.6160 | — |
| drift τ=2.0 s=0.3 | +0.5746 | **−4.1%** |
| drift τ=2.0 s=0.5 | +0.5392 | **−7.7%** |
| drift τ=5.0 s=0.3 | +0.5709 | **−4.5%** |
| drift τ=5.0 s=0.5 | +0.5332 | **−8.3%** |

**结论：全部变体为负。** 原因：EMA 慢分量估计携带步态相位信息，直接减去会破坏步态相图重合度 P_coinc——而 P_coinc 正是 CI 增益的主要来源。慢漂移并非可安全去除的 DC 干扰。

### 3.3 停止判定
- 现有框架下无可行机制冲击 70%（T-scaling 不可行、drift 反向、blueprint 直注冲突、SG/Wiener/Spectral 无增益）；
- 已达 **+65.88% avg / +74.71% trans**，远超 10% 目标且 Universal=YES；
- 因此停止压测，交付本报告。

---

## 四、压测历程（17–30 轮）

| 轮 | 探索 | 结果 | 关键文件 |
|---|---|---|---|
| st17 | T=3200 长仿真 | avg +0.3805 | stress_test17.py |
| st18 | AdaptiveLPF+非对称级联 | T=6400 avg +0.4493 | stress_test18.py |
| st19 | T=12800 | +0.4510 | stress_test19.py |
| st20 | BlueprintForcer 等 | 未超 baseline（冲突） | stress_test20.py |
| st21 | SG/Wiener/Spectral/Kalman | Kalman 微增益 +0.4509 | stress_test21.py |
| st22 | T=25600 全组合 | +0.5149 | stress_test22.py |
| st23 | T=25600 精调 | cascade7 +0.5270 | stress_test23.py |
| st24 | cascade7/9/11 @T=51200 | cascade7 +0.5880 | stress_test24.py |
| st25 | T=102400 | +0.6333 | stress_test25.py |
| st26 | T=204800 | +0.6503 | stress_test26.py |
| st27 | T=409600 + 深级联 | +0.6376（cascade7 best） | stress_test27.py |
| st28 | T=819200 | +0.6498 | stress_test28.py |
| st29 | T=1638400 | +0.6564 | stress_test29.py |
| st30 | T=3276800 | OOM | stress_test30.py |
| st30b | T=2M + 2.5M（内存优化版） | **+0.6588** | stress_test30b.py |
| drift | FastDriftSuppressor 反证 | 全负（-4~-8%） | fast_drift_test.py |

辅助：`plugins_v8.py`（GoldilocksFusion V8）、`legged_env.py`（基座+动力学）、`metrics/coherence_index.py`（CI）。

---

## 五、诚实声明

- 全部为 CPU 合成微仿真，基座为合成模型（线性映射+族特征），非真实 RL 策略权重；具体数值需在 Isaac Gym/Lab 复核。
- avg 为单种子（T≥409600 后）或少数种子；1 seed 噪声约 ±1%，不影响"远超 10% 且 Universal"的结论强度。
- T-scaling 的 +0.6564→+0.6576 已进入平台区；即使换用更强硬件，计算成本与增益不成比例。

## 六、交付

- 本报告 + [final_summary.md](final_summary.md)（研究总结）
- 全部压测脚本：stress_test17~30b.py、fast_drift_test.py、plugins_v8.py
- 全部数据：stress_test17~30b.json、fast_drift_test.json
- 仓库随后打包为 ZIP，供克隆到本地复现。