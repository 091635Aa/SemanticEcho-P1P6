# P6（情感LoRA外挂×旁路由选优）Spec

## Why
用户要求研发 P6 模型：必须显著优于前辈 PE(P1)→G(统一版) 全部方案（要求1），底层思路与前辈本质不同（要求2），情感输出能力优化 80% 以上（要求3），前奏不可修改、允许 LoRA/旁路由扩展（要求7），全自主迭代完成（要求5/6/9/10），最终提交完整方案与实验结果（要求11）。

## What Changes
- 新建 P6 训练数据管线（情感 4409 + 温柔 80 + 人味增强，chat 格式）
- 训练情感 LoRA 适配器 `lora_adapters/p6_emotion/`（Qwen2.5-1.5B 基座，q/k/v/o，r/α 可扫描）
- 新建 P6 旁路由生成器：多候选生成（N=3 种子）+ 情感路由评分选优 + 裸采样兜底
- 新建 P6 统一评测脚本：30 条样本 + 种子 2026 + 7B 裁判配对盲评，对比 裸/PE/P1.5/P2.5/P3/P4/P5/G
- 迭代优化至 win_rate ≥ 0.48（相对 G 最佳 P2.5 的 0.2667 提升 ≥ +80%）
- **BREAKING**：不修改模型前奏（system prompt/chat 模板）；基座权重零改动

## Impact
- Affected specs: 无既有 spec；新增 P6 架构（权重空间 + 路由空间）
- Affected code:
  - 训练：`f:\lora外挂\training_scripts\`（复用 train_emotion_lora.py 基建）
  - 生成/评测：`i:\Desktop\语义回响\图灵测试\统一基准\`、`c:\Users\Administrator\Documents\KV 情感共振解码\核心\P5裁判_30条.py`（复用裁判基建）
  - 模型：`c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct`（目标）、`Qwen2.5-7B-Instruct`（裁判）

## ADDED Requirements

### Requirement: P6 情感LoRA训练
系统 SHALL 构建 chat 格式训练语料并训练情感 LoRA 适配器。
#### Scenario: 训练成功
- **WHEN** 运行训练脚本
- **THEN** 输出 `lora_adapters/p6_emotion/`（adapter_model.safetensors + adapter_config.json），基座权重不变

### Requirement: P6 旁路由选优生成
系统 SHALL 在主路径零注入的前提下，用 LoRA 挂载模型生成 N 候选，并按情感路由评分选出最优回复。
#### Scenario: 正常生成
- **WHEN** 输入用户消息
- **THEN** 输出情感路由选优后的回复（20~80 字、重复率≤0.02、无 AI腔残留）
#### Scenario: 全部候选不合格
- **WHEN** 所有候选情感分低于阈值
- **THEN** 降级为裸采样兜底，保证不坍缩

### Requirement: P6 统一评测
系统 SHALL 在统一基准（30 条、种子 2026、7B 裁判 AB 配对）上对比 P6 与全体前辈。
#### Scenario: 评测完成
- **WHEN** 评测脚本运行结束
- **THEN** 输出 P6 win_rate、健康度（熵/重复率/命中率/长度），并与 裸/PE/P1.5/P2.5/P3/P4/P5/G 对比

### Requirement: P6 达标判定
P6 win_rate SHALL ≥ 0.48（相对 G 最佳 +80%），情感命中率 ≥ 0.20，且 win_rate 高于全体前辈。
#### Scenario: 达标
- **WHEN** 评测结果显示 win_rate ≥ 0.48
- **THEN** 归档 P6 最终方案与实验报告

## MODIFIED Requirements
无（P6 为全新架构，不修改前辈方案）

## REMOVED Requirements
无
