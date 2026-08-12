# Tasks

- [ ] Task 1: 构建 P6 训练数据管线
  - [ ] 1.1 读取 emotion_dataset.jsonl(4409) + gentle_dataset.jsonl(80)，转换为 QA chat 模板格式
  - [ ] 1.2 生成人味对话增强语料（girl 风格改写，不与评测样本逐字重复）
  - [ ] 1.3 输出 `data/p6_train.jsonl`（instruction/response 对）
- [ ] Task 2: 训练 P6 情感 LoRA（Qwen2.5-1.5B）
  - [ ] 2.1 编写训练脚本（复用 train_emotion_lora.py 基建，r=8/α=16 起）
  - [ ] 2.2 执行训练，输出 `lora_adapters/p6_emotion/`
  - [ ] 2.3 冒烟：加载适配器，人工检查 5 条情感回复质量
- [ ] Task 3: 构建 P6 旁路由选优生成器
  - [ ] 3.1 实现情感路由评分器（情感命中/AI腔惩罚/重复/长度/熵）
  - [ ] 3.2 实现多候选生成 + 选优 + 裸采样兜底
  - [ ] 3.3 冒烟测试 10 条样本，检查健康度
- [ ] Task 4: P6 统一评测（30 条 + 7B 裁判）
  - [ ] 4.1 生成 30 条 P6 回复缓存（种子 2026）
  - [ ] 4.2 7B 裁判 AB 配对盲评（60 配对），输出 win_rate
  - [ ] 4.3 汇总对比 裸/PE/P1.5/P2.5/P3/P4/P5/G
- [ ] Task 5: 达标判定与迭代优化
  - [ ] 5.1 若 win_rate ≥ 0.48 且情感命中 ≥ 0.20 → 归档完成
  - [ ] 5.2 若未达标 → 分析失败样本 → 调数据/超参/路由权重 → 重测（≤5 轮）
- [ ] Task 6: 方案归档
  - [ ] 6.1 撰写《P6 实验报告》（方案/数据/结果/对比/结论）
  - [ ] 6.2 更新 P6 设计文档 v1.1（含实验结果）

# Task Dependencies
- Task 2 依赖 Task 1
- Task 3 依赖 Task 2
- Task 4 依赖 Task 3
- Task 5 依赖 Task 4
- Task 6 依赖 Task 5
