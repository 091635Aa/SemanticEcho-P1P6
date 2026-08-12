# -*- coding: utf-8 -*-
"""生成 P1→P4 总结报告所需的对比图表（新编号：P1含P1.5 → P2 ETD → P3 AE → P4 KER）"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "图表")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)


# ── 图1：P1 λ 扫描 U 型曲线（E1–E6）────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 4.2))
labels = ["E1 裸\n(top_p=0.9)", "E2 裸\n(temp=1.0)", "E3\nλ=0.5", "E4\nλ=1.0", "E6\nλ=1.0,γ小", "E5\nλ=2.0"]
vals = [1.8011, 3.7931, 2.124, 0.7199, 0.6921, 0.1914]
colors = ["#7f8c8d", "#7f8c8d", "#2ecc71", "#e67e22", "#e67e22", "#e74c3c"]
bars = ax.bar(labels, vals, color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylabel("平均语义熵")
ax.set_title("P1 语义回响：λ 注入强度扫描（Qwen2.5-0.5B fp16，E1–E6）")
ax.axhline(1.8011, color="#2c3e50", ls="--", lw=0.8)
ax.text(5.35, 1.85, "E1 基线 1.8011", fontsize=8, color="#2c3e50", ha="right")
ax.set_ylim(0, 4.3)
save(fig, "图1_λ扫描U型曲线.png")

# ── 图2：P3（锚点回响）五基准相对提升 ─────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 4.2))
bases = ["HeartBench\noverall", "TuringBench\n人似度", "LLM-Judge\n+人设", "HEART-BENCH\n一致性", "EmoCharacter\nfidelity"]
deltas = [21.7, 200.0, 56.25, 8.3, -2.7]
colors = ["#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#e74c3c"]
bars = ax.bar(bases, deltas, color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)
for b, v in zip(bars, deltas):
    ax.text(b.get_x() + b.get_width() / 2, v + (5 if v >= 0 else -8), f"{v:+.1f}%", ha="center", fontsize=9)
ax.axhline(0, color="#2c3e50", lw=1)
ax.set_ylabel("相对裸模型提升 (%)")
ax.set_title("P3 锚点回响（原P4）：五基准评测相对提升（Qwen2.5-1.5B）")
ax.set_ylim(-15, 230)
save(fig, "图2_P3五基准提升.png")

# ── 图3：多模型泛化（LLM-Judge 相对裸提升）────────────────────
fig, ax = plt.subplots(figsize=(8.0, 4.2))
models = ["1.5B\nP2.5潮汐", "1.5B\nP2.5混合", "3B\nP2.5潮汐", "1.5B\nP3纯净", "1.5B\nP3+人设", "Qwen3-1.7B\nP3", "gemma-2b\nP3兜底", "1.5B\nP4双通道"]
vals3 = [37.5, 87.5, 33.0, 12.5, 56.25, 6.91, 28.53, 23.0]
colors = ["#e67e22", "#e74c3c", "#f39c12", "#3498db", "#9b59b6", "#2ecc71", "#1abc9c", "#8e44ad"]
bars = ax.bar(models, vals3, color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)
for b, v in zip(bars, vals3):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:+.1f}%", ha="center", fontsize=8.5)
ax.axhline(0, color="#2c3e50", lw=1)
ax.set_ylabel("LLM-Judge win_rate 相对提升 (%)")
ax.set_title("多模型泛化：P2.5 / P3 / P4 相对裸模型提升（20/30 样本，P4 独立种子）")
ax.set_ylim(-5, 100)
save(fig, "图3_多模型泛化对比.png")

# ── 图4：方案关键成果总览（最终编号 P1→P1.5→P2.5→P3→P4→P5）────────────
fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.8))
ax = axes[0, 0]
names = ["P1", "P1.5", "P2.5\n潮汐", "P3\n锚点", "P4\nKV", "P5\n超融合"]
vals = [45, 45, 0, 10.0, 0, 0]
labels = ["+45%", "+45%", "≈不退化", "+10%", "≈无坍缩", "熵1.7~2.0"]
b = ax.bar(names, vals, color=["#95a5a6", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#c0392b"])
for bb, v, lb in zip(b, vals, labels):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 1, lb, ha="center", fontsize=7)
ax.set_title("语义熵提升（相对裸）", fontsize=10)
ax.set_ylim(0, 60)

ax = axes[0, 1]
names2 = ["P1/P1.5\n无评测", "P2.5 混合\nv4.2", "P2.5 v8\n+人设", "P3\n+人设", "P4\n双通道", "P5\nDMR"]
vals2 = [0, 91, 48, 56.25, 23, 0]
b = ax.bar(names2, vals2, color=["#bdc3c7", "#e74c3c", "#e67e22", "#8e44ad", "#16a085", "#c0392b"])
for bb, v in zip(b, vals2):
    ax.text(bb.get_x() + bb.get_width() / 2, v + (1.5 if v else 3), ("—" if v == 0 else f"+{v}%"), ha="center", fontsize=7.5)
ax.set_title("LLM-Judge win_rate 最高增益", fontsize=10)
ax.set_ylim(0, 105)

ax = axes[1, 0]
names3 = ["P2.5 潮汐", "P3 锚点", "P5 未测"]
vals3b = [-9.4, 21.7, 0]
b = ax.bar(names3, vals3b, color=["#e67e22", "#8e44ad", "#bdc3c7"])
for bb, v in zip(b, vals3b):
    ax.text(bb.get_x() + bb.get_width() / 2, v + (1 if v >= 0 else -4), ("未测" if v == 0 else f"{v:+.1f}%"), ha="center", fontsize=9)
ax.axhline(0, color="#2c3e50", lw=1)
ax.set_title("HeartBench overall 提升", fontsize=10)
ax.set_ylim(-18, 30)

ax = axes[1, 1]
names4 = ["P1", "P1.5\n(同P1)", "P2.5\n(≈0)", "P3", "P4\n(≈0)", "P5\n(≈0)"]
vals4 = [933, 933, 0.001, 3.6, 0.001, 0.001]
b = ax.bar(names4, vals4, color=["#95a5a6", "#27ae60", "#e67e22", "#8e44ad", "#16a085", "#c0392b"])
for bb, v in zip(b, vals4):
    ax.text(bb.get_x() + bb.get_width() / 2, v * 1.8, ("933MB" if v == 933 else ("≈0" if v < 0.01 else f"{v}MB")), ha="center", fontsize=7.5)
ax.set_title("注入内存成本（对数视角）", fontsize=10)
ax.set_yscale("log")
ax.set_ylim(1e-3, 3e3)

fig.suptitle("语义回响家族方案关键成果总览（最终编号：P1 → P1.5 → P2.5 → P3 → P4 → P5）", fontsize=11.5, y=0.99)
save(fig, "图4_四代关键成果总览.png")

# ── 图6：P5 超融合解码器 UFD 四模式对照（种子 42）─────────────
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
ax = axes[0]
modes6 = ["裸", "锚点P3\n(种子42)", "混合\n全开", "超融合\nDMR"]
wins6 = [0.2333, 0.2333, 0.15, 0.1833]
colors6 = ["#bdc3c7", "#8e44ad", "#e67e22", "#c0392b"]
b = ax.bar(modes6, wins6, color=colors6, edgecolor="black", linewidth=0.5)
for bb, v in zip(b, wins6):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
ax.axhline(0.2333, color="#2c3e50", ls="--", lw=0.8)
ax.text(3.35, 0.24, "裸 0.2333", fontsize=8, color="#2c3e50", ha="right")
ax.set_ylim(0, 0.3)
ax.set_ylabel("win_rate")
ax.set_title("LLM-Judge win_rate（60 配对，种子 42）", fontsize=9.5)

ax = axes[1]
modes6b = ["裸", "锚点P3", "混合\n全开", "超融合\nDMR", "超融合\n全开"]
lens6 = [198.2, 66.2, 51.6, 62.0, 57.4]
b = ax.bar(modes6b, lens6, color=["#bdc3c7", "#8e44ad", "#e67e22", "#c0392b", "#c0392b"], edgecolor="black", linewidth=0.5)
for bb, v in zip(b, lens6):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 2, f"{v:.1f}", ha="center", fontsize=9)
ax.set_ylim(0, 220)
ax.set_ylabel("平均长度(字)")
ax.set_title("回复长度（健康度 5 提示，熵 1.7~2.0 无坍缩）", fontsize=9.5)

fig.suptitle("P5 超融合解码器 UFD（合成方案）：P1×P2×P3×P4 机制级融合", fontsize=12, y=0.99)
save(fig, "图6_P5超融合对照.png")

# ── 图5：P4（KER）五模式对照：win_rate + 回复长度 ─────────────
fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
ax = axes[0]
modes = ["裸", "锚点P3\n(种子200)", "超融合\nDMR", "P4-KER\n单通道", "P3+P4\n双通道"]
wins = [0.2167, 0.1167, 0.1667, 0.1167, 0.2667]
colors = ["#bdc3c7", "#8e44ad", "#95a5a6", "#16a085", "#e74c3c"]
b = ax.bar(modes, wins, color=colors, edgecolor="black", linewidth=0.5)
for bb, v in zip(b, wins):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
ax.axhline(0.2167, color="#2c3e50", ls="--", lw=0.8)
ax.text(4.35, 0.225, "裸 0.2167", fontsize=8, color="#2c3e50", ha="right")
ax.set_ylim(0, 0.32)
ax.set_ylabel("win_rate")
ax.set_title("LLM-Judge win_rate（60 配对/模式，独立种子）", fontsize=9.5)

ax = axes[1]
lengths = [122.7, 52.7, 49.8, 45.1, 43.1]
b = ax.bar(modes, lengths, color=colors, edgecolor="black", linewidth=0.5)
for bb, v in zip(b, lengths):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 2, f"{v:.1f}", ha="center", fontsize=9)
ax.set_ylim(0, 145)
ax.set_ylabel("平均长度(字)")
ax.set_title("回复长度（对齐真人短回复）", fontsize=9.5)

fig.suptitle("P4 KV 情感共振 KER：五模式对照（Qwen2.5-1.5B，30 条样本）", fontsize=12, y=0.99)
save(fig, "图5_P4_KER五模式对照.png")

print("ALL CHARTS DONE ->", OUT)
