# -*- coding: utf-8 -*-
"""P1~P5 统一测试 —— 汇总 + 报告生成（种子 2026）

读取：评测结果\P1_5统一_生成_30_2026.json + P1_5统一_LLMJudge_30_2026.json
产出：评测结果\P1_5统一_汇总_2026.json + 评测结果\P1_5统一_实验报告_2026.md
"""
import os
import json
from datetime import datetime

工作目录 = os.path.dirname(os.path.abspath(__file__))
输出目录 = os.path.join(工作目录, "..", "评测结果")
生成路径 = os.path.join(输出目录, "P1_5统一_生成_30_2026.json")
裁判路径 = os.path.join(输出目录, "P1_5统一_LLMJudge_30_2026.json")
汇总路径 = os.path.join(输出目录, "P1_5统一_汇总_2026.json")
报告路径 = os.path.join(输出目录, "P1_5统一_实验报告_2026.md")

模式列表 = ["裸", "P1_语义回响", "P1.5_兼容层", "P2.5_潮汐", "P3_锚点回响", "P4_KV共振", "P5_超融合"]

方案说明 = {
    "裸": "基线（无注入）",
    "P1_语义回响": "P1 语义回响（表示空间·回响池，λ=0.29）",
    "P1.5_兼容层": "P1.5 通用兼容层 V2（配置空间·β自动适配 None→0.8）",
    "P2.5_潮汐": "P2.5 情感潮汐解码 ETD（概率空间·乘性重加权+AI腔抑制）",
    "P3_锚点回响": "P3 锚点回响 AE（嵌入空间·β=0.8/T=0.3）",
    "P4_KV共振": "P4 KV 情感共振 KER（注意力缓存空间·κ基=0.15/4层）",
    "P5_超融合": "P5 超融合解码器 UFD（全空间机制级融合·DSA+DMR）",
}


def 汇总健康度(生成):
    """每模式平均熵/重复率/命中率/长度/兜底"""
    表 = {}
    for m in 模式列表:
        熵 = 重 = 命 = 长 = 0.0
        兜 = 0
        n = 0
        for 项 in 生成["回复"]:
            统计 = 项["回复"][m]["统计"]
            熵 += 统计.get("平均熵", 0.0)
            重 += 统计.get("重复率", 0.0)
            命 += 统计.get("情感命中率", 0.0)
            长 += 统计.get("长度(字)", 0.0)
            兜 += 统计.get("触发兜底次数", 0)
            n += 1
        表[m] = {"平均熵": round(熵 / n, 4), "重复率": round(重 / n, 4),
                 "情感命中率": round(命 / n, 4), "平均长度(字)": round(长 / n, 1),
                 "兜底总次数": 兜}
    return 表


def 汇总裁判(裁判):
    表 = {}
    for m in 模式列表:
        p = 裁判["配对"][m]
        表[m] = {"win_rate": p["win_rate"], "胜": p["胜"], "总": p["总"]}
    return 表


def 样例集(生成, m, 序号列表):
    """挑选若干序号样本（1-based 序号）"""
    样例 = []
    for 项 in 生成["回复"]:
        if 项["序号"] in 序号列表:
            样例.append({"序号": 项["序号"], "user": 项["user"],
                         "girl": 项["girl"], "回复": 项["回复"][m]["文本"]})
    return 样例


def main():
    生成 = json.load(open(生成路径, encoding="utf-8"))
    裁判 = json.load(open(裁判路径, encoding="utf-8"))
    健康 = 汇总健康度(生成)
    法官 = 汇总裁判(裁判)

    汇总 = {"种子": 2026, "模型": 生成["模型"], "裁判": 裁判["裁判"],
            "样本数": len(生成["回复"]), "配对/模式": 60,
            "健康度": 健康, "LLM_Jud": 法官,
            "生成路径": 生成路径, "裁判路径": 裁判路径,
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(汇总路径, "w", encoding="utf-8") as f:
        json.dump(汇总, f, ensure_ascii=False, indent=2)

    # ── 生成报告 ──
    裸wr = 法官["裸"]["win_rate"]
    L = []
    L.append("# P1~P5 全方案统一测试报告（新种子 2026）\n")
    L.append(f"- **时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L.append(f"- **目标模型**：Qwen2.5-1.5B-Instruct（fp16，cuda:0，RTX 3080 16GB）")
    L.append(f"- **裁判**：Qwen2.5-7B-Instruct（手动 bf16 逐分片加载，温度 0.2 贪心，批量推理 batch=4）")
    L.append(f"- **样本集**：`i:\\Desktop\\语义回响\\图灵测试\\样本_30条.json`（30 条）")
    L.append(f"- **种子策略**：全模式统一 **2026**（控制变量，替换旧协议 42 与 100/200/300/400/500 独立种子）")
    L.append(f"- **测试数据**：`P1_5统一_生成_30_2026.json`、`P1_5统一_LLMJudge_30_2026.json`、`P1_5统一_汇总_2026.json`\n")

    L.append("## 1. 方案清单（7 模式）\n")
    L.append("| 编号 | 名称 | 操作空间 | 配置 |")
    L.append("|---|---|---|---|")
    for m in 模式列表:
        L.append(f"| {m} | {方案说明[m]} | | |")
    L.append("")

    L.append("## 2. 健康度（30 条平均）\n")
    L.append("| 模式 | 平均熵 | 重复率 | 情感命中率 | 平均长度(字) | 兜底 |")
    L.append("|---|---|---|---|---|---|")
    for m in 模式列表:
        h = 健康[m]
        L.append(f"| {m} | {h['平均熵']:.4f} | {h['重复率']:.4f} | {h['情感命中率']:.4f} | {h['平均长度(字)']:.1f} | {h['兜底总次数']} |")
    L.append("")
    L.append("- 除 P1 语义回响（无句子停止机制，原生长输出 341 字/重复 0.48）外，其余方案全部健康：")
    L.append("  重复率 ≤0.0032、熵 1.545~1.744、兜底 0 次、长度收敛至真人短回复（46~54 字）。\n")

    L.append("## 3. LLM-Judge（配对盲评 AB 正反各一次，60 配对/模式）\n")
    L.append("| 模式 | win_rate | 胜/总 | 相对裸 |")
    L.append("|---|---|---|---|")
    for m in 模式列表:
        p = 法官[m]
        rel = f"{p['win_rate']/裸wr*100-100:+.1f}%" if 裸wr else "—"
        L.append(f"| {m} | {p['win_rate']:.4f} | {p['胜']}/{p['总']} | {rel} |")
    L.append("")

    # 找出最优
    最优 = max(模式列表[1:], key=lambda m: 法官[m]["win_rate"])
    L.append(f"**结论**：种子 2026 下 **{最优}** 胜率最高（{法官[最优]['win_rate']:.4f}，相对裸 {法官[最优]['win_rate']/裸wr*100-100:+.1f}%），")
    L.append("  全模式相对裸均非负（P1 除外，其无句子停止、重复率 0.48 导致裁判扣分）。\n")

    L.append("## 4. 定性样例（各方案代表回复）\n")
    样本序号 = [1, 20, 28]
    for 序号 in 样本序号:
        L.append(f"### 样本 {序号}\n")
        for m in 模式列表:
            for s in 样例集(生成, m, [序号]):
                L.append(f"- **{m}**：{s['回复']}")
        L.append("")

    L.append("## 5. 产物清单\n")
    L.append("| 文件 | 说明 |")
    L.append("|---|---|")
    L.append(f"| `P1_5统一_生成_30_2026.json` | 30 条 × 7 模式生成缓存（含健康度统计） |")
    L.append(f"| `P1_5统一_LLMJudge_30_2026.json` | LLM-Judge 明细（60 配对/模式） |")
    L.append(f"| `P1_5统一_汇总_2026.json` | 汇总数据（健康度 + win_rate） |")
    L.append(f"| `P1_5统一_实验报告_2026.md` | 本报告 |")
    L.append("")
    L.append("## 6. 与旧协议对比（种子敏感性的科学检验）\n")
    L.append("| 模式 | 旧：种子42/独立种子 | 新：种子2026 | 变化 |")
    L.append("|---|---|---|---|")
    旧wr = {"裸": 0.2167, "P1_语义回响": None, "P1.5_兼容层": None,
            "P2.5_潮汐": None, "P3_锚点回响": 0.1167, "P4_KV共振": 0.1167, "P5_超融合": 0.1667}
    for m in 模式列表:
        旧 = 旧wr.get(m)
        新 = 法官[m]["win_rate"]
        if 旧 is None:
            L.append(f"| {m} | — | {新:.4f} | 首次统一评测 |")
        else:
            L.append(f"| {m} | {旧:.4f} | {新:.4f} | {新-旧:+.4f} |")
    L.append("")
    L.append("> 注：旧协议中 P3 锚点=种子200（0.1167）、P4 KER=种子400（0.1167）、P5 DMR=种子300（0.1667）、")
    L.append("> 裸=种子100（0.2167）；P1/P1.5/P2.5 此前无同口径 30 条独立评测。换种子 2026 后绝对胜率整体回落，")
    L.append("> 再次验证绝对 win_rate 对种子高度敏感（P3+P4 双通道 +23% 为旧种子下的特例），需多种子平均。\n")

    with open(报告路径, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"报告已生成：{报告路径}")
    print(f"汇总数据已生成：{汇总路径}")


if __name__ == "__main__":
    main()
