# -*- coding: utf-8 -*-
"""汇总 P5 30 条生成的健康度指标"""
import json
from collections import defaultdict

路径 = r"i:\Desktop\语义回响\发布\SemanticEcho-AnchorEcho\评测结果\P5_生成_30.json"
with open(路径, encoding="utf-8") as f:
    d = json.load(f)

汇总 = defaultdict(lambda: {"熵": 0.0, "重复": 0.0, "命中": 0.0, "长度": 0.0, "兜底": 0, "n": 0})
for 项 in d["回复"]:
    for 模式 in d["模式"]:
        s = 项["回复"][模式]["统计"]
        t = 项["回复"][模式]["文本"]
        v = 汇总[模式]
        v["熵"] += s.get("平均熵", 0)
        v["重复"] += s.get("重复率", 0)
        v["命中"] += s.get("情感命中率", 0)
        v["长度"] += len(t)
        v["兜底"] += s.get("触发兜底次数", 0)
        v["n"] += 1

print(f"{'模式':<10}{'平均熵':>8}{'重复率':>8}{'情感命中':>9}{'平均长度':>9}{'兜底':>5}")
for 模式 in d["模式"]:
    v = 汇总[模式]
    n = v["n"]
    print(f"{模式:<10}{v['熵']/n:>8.3f}{v['重复']/n:>8.3f}{v['命中']/n:>9.3f}{v['长度']/n:>9.1f}{v['兜底']:>5}")
