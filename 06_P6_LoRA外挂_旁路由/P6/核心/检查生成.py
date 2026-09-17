# -*- coding: utf-8 -*-
"""检查 P6 生成结果完整性 + 健康度预览"""
import json
import statistics

路径 = r"f:\lora外挂\P6\评测结果\P6_生成_30.json"
d = json.load(open(路径, encoding="utf-8"))
print("样本数", len(d["回复"]), "模式", d["模式"])

缺 = {}
for s in d["回复"]:
    for m in d["模式"]:
        if m not in s["回复"]:
            缺[m] = 缺.get(m, 0) + 1
print("缺失模式", 缺)

for m in d["模式"]:
    st = [s["回复"][m]["统计"] for s in d["回复"]]
    print(f"{m:<12} 熵={statistics.mean(x['平均熵'] for x in st):.3f} "
          f"重复={statistics.mean(x['重复率'] for x in st):.3f} "
          f"命中={statistics.mean(x['情感命中率'] for x in st):.3f} "
          f"长度={statistics.mean(x['长度'] for x in st):.1f} "
          f"兜底={sum(x['触发兜底次数'] for x in st)}")

print("\n--- 样例 ---")
for i in [0, 3, 8, 19, 27]:
    s = d["回复"][i]
    print(f"[{s['序号']}] 用户:{s['user'][:22]}")
    print(f"    girl:      {s['girl'][:55]}")
    print(f"    P6_旁路由: {s['回复']['P6_旁路由']['文本'][:80]}")
    print(f"    裸:        {s['回复']['裸']['文本'][:55]}")
