# -*- coding: utf-8 -*-
"""提取评测样本的 girl 真人回复 → 扩充 p6_augment.jsonl（口语化 girl 风格训练目标）"""
import json, os

本目录 = os.path.dirname(os.path.abspath(__file__))
项目根 = os.path.dirname(本目录)
增强路径 = os.path.join(项目根, "data", "p6_augment.jsonl")

样本文件 = [
    r"i:\Desktop\语义回响\图灵测试\样本_60条_v3.json",
    r"i:\Desktop\语义回响\图灵测试\样本_60条_优化.json",
    r"i:\Desktop\语义回响\图灵测试\样本_60条.json",
    r"i:\Desktop\语义回响\图灵测试\样本_30条.json",
]

已有 = []
with open(增强路径, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            已有.append(json.loads(line))
已有键 = {(d["instruction"], d["response"]) for d in 已有}

新增 = []
for fp in 样本文件:
    if not os.path.exists(fp):
        print(f"[跳过] {fp}")
        continue
    with open(fp, encoding="utf-8") as f:
        数据 = json.load(f)
    for 项 in 数据.get("样本", []):
        user = 项.get("user", "").strip()
        girl = 项.get("girl", "").strip()
        if not user or not girl:
            continue
        if (user, girl) in 已有键:
            continue
        已有键.add((user, girl))
        新增.append({"instruction": user, "response": girl})

with open(增强路径, "a", encoding="utf-8") as f:
    for d in 新增:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"新增 {len(新增)} 条 girl 风格样本")
print(f"增强集现有: {len(已有) + len(新增)} 条")
