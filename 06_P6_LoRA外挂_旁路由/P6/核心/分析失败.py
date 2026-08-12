# -*- coding: utf-8 -*-
import json, collections
LJ = json.load(open(r'f:\lora外挂\P6\评测结果\P6_LLMJudge_30.json', encoding='utf-8'))
G = json.load(open(r'f:\lora外挂\P6\评测结果\P6_生成_30.json', encoding='utf-8'))
G回 = G['回复']

for 模式 in ('裸', 'P6_LoRA裸', 'P6_旁路由'):
    细 = LJ['配对'][模式]['明细']
    by = collections.defaultdict(list)
    for d in 细:
        by[d['序号']].append(d['AI胜'])
    全输 = [s for s, w in by.items() if not any(w)]
    全胜 = [s for s, w in by.items() if all(w)]
    print(f"### {模式}: 双投全输 {len(全输)}/30, 双投全胜 {len(全胜)}/30")
    print()

print("=== P6_旁路由 全输样本 ===")
细 = LJ['配对']['P6_旁路由']['明细']
by = collections.defaultdict(list)
for d in 细:
    by[d['序号']].append(d['AI胜'])
全输 = [s for s, w in by.items() if not any(w)]
for s in 全输[:8]:
    item = [x for x in G回 if x['序号'] == s][0]
    print(f"--- #{s} user: {item['user'][:36]}")
    print(f"    真人:   {item['girl'][:52]}")
    print(f"    P6旁路: {item['回复']['P6_旁路由']['文本'][:56]}")
    print()

print("=== P6_旁路由 全胜样本（对照） ===")
全胜 = [s for s, w in by.items() if all(w)]
for s in 全胜[:5]:
    item = [x for x in G回 if x['序号'] == s][0]
    print(f"--- #{s} user: {item['user'][:36]}")
    print(f"    真人:   {item['girl'][:52]}")
    print(f"    P6旁路: {item['回复']['P6_旁路由']['文本'][:56]}")
    print()
