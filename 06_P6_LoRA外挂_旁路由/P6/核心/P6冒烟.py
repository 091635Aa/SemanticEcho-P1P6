# -*- coding: utf-8 -*-
"""P6 冒烟测试：挂载 LoRA 后对 6 条代表性消息生成，检查情感质量"""
import os
import sys

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from P6旁路由 import P6旁路由生成器

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
lora路径 = r"f:\lora外挂\lora_adapters\p6_emotion"

测试消息 = [
    "妈妈生病住院了，我好担心她。",
    "有时候我会想，努力到底有什么意义。",
    "这饮料味道不错。",
    "我好像越来越不会表达情绪了，怕说出来别人会烦。",
    "今天被同事抢了功劳，好气。",
    "你为什么总能接我梗？",
]

生成器 = P6旁路由生成器(模型路径, lora路径, 挂载=True)
for i, 消息 in enumerate(测试消息):
    print(f"\n{'='*60}\n[测试 {i+1}] 用户：{消息}")
    文本, tokens, 统计, 候选 = 生成器.生成(
        [{"role": "user", "content": 消息}], 种子=2026 + i, N=3, 返回候选=True)
    print("  ── 3 条候选：")
    for c in 候选:
        print(f"    [分{c['分数']:.3f}] {c['文本'][:60]}")
    print(f"  ── 路由选优：{文本}")
    print(f"  统计：熵={统计['平均熵']} 重复={统计['重复率']} 命中={统计['情感命中率']} 长度={统计['长度']} 兜底={统计['触发兜底次数']}")
生成器.清理()
print("\n冒烟测试完成")
