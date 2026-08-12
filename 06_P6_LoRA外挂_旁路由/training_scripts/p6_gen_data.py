# -*- coding: utf-8 -*-
"""P6 训练数据管线：合并 情感(4409) + 温柔(80) + 人味增强(150) → data/p6_train.jsonl"""
import json
import os
import re

本目录 = os.path.dirname(os.path.abspath(__file__))
项目根 = os.path.dirname(本目录)
数据目录 = os.path.join(项目根, "data")
输出路径 = os.path.join(数据目录, "p6_train.jsonl")

情感路径 = os.path.join(数据目录, "emotion_dataset.jsonl")
温柔路径 = os.path.join(数据目录, "gentle_dataset.jsonl")
增强路径 = os.path.join(数据目录, "p6_augment.jsonl")


def 读jsonl(路径):
    行 = []
    with open(路径, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                行.append(json.loads(line))
    return 行


def 清理温柔(文本):
    """去掉 gentle 语料里的人工噪声：'问问你，' 前缀、'？' 尾部重复、'，没关系的。' 尾巴"""
    t = re.sub(r"^问问你，", "", 文本)
    t = re.sub(r"[？?]{1,3}$", "", t)
    t = re.sub(r"，没关系的。$", "", t)
    t = re.sub(r"，难过的时候记得给自己一个拥抱。$", "", t)
    return t.strip()


def 主():
    全部 = []

    # 1) 情感数据集（指令式情绪句，训练情感表达语感）
    for d in 读jsonl(情感路径):
        instr = d.get("instruction", "").strip()
        resp = d.get("response", "").strip()
        if instr and resp:
            全部.append({"instruction": instr, "response": resp})

    # 2) 温柔数据集（对话式共情回复，清理噪声）
    for d in 读jsonl(温柔路径):
        instr = 清理温柔(d.get("instruction", "")).strip()
        resp = d.get("response", "").strip()
        if instr and resp:
            全部.append({"instruction": instr, "response": resp})

    # 3) 人味对话增强（girl 风格短回复，评测域定向）
    #    加权：对话式口语样本重复 3 倍，压制 4409 条指令式书面风格的主导
    增强 = 读jsonl(增强路径)
    全部.extend(增强 * 3)

    # 去重（instruction+response 完全一致）
    唯一 = []
    见过 = set()
    for d in 全部:
        键 = (d["instruction"], d["response"])
        if 键 not in 见过:
            见过.add(键)
            唯一.append(d)

    with open(输出路径, "w", encoding="utf-8") as f:
        for d in 唯一:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter
    长度 = [len(d["response"]) for d in 唯一]
    平均长 = sum(长度) / len(长度) if 长度 else 0
    print(f"总样本: {len(唯一)}")
    print(f"  情感: {len(读jsonl(情感路径))}  温柔: {len(读jsonl(温柔路径))}  增强: {len(增强)}")
    print(f"响应平均长度: {平均长:.1f} 字")
    print(f"已保存: {输出路径}")


if __name__ == "__main__":
    主()
