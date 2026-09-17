# -*- coding: utf-8 -*-
"""P4 锚点回响（Anchor Echo）· Task1 灵感定稿与可行性验证脚本

三项验证：
  a) 方向分离度：种子情感词 vs 中性词 与对应锚点的余弦相似度分布对比
     （均值/标准差/均值差/t 统计量），并在「原始 embedding / 层归一化
     embedding」两种模式下对比分离度，取分离度高的作为主模式；
     同时输出「所有情感词 vs 所有中性词 相对全部 6 锚点 max 余弦」的均值对比。
  b) 只读证明：构建锚点 + 打分前后，原始 embedding 权重 W_e 的 sum 与
     data_ptr 完全一致（零修改）。
  c) 近义覆盖对比（P3 稀疏 vs P4 稠密）：词库外近义情感词的 P4 稠密打分
     显著高于中性词；top-30 高频普通 token 不误报高情感。

用法：python 验证_灵感.py
"""
import os
import sys
import json
import math
from collections import Counter

os.environ["HF_HUB_OFFLINE"] = "1"
工作目录 = r"h:\锚点回响（Anchor Echo）"
sys.path.insert(0, 工作目录)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from 锚点库 import 锚点库, 默认词集

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
中性词表 = ["桌子", "电脑", "椅子", "马路", "天气", "数字", "蓝色",
             "苹果", "时间", "材料", "规则", "声音", "纸张", "玻璃", "金属"]
近义情感词表 = ["扎心", "破防", "心累", "崩溃", "泪目", "上头", "内耗", "治愈", "宠溺", "孤寡"]
# 停用字符（高频单字虚词/标点），用于过滤 top-30 高频 token 抽查
停用字符 = set("的了是在有和人这那我都你就他她它与及把被着过也还很到说就对吧呢吗啊吧哦呃嗯"
                "不一个上来下去中大小多少什么怎么东西你们咱们自己时候地方工作生活"
                "，。！？、；：“”‘’（）【】…—·《》年月日时分秒")


def 统计(值列表):
    """返回 (均值, 标准差[ddof=1], 数量)。"""
    n = len(值列表)
    m = sum(值列表) / n
    s = (sum((x - m) ** 2 for x in 值列表) / max(n - 1, 1)) ** 0.5
    return m, s, n


def 方向分离度(库):
    """验证 a：逐维度种子词 vs 中性词相对对应锚点的余弦分布；全词 max 余弦对比。"""
    结果 = {"每维": []}
    全情感词 = [w for 维 in 库.维度名() for w in 库.词集[维]]
    情感得分 = {w: 库.词得分(w) for w in 全情感词}
    中性得分 = {w: 库.词得分(w) for w in 中性词表}
    for i, 维 in enumerate(库.维度名()):
        种子 = [float(情感得分[w][i]) for w in 库.词集[维]]
        中性 = [float(中性得分[w][i]) for w in 中性词表]
        m1, s1, n1 = 统计(种子)
        m2, s2, n2 = 统计(中性)
        t = (m1 - m2) / (math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2) + 1e-12)
        结果["每维"].append({"维度": 维, "情感均值": round(m1, 4), "情感标准差": round(s1, 4),
                             "中性均值": round(m2, 4), "中性标准差": round(s2, 4),
                             "均值差": round(m1 - m2, 4), "t": round(t, 2)})
    情感max = [float(max(情感得分[w])) for w in 全情感词]
    中性max = [float(max(中性得分[w])) for w in 中性词表]
    m3, s3, n3 = 统计(情感max)
    m4, s4, n4 = 统计(中性max)
    结果["全情感max均值"] = round(m3, 4)
    结果["全情感max标准差"] = round(s3, 4)
    结果["全中性max均值"] = round(m4, 4)
    结果["全中性max标准差"] = round(s4, 4)
    结果["max均值差"] = round(m3 - m4, 4)
    结果["平均t"] = round(sum(d["t"] for d in 结果["每维"]) / len(结果["每维"]), 2)
    结果["平均均值差"] = round(sum(d["均值差"] for d in 结果["每维"]) / len(结果["每维"]), 4)
    return 结果


def 只读证明(库):
    """验证 b：构建锚点 + 打分前后，原始 embedding 权重 sum 与 data_ptr 一致。"""
    基线 = 库.记录只读基线()
    _ = 库.得分([0, 1, 2, 3, 12345])          # 跑一次逐 token 打分
    _ = 库.词得分("测试")
    return 库.验证只读(基线)


def 近义覆盖(库, tokenizer):
    """验证 c：P3 稀疏 vs P4 稠密对比 + top-30 高频普通 token 误报检查。"""
    结果 = {}
    种子词集 = {w for 维 in 库.维度名() for w in 库.词集[维]}
    # P3 稀疏打分：词出现在（种子）情感词库为 1，否则 0（简化判定）
    近义P3 = [1.0 if w in 种子词集 else 0.0 for w in 近义情感词表]
    近义P4 = [float(max(库.词得分(w))) for w in 近义情感词表]
    中性max = [float(max(库.词得分(w))) for w in 中性词表]
    结果["近义P3"] = [int(x) for x in 近义P3]
    结果["近义P4"] = [round(x, 4) for x in 近义P4]
    结果["近义词平均P4"] = round(sum(近义P4) / len(近义P4), 4)
    结果["中性词平均max"] = round(sum(中性max) / len(中性max), 4)
    m4, s4, _ = 统计(中性max)
    结果["中性词max标准差"] = round(s4, 4)
    结果["识别判定"] = bool(结果["近义词平均P4"] > m4 + 2 * s4)
    # top-30 高频普通 token 抽查
    文本 = (
        "语义回响项目研究语言模型的情感表达能力，系统从输入文本中提取隐藏状态，"
        "通过向量计算得到语义特征，然后生成新的文本内容。"
        "模型训练需要大量数据，训练过程使用显卡加速，工程师调试代码时发现一个错误，"
        "修复后程序正常运行。办公桌上放着电脑键盘和文件，墙上挂着日历和时钟。"
        "公司召开会议讨论新产品方案，市场部提交了详细报告，产品经理说明技术需求。"
        "学校图书馆藏书丰富，学生可以借阅各种专业书籍，老师讲解数学公式。"
        "城市交通系统包含地铁公交和出租车，出行非常方便，政府规划新的道路。"
        "医生建议保持健康的生活方式，早睡早起，适当运动，注意饮食均衡。"
        "国家经济发展迅速，人民生活水平不断提高，科技创新推动产业升级。"
        "今天天气不错，我们去公园散步，路上看到很多行人，孩子们在草地上玩耍。"
        "研究员阅读文献，记录实验数据，分析结果，撰写论文，提交给学术会议。"
    )
    ids = tokenizer.encode(文本, add_special_tokens=False)
    计数 = Counter(ids)
    高频 = []
    for tid, cnt in 计数.most_common():
        字面 = tokenizer.decode([tid]).strip()
        if not 字面 or (len(字面) == 1 and 字面 in 停用字符):
            continue
        高频.append((tid, 字面, cnt))
        if len(高频) >= 30:
            break
    高频max = [float(max(库.得分([tid])[0])) for tid, _, _ in 高频]
    结果["高频token"] = [{"token": 字面, "频次": int(cnt),
                           "max余弦": round(float(m), 4)} for (_, 字面, cnt), m in zip(高频, 高频max)]
    结果["高频平均max"] = round(sum(高频max) / len(高频max), 4)
    结果["误报判定"] = bool(结果["高频平均max"] < 结果["近义词平均P4"])
    return 结果


def main():
    print("=" * 70)
    print("P4 锚点回响（Anchor Echo）· Task1 灵感定稿与可行性验证")
    print("=" * 70)
    print("[1/3] 加载模型 ...")
    tokenizer = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()
    print(f"  模型加载完成 dtype={model.dtype}  "
          f"embedding 权重 shape={tuple(model.get_input_embeddings().weight.shape)}")

    汇总 = {"模型": "Qwen2.5-1.5B-Instruct", "dtype": str(model.dtype),
            "模式对比": {}, "最终": {}}
    库们 = {}
    for 模式 in ("原始", "层归一化"):
        print(f"\n[2/3] 构建锚点库（模式：{模式}）...")
        库 = 锚点库(model, tokenizer, 模式=模式)
        E = 库.构建()
        print(f"  锚点矩阵 shape={tuple(E.shape)} 维度={库.维度名()}")
        库们[模式] = 库

    # 验证 a：两种模式的方向分离度
    for 模式, 库 in 库们.items():
        print(f"\n{'=' * 70}")
        print(f"验证 a · 方向分离度 [{模式}]")
        print("=" * 70)
        r = 方向分离度(库)
        汇总["模式对比"][模式] = r
        for d in r["每维"]:
            print(f"  {d['维度']:<4} 情感 {d['情感均值']:.4f}±{d['情感标准差']:.4f}  |  "
                  f"中性 {d['中性均值']:.4f}±{d['中性标准差']:.4f}  |  "
                  f"均值差 {d['均值差']:+.4f}  |  t={d['t']:.2f}")
        print(f"  全情感词 max 余弦：{r['全情感max均值']:.4f}±{r['全情感max标准差']:.4f}  vs  "
              f"全中性词：{r['全中性max均值']:.4f}±{r['全中性max标准差']:.4f}  （差 {r['max均值差']:+.4f}）")
        print(f"  平均 t 统计量 = {r['平均t']:.2f}   平均均值差 = {r['平均均值差']:+.4f}")

    # 选择主模式：平均 t 更大者（方向分离度更高）
    主模式 = max(汇总["模式对比"], key=lambda m: 汇总["模式对比"][m]["平均t"])
    print(f"\n>>> 主模式选定：{主模式}（平均 t 更高，方向分离度更大）")
    汇总["主模式"] = 主模式
    主库 = 库们[主模式]

    # 验证 b：只读证明（与模式无关，用主库）
    print(f"\n{'=' * 70}")
    print(f"验证 b · 只读证明 [{主模式}]")
    print("=" * 70)
    rb = 只读证明(主库)
    汇总["只读证明"] = rb
    print(f"  构建+打分前  W_e.sum() = {rb['sum_before']:.6f}   data_ptr = {rb['data_ptr_before']}")
    print(f"  构建+打分后  W_e.sum() = {rb['sum_after']:.6f}   data_ptr = {rb['data_ptr_after']}")
    print(f"  sum 一致 = {rb['sum一致']}   指针一致 = {rb['指针一致']}")

    # 验证 c：近义覆盖对比（用主模式）
    print(f"\n{'=' * 70}")
    print(f"验证 c · 近义覆盖对比（P3 稀疏 vs P4 稠密）[{主模式}]")
    print("=" * 70)
    rc = 近义覆盖(主库, tokenizer)
    汇总["近义覆盖"] = rc
    print("  词库外近义情感词（P3 稀疏打分均应为 0，P4 稠密打分 = 与 6 锚点最大余弦）：")
    print("  词         P3稀疏   P4稠密")
    for i, 词 in enumerate(近义情感词表):
        print(f"  {词:<8} {rc['近义P3'][i]:<6}   {rc['近义P4'][i]:.4f}")
    print(f"  近义词平均 P4 稠密打分 = {rc['近义词平均P4']:.4f}")
    print(f"  中性词平均 max 余弦    = {rc['中性词平均max']:.4f}±{rc['中性词max标准差']:.4f}")
    print(f"  提升 = {rc['近义词平均P4'] - rc['中性词平均max']:+.4f}  "
          f"识别判定 = {'成立' if rc['识别判定'] else '不成立'}")
    print(f"\n  top-30 高频普通 token 误报检查：")
    for 项 in rc["高频token"]:
        print(f"    {项['token']:<8} 频次 {项['频次']:<4} max 余弦 {项['max余弦']:.4f}")
    print(f"  高频普通 token 平均 max 余弦 = {rc['高频平均max']:.4f}  <  近义词平均 {rc['近义词平均P4']:.4f}  "
          f"误报判定 = {'无显著误报' if rc['误报判定'] else '存在误报风险'}")

    # 保存结果 JSON
    保存路径 = os.path.join(工作目录, "对照结果", "灵感验证结果.json")
    os.makedirs(os.path.dirname(保存路径), exist_ok=True)
    with open(保存路径, "w", encoding="utf-8") as f:
        json.dump(汇总, f, ensure_ascii=False, indent=2)
    print(f"\n>>> 全部验证完成，结果已保存：{保存路径}")
    print(f">>> 主模式：{主模式}  |  只读成立：{rb['sum一致'] and rb['指针一致']}"
          f"  |  近义识别：{'成立' if rc['识别判定'] else '不成立'}"
          f"  |  误报：{'无' if rc['误报判定'] else '有'}")


if __name__ == "__main__":
    main()
