# -*- coding: utf-8 -*-
"""
引导强度扫描：定位"能有效改变输出但不破坏自然度"的引导幅度区间
================================================================
诊断问题：v1/v2 引导幅度 α/T×|V| ≈ 0.05，相对 logits（几十量级）太弱，
导致输出几乎不变。本脚本在同一 prompt 下扫描引导倍率 k：
  logits[情感token] += k
观察情感 token 概率提升、输出变化、熵/重复率，找到有效区间。
"""
import os
import sys
import torch
import torch.nn.functional as F

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from transformers import AutoModelForCausalLM, AutoTokenizer
from 潮汐感知器 import 潮汐感知器
from 潮汐决策器 import 潮汐决策器
from 潮汐解码器 import 潮汐解码器

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"

测试提示 = [
    ("负面-强烈", "我真的崩溃了，彻底受不了了！"),
    ("负面-温和", "我今天有点烦，心情不太好。"),
    ("正面-强烈", "我升职了！！太开心了！！"),
    ("中性", "你好，请问今天天气怎么样？"),
]

扫描倍率 = [0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0]


def 情感词概率(tokenizer, logits, 情感表):
    """情感 token 概率 vs 全体 token 概率"""
    probs = F.softmax(logits, dim=-1)
    情感p = sum(probs[0, tid].item() for tid in 情感表 if tid < probs.shape[-1])
    return 情感p


def 主程序():
    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[加载] → {设备} ...")
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16 if 设备 == "cuda" else torch.float32,
        trust_remote_code=True).to(设备)
    模型.eval()
    print("[加载] 完成\n")

    感知器 = 潮汐感知器()
    决策器 = 潮汐决策器(感知器)
    解码器 = 潮汐解码器(模型, 分词器, 感知器, 决策器)

    for 名称, 提示 in 测试提示:
        状态, 关键词 = 感知器.测量(提示)
        print(f"═══ [{名称}] {提示!r} → V={状态.valence:.3f} A={状态.arousal:.3f} ═══")

        # 计算决策目标
        感知器2 = 潮汐感知器()
        决策器2 = 潮汐决策器(感知器2)
        感知器2.追加轨迹("用户", 状态)
        目标 = 决策器2.计算目标(状态)
        print(f"   目标 V={目标.目标状态.valence:.3f} α={目标.引导强度:.3f}")

        消息 = [{"role": "user", "content": 提示}]
        应用提示 = 分词器.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
        inputs = 分词器(应用提示, return_tensors="pt").to(设备)

        with torch.no_grad():
            out = 模型(inputs.input_ids)
            base_logits = out.logits[:, -1, :].clone()

        print(f"   {'k':>5} {'情感token概率':>12} {'提升':>8} {'情感p排位':>8}")
        基线 = 情感词概率(分词器, base_logits, 解码器._情感token表)
        全体p = F.softmax(base_logits, dim=-1)[0].sort(descending=True).values
        # 情感 token 在 top-k 中的平均排位
        for k in 扫描倍率:
            引导logits = base_logits.clone()
            if k > 0 and 目标 is not None and abs(目标.目标状态.valence) >= 0.03:
                方向 = 1.0 if 目标.目标状态.valence > 0 else -1.0
                for tid, 极性 in 解码器._情感token表.items():
                    if tid >= 引导logits.shape[-1]:
                        continue
                    方向系数 = 极性 * 方向
                    if 方向系数 > 0:
                        引导logits[0, tid] += k
                    else:
                        引导logits[0, tid] -= k * 0.3
            情感p = 情感词概率(分词器, 引导logits, 解码器._情感token表)
            # 情感词在生成分布中的实际采样占比（估算：情感p/1）
            print(f"   {k:>5.1f} {情感p:>12.4f} {情感p-基线:>+8.4f}")
        print()

    模型 = 模型.to("cpu")
    del 模型, 分词器
    torch.cuda.empty_cache()


if __name__ == "__main__":
    主程序()
