# -*- coding: utf-8 -*-
"""超融合解码器（UFD）对比测试 —— 裸 / 锚点P4 / 混合全开 / 超融合DMR / 超融合全开

协议：同种子 42、同提示词、同采样参数（temperature=1.0, top_p=0.9, top_k=50,
repetition_penalty=1.05, max_new_tokens=256），唯一变量是挂载的解码方案。

指标：平均熵（注入后 logits）、重复率、情感命中率、长度、兜底触发次数。
"""
import os
import sys
import json
import time
import torch
from datetime import datetime

工作目录 = os.path.dirname(os.path.abspath(__file__))
if 工作目录 not in sys.path:
    sys.path.insert(0, 工作目录)
回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)

from transformers import AutoModelForCausalLM, AutoTokenizer

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器
from 锚点解码器 import 锚点解码器, 计算熵
from 混合锚点器 import 混合锚点器
from 超融合解码器 import 超融合解码器

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
输出目录 = os.path.join(工作目录, "..", "评测结果")
os.makedirs(输出目录, exist_ok=True)

# 测试提示词（四模式对照同款：两负一正一中性一担忧）
测试提示词 = [
    "我失恋了，心里好难受，感觉整个世界都塌了。",
    "今天在公司被领导当众批评，特别委屈。",
    "我升职了！同事们都说我实至名归！",
    "你好，请问今天天气怎么样？",
    "妈妈生病住院了，我好担心她。",
]

种子 = 42
最大长度 = 256


def 记录(msg, 路径):
    print(msg, flush=True)
    with open(路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def 裸生成(model, tokenizer, 提示, 种子, 最大=最大长度):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    消息 = [{"role": "user", "content": 提示}]
    应用提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(应用提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids, temperature=1.0, top_p=0.9, top_k=50, do_sample=True,
            repetition_penalty=1.05, max_new_tokens=最大,
            pad_token_id=tokenizer.eos_token_id,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    文本 = tokenizer.decode(新token, skip_special_tokens=True).strip()
    # 平均熵：按前缀逐位前向
    熵, 步 = 0.0, 0
    ids = inputs.input_ids
    with torch.no_grad():
        for i in range(len(新token)):
            out2 = model(ids)
            熵 += 计算熵(out2.logits[0, -1, :])
            步 += 1
            ids = torch.cat([ids, 新token[i:i + 1].unsqueeze(0)], dim=-1)
    统计 = {"平均熵": round(熵 / max(步, 1), 4), "重复率": 0.0, "情感命中率": 0.0,
            "长度(字)": len(文本), "触发兜底": 0, "模式": "裸"}
    return 文本, 统计


def 方案生成(model, tokenizer, 解码器, 提示, 种子, 最大=最大长度):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    消息 = [{"role": "user", "content": 提示}]
    应用提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(应用提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out, 统计 = 解码器.生成(
            inputs.input_ids, max_new_tokens=最大,
            temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id, 用户文本=提示)
    新token = out[0, inputs.input_ids.shape[1]:]
    文本 = tokenizer.decode(新token, skip_special_tokens=True).strip()
    统计["长度(字)"] = len(文本)
    return 文本, 统计


def main():
    日志路径 = os.path.join(输出目录, "超融合对比.log")
    结果路径 = os.path.join(输出目录, "超融合对比_结果.json")
    记录(f"=== UFD 超融合对比测试 {datetime.now().strftime('%Y%m%d_%H%M%S')} ===", 日志路径)

    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    记录(f"加载模型 {模型路径} ...", 日志路径)
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16 if 设备 == "cuda" else torch.float32,
        trust_remote_code=True).to(设备)
    记录("模型加载完成", 日志路径)

    # ── 共享基础：锚点库 + 目标决策器（P2 自动适配）──
    锚库 = 锚点库(模型, 分词器)
    锚库.构建()
    记录(f"锚点库构建完成 K={锚库.锚点矩阵.shape[0]}", 日志路径)
    感知器, 潮汐决策 = None, None
    try:
        sys.path.insert(0, r"h:\情感潮汐解码（Emotion Tidal Decoding, ETD）")
        from 潮汐感知器 import 潮汐感知器
        from 潮汐决策器 import 潮汐决策器
        感知器 = 潮汐感知器()
        潮汐决策 = 潮汐决策器(感知器)
    except Exception as e:
        记录(f"P3 潮汐感知器加载失败（走简易 VAD 兜底）：{e}", 日志路径)
    目标决策 = 目标决策器(感知器=感知器, 潮汐决策器=潮汐决策, 锚点库=锚库)

    # ── 各解码方案 ──
    解码器 = {}
    解码器["锚点P4"] = 锚点解码器(模型, 分词器, 锚库, 目标决策,
                                β=0.8, T_anchor=0.3, 句子停止=True)
    解码器["混合全开"] = 混合锚点器(模型, 分词器, 锚库, 目标决策,
                                 锚点β=0.8, 锚点T=0.3, 回响λ=0.08, 潮汐倍率=12.0,
                                 开启A=True, 开启B=True, 开启C=True, 句子停止=True)
    解码器["超融合DMR"] = 超融合解码器(模型, 分词器, 锚库, 目标决策,
                                     开启DSA=True, 开启DMR=True, 开启锚点偏置=False,
                                     α基=0.15, α倍率=1.0, T_emo=0.5, 句子停止=True)
    解码器["超融合全开"] = 超融合解码器(模型, 分词器, 锚库, 目标决策,
                                      开启DSA=True, 开启DMR=True, 开启锚点偏置=True,
                                      α基=0.15, α倍率=1.0, T_emo=0.5,
                                      β=0.8, T_anchor=0.3, 句子停止=True)

    结果 = {模式: [] for 模式 in ["裸", *解码器.keys()]}
    for i, 提示 in enumerate(测试提示词):
        记录(f"\n── 提示词[{i}] {提示}", 日志路径)
        # 裸
        文本, 统计 = 裸生成(模型, 分词器, 提示, 种子)
        统计["文本"] = 文本
        结果["裸"].append(统计)
        记录(f"[裸] {文本}", 日志路径)
        # 各方案
        for 名称, 解码 in 解码器.items():
            文本, 统计 = 方案生成(模型, 分词器, 解码, 提示, 种子)
            统计["文本"] = 文本
            统计["模式"] = 名称
            结果[名称].append(统计)
            记录(f"[{名称}] {文本}", 日志路径)

    # ── 汇总表 ──
    记录("\n\n================ 汇总 ================", 日志路径)
    汇总 = {}
    for 模式, 列表 in 结果.items():
        平均熵 = round(sum(x["平均熵"] for x in 列表) / len(列表), 4)
        重复率 = round(sum(x["重复率"] for x in 列表) / len(列表), 4)
        命中率 = round(sum(x["情感命中率"] for x in 列表) / len(列表), 4)
        长度 = round(sum(x["长度(字)"] for x in 列表) / len(列表), 1)
        兜底 = sum(x.get("触发兜底次数", x.get("触发兜底", 0)) for x in 列表)
        汇总[模式] = {"平均熵": 平均熵, "重复率": 重复率, "情感命中率": 命中率,
                      "平均长度(字)": 长度, "兜底总次数": 兜底}
        记录(f"{模式:<8} 熵={平均熵:<8} 重复率={重复率:<8} "
             f"情感命中={命中率:<8} 长度={长度:<8} 兜底={兜底}", 日志路径)

    保存 = {"汇总": 汇总, "明细": 结果,
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "模型": "Qwen2.5-1.5B-Instruct", "种子": 种子}
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump(保存, f, ensure_ascii=False, indent=2)
    记录(f"\n结果已保存：{结果路径}", 日志路径)


if __name__ == "__main__":
    main()
