# -*- coding: utf-8 -*-
"""P1~P5 大厂式情感理解测试 —— 阶段二：裁判（7B bf16 手动分片加载）

读取 评测结果\P1_5统一_情感理解_生成_2026.json：
① 情绪识别：期望 vs 模型标签 → 准确率
② 情绪推理：裁判按 推理质量 0-1 评分
③ 四分支 EI：裁判按 感知/理解/促进/管理 1-5 分
输出：评测结果\P1_5统一_情感理解_2026.json
"""
import os
import sys
import json
import re
import gc
import glob
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors import safe_open

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
裁判模型名 = "Qwen2.5-7B-Instruct"
评测结果 = os.path.join(本目录, "..", "评测结果")
生成路径 = os.path.join(评测结果, "P1_5统一_情感理解_生成_2026.json")
输出路径 = os.path.join(评测结果, "P1_5统一_情感理解_2026.json")

情绪类别 = ["快乐", "悲伤", "愤怒", "恐惧", "惊讶", "中性", "疲惫"]

推理评分提示 = """你是情感推理质量评估专家。
情境：「{情境}」
模型给出的情绪推理：
{推理}
请评估该推理的质量（触发因素是否具体、合理、贴合情境）：
必须严格按 JSON 输出：{{"推理质量": 0到1的小数, "reason": "一句话理由"}}"""

四分支提示 = """你是一位情感智力（Emotional Intelligence）评估专家，依据 Mayer-Salovey-Caruso 四分支模型评估以下回复：
用户说：「{user}」
AI 回复：「{reply}」

请对回复在以下四个维度各打 1-5 分（1=差，5=优秀）：
1. 情绪感知（Perceiving）：是否准确识别用户的情绪状态
2. 情绪理解（Understanding）：是否理解情绪产生的原因与关系
3. 情绪促进思维（Facilitating Thought）：是否帮助用户以更好的方式思考
4. 情绪管理（Managing）：是否有效调节和改善用户情绪
必须严格按 JSON 输出：
{{"情绪感知": 1到5的整数, "情绪理解": 1到5的整数, "情绪促进思维": 1到5的整数, "情绪管理": 1到5的整数}}"""


def 加载裁判():
    gc.collect()
    torch.cuda.empty_cache()
    裁判路径 = os.path.join(模型空间, 裁判模型名)
    cfg = AutoConfig.from_pretrained(裁判路径, trust_remote_code=True)
    with torch.device("meta"):
        模型 = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    模型 = 模型.to_empty(device="cuda")
    for _分片 in sorted(glob.glob(os.path.join(裁判路径, "model-*.safetensors"))):
        with safe_open(_分片, framework="pt", device="cpu") as f:
            _sd = {k: f.get_tensor(k) for k in f.keys()}
        模型.load_state_dict(_sd, strict=False)
        del _sd
        gc.collect()
    _base = getattr(cfg, "rope_theta", 1000000.0)
    _头维 = cfg.hidden_size // cfg.num_attention_heads
    _inv = 1.0 / (_base ** (torch.arange(0, _头维, 2, dtype=torch.int64).float() / _头维))
    _inv = _inv.to(torch.float32)
    for _模块 in 模型.modules():
        if hasattr(_模块, "inv_freq") and _模块.inv_freq is not None:
            _模块.inv_freq.copy_(_inv)
            if hasattr(_模块, "original_inv_freq") and _模块.original_inv_freq is not None:
                _模块.original_inv_freq.copy_(_inv)
    torch.cuda.empty_cache()
    模型.eval()
    return 模型, AutoTokenizer.from_pretrained(裁判路径, trust_remote_code=True)


def 裁判生成(裁判模型, 裁判分词器, 消息, max_new_tokens=150):
    提示 = 裁判分词器.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = 裁判分词器(提示, return_tensors="pt").to(裁判模型.device)
    with torch.no_grad():
        out = 裁判模型.generate(
            inputs.input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=裁判分词器.eos_token_id)
    新 = out[0, inputs.input_ids.shape[1]:]
    return 裁判分词器.decode(新, skip_special_tokens=True).strip()


def 解析情绪(文本):
    m = re.search(r'"情绪"\s*[:：]\s*"([^"]+)"', 文本)
    if m:
        return m.group(1)
    for c in 情绪类别:
        if c in 文本:
            return c
    return None


def 解析小数(文本, 键):
    m = re.search(rf'"{键}"\s*[:：]\s*([0-9]*\.?[0-9]+)', 文本)
    return float(m.group(1)) if m else None


def 解析整数(文本, 键):
    m = re.search(rf'"{键}"\s*[:：]\s*([1-5])', 文本)
    return int(m.group(1)) if m else None


def 主():
    print(f"=== P1~P5 情感理解 阶段二裁判（7B bf16）{datetime.now().strftime('%H:%M:%S')} ===", flush=True)
    with open(生成路径, encoding="utf-8") as f:
        数据 = json.load(f)
    模式们 = 数据["模式"]
    裁判模型, 裁判分词器 = 加载裁判()
    print("裁判模型加载完成", flush=True)

    结果 = {"模型": 数据["模型"], "裁判": 裁判模型名, "种子": 2026,
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "方法论": "①情绪识别(MME-Emotion式) ②情绪推理(CAREBench式) ③四分支EI(AttuneBench式)",
            "结果": {}}
    for 模式 in 模式们:
        print(f"── [{模式}] ──", flush=True)
        识别正确 = 识别总 = 0
        推理分 = []
        四分支统计 = {"情绪感知": [], "情绪理解": [], "情绪促进思维": [], "情绪管理": []}
        识别明细 = 推理明细 = 四分支明细 = []
        识别明细, 推理明细, 四分支明细 = [], [], []
        for i, s in enumerate(数据["样本"][模式]):
            if s["任务"] == "识别":
                txt = s["回复"]
                if not txt.strip():
                    txt = "（空回复）"
                判定 = 解析情绪(txt)
                正确 = (判定 == s["期望"])
                识别正确 += 1 if 正确 else 0
                识别总 += 1
                识别明细.append({"情境": s["情境"][:20], "期望": s["期望"], "输出": 判定, "正确": 正确})
                print(f"  [识别 {i+1}] 期望={s['期望']} 输出={判定} {'✓' if 正确 else '✗'}", flush=True)
            elif s["任务"] == "推理":
                txt = s["回复"]
                if not txt.strip():
                    txt = "（空回复）"
                裁判文本 = 裁判生成(裁判模型, 裁判分词器,
                                  [{"role": "user", "content": 推理评分提示.format(情境=s["情境"], 推理=txt)}])
                分 = 解析小数(裁判文本, "推理质量")
                if 分 is not None:
                    推理分.append(分)
                推理明细.append({"情境": s["情境"][:20], "模型推理": txt[:80], "裁判分": 分})
                print(f"  [推理 {i+1}] 裁判分={分}", flush=True)
            else:  # 四分支
                txt = s["回复"]
                if not txt.strip():
                    txt = "（空回复）"
                裁判文本 = 裁判生成(裁判模型, 裁判分词器,
                                  [{"role": "user", "content": 四分支提示.format(user=s["user"], reply=txt)}])
                四分支分 = {k: 解析整数(裁判文本, k) for k in 四分支统计}
                四分支明细.append({"user": s["user"][:16], "回复": txt[:60], "裁判": 四分支分})
                for k, v in 四分支分.items():
                    if v is not None:
                        四分支统计[k].append(v)
                print(f"  [四分支 {i+1}] {四分支分}", flush=True)
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        结果["结果"][模式] = {
            "情绪识别准确率": round(识别正确 / 识别总, 4) if 识别总 else 0.0,
            "识别正确/总": f"{识别正确}/{识别总}",
            "情绪推理质量(0-1)": round(sum(推理分) / len(推理分), 4) if 推理分 else 0.0,
            "四分支EI": {k: round(sum(v) / len(v), 2) if v else 0.0 for k, v in 四分支统计.items()},
            "四分支平均": round(sum(sum(v) for v in 四分支统计.values()) / max(sum(len(v) for v in 四分支统计.values()), 1), 2),
            "明细": {"识别": 识别明细, "推理": 推理明细, "四分支": 四分支明细},
        }
        print(f"  {json.dumps({k: v for k, v in 结果['结果'][模式].items() if k != '明细'}, ensure_ascii=False)}", flush=True)

    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=2)
    print(f"已保存：{输出路径}")


if __name__ == "__main__":
    主()
