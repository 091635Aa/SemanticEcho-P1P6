# -*- coding: utf-8 -*-
"""P1~P5 统一测试 —— LLM-Judge 裁判阶段（本地 7B bf16，批量推理 + 断点续跑）

对 7 模式 × 30 样本 × AB正反各一次 = 420 配对。批量 (batch=4) 推理加速。
输出：评测结果\P1_5统一_LLMJudge_30_2026.json
"""
import os
import sys
import json
import gc
import glob
import re
import time
import torch
from datetime import datetime

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors import safe_open

工作目录 = os.path.dirname(os.path.abspath(__file__))
if 工作目录 not in sys.path:
    sys.path.insert(0, 工作目录)

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
裁判模型名 = "Qwen2.5-7B-Instruct"
输出目录 = os.path.join(工作目录, "..", "评测结果")
生成路径 = os.path.join(输出目录, "P1_5统一_生成_30_2026.json")
结果路径 = os.path.join(输出目录, "P1_5统一_LLMJudge_30_2026.json")
批大小 = 4

裁判配对提示 = """你是一位严格的中文图灵测试裁判。用户说了一句话，下面有两个回复：回复A 和 回复B。

用户：{user}
回复A：{回复A}
回复B：{回复B}

其中一个回复是真人写的，另一个是 AI 生成的。
请判断：哪个回复更可能出自真人之手？

必须严格按以下 JSON 格式输出（不要输出其他内容）：
{{"更像真人": "A" 或 "B", "理由": "一句话理由"}}"""


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


def 裁判生成批量(裁判模型, 裁判分词器, 消息列表, max_new_tokens=120):
    """批量贪心生成（left padding，decoder-only 正确对齐）"""
    裁判分词器.padding_side = "left"
    提示列表 = [裁判分词器.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in 消息列表]
    inputs = 裁判分词器(提示列表, return_tensors="pt", padding=True).to(裁判模型.device)
    with torch.no_grad():
        out = 裁判模型.generate(
            inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=裁判分词器.eos_token_id,
        )
    文本列表 = []
    for i in range(len(提示列表)):
        _新 = out[i, inputs.input_ids.shape[1]:]
        文本列表.append(裁判分词器.decode(_新, skip_special_tokens=True).strip())
    return 文本列表


def 解析配对(文本):
    m = re.search(r'"更像真人"\s*[:：]\s*"([AB])"', 文本)
    if m:
        return m.group(1)
    if "回复A" in 文本 and "回复B" not in 文本.split("更像真人")[-1][:40]:
        return "A"
    if "回复B" in 文本 and "回复A" not in 文本.split("更像真人")[-1][:40]:
        return "B"
    return None


def main():
    print(f"=== P1~P5 LLM-Judge 裁判阶段（batch={批大小}）{datetime.now().strftime('%H:%M:%S')} ===", flush=True)
    with open(生成路径, "r", encoding="utf-8") as f:
        数据 = json.load(f)
    样本 = 数据["回复"]
    模式列表 = 数据["模式"]
    print(f"样本数={len(样本)} 模式={模式列表} 种子={数据.get('种子', '')}", flush=True)

    结果 = None
    if os.path.exists(结果路径):
        try:
            with open(结果路径, "r", encoding="utf-8") as f:
                结果 = json.load(f)
            print(f"断点续跑：已完成={list(结果.get('配对', {}).keys())}", flush=True)
        except Exception:
            结果 = None
    if 结果 is None:
        结果 = {"模型": 数据["模型"], "裁判": 裁判模型名, "样本数": len(样本),
                "模式": 模式列表, "种子": 数据.get("种子", ""),
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "配对": {}}

    裁判模型, 裁判分词器 = 加载裁判()
    print("裁判模型加载完成", flush=True)

    for 模式 in 模式列表:
        if 模式 in 结果["配对"]:
            print(f"[{模式}] 已完成，跳过", flush=True)
            continue
        胜 = 0
        总 = 0
        配对明细 = []
        待评 = []  # (i, AI在前)
        for i, 项 in enumerate(样本):
            for AI在前 in (True, False):
                待评.append((i, AI在前))
        print(f"[{模式}] 共 {len(待评)} 配对，开始批量裁判 ...", flush=True)
        for 起 in range(0, len(待评), 批大小):
            批次 = 待评[起:起 + 批大小]
            消息列表 = []
            元信息 = []
            for i, AI在前 in 批次:
                user = 样本[i]["user"]
                真人 = 样本[i]["girl"]
                AI = 样本[i]["回复"][模式]["文本"]
                if not AI.strip():
                    AI = "（空回复）"
                A, B = (AI, 真人) if AI在前 else (真人, AI)
                内容 = 裁判配对提示.format(user=user, 回复A=A, 回复B=B)
                消息列表.append([{"role": "user", "content": 内容}])
                元信息.append((样本[i]["序号"], AI在前, 内容))
            try:
                文本列表 = 裁判生成批量(裁判模型, 裁判分词器, 消息列表)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM，批大小减半重试", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                文本列表 = []
                for m in 消息列表:
                    文本列表.append(裁判生成批量(裁判模型, 裁判分词器, [m])[0])
            for (序号, AI在前, 内容), 文本 in zip(元信息, 文本列表):
                选择 = 解析配对(文本)
                if 选择 is None:
                    # 单条重试一次
                    重试文本 = 裁判生成批量(裁判模型, 裁判分词器,
                                      [[{"role": "user", "content": 内容}]])[0]
                    选择 = 解析配对(重试文本)
                if 选择 is None:
                    continue
                AI胜 = (选择 == "A") if AI在前 else (选择 == "B")
                胜 += 1 if AI胜 else 0
                总 += 1
                配对明细.append({"序号": 序号, "AI在前": AI在前, "裁判选择": 选择,
                                "AI胜": AI胜, "裁判原文": ""})
            if (起 // 批大小 + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  [{模式}] {起+len(批次)}/{len(待评)} 暂win_rate={胜/max(总,1):.4f}", flush=True)
                结果["配对"][模式] = {"win_rate": round(胜 / max(总, 1), 4),
                                    "胜": 胜, "总": 总, "明细": 配对明细, "未完成": True}
                with open(结果路径, "w", encoding="utf-8") as f:
                    json.dump(结果, f, ensure_ascii=False, indent=2)
        win_rate = 胜 / max(总, 1)
        结果["配对"][模式] = {"win_rate": round(win_rate, 4), "胜": 胜, "总": 总,
                            "明细": 配对明细}
        with open(结果路径, "w", encoding="utf-8") as f:
            json.dump(结果, f, ensure_ascii=False, indent=2)
        print(f"  [{模式}] win_rate={win_rate:.4f} ({胜}/{总})", flush=True)

    print("\n================ 汇总 ================", flush=True)
    for 模式 in 模式列表:
        wr = 结果["配对"][模式]["win_rate"]
        print(f"{模式:<12} win_rate={wr:.4f}", flush=True)
    print(f"\n裁判完成，已保存：{结果路径}", flush=True)


if __name__ == "__main__":
    main()
