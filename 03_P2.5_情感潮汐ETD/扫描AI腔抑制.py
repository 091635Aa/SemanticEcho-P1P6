# -*- coding: utf-8 -*-
"""
扫描 AI 腔抑制：诊断 + 强度扫描
================================
Step1 诊断：AI 腔词在 Qwen2.5 tokenizer 中的切词情况
  → 若多为多 token，则 _构建AI腔token表 过滤后抑制根本没生效（这就是 v3 效果微弱的原因）
Step2 扫描：AI抑制 ∈ {0, 2, 4, 6, 8} × 30 条 prompt
  → 统计每条输出命中 AI 腔短语的条数/词频/长度，找能真正压住 AI 腔的配置
"""
import json
import os
import re
import sys
import torch

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from transformers import AutoModelForCausalLM, AutoTokenizer
from 潮汐感知器 import 潮汐感知器
from 潮汐决策器 import 潮汐决策器
from 潮汐解码器 import 潮汐解码器

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
样本路径 = r"i:\Desktop\语义回响\图灵测试\样本_30条.json"

# 多 token 级 AI 腔短语（正则，任意长度都检测）
AI腔短语 = [
    r"作为.{0,8}(?:AI|人工智能|语言模型|模型|助手|智能)",
    r"我是.{0,4}AI|我是一个.{0,6}(?:AI|模型|助手)",
    r"AI\s*(?:助手|模型|语言模型)",
    r"人工智能助手",
    r"语言模型",
    r"我的(?:主要)?目标",
    r"来帮助你|帮助用户|为用户提供|提供服务|来帮助和指导",
    r"请(?:随时|告诉我)|有什么可以帮|欢迎告诉我",
    r"我(?:的)?存在(?:就是|是)?为了|被设计",
    r"无法(?:感受|理解|感知|回答|记得)|不能回答",
    r"抱歉|对不起",
    r"很高兴.{0,6}(?:为你|帮助)|非常高兴",
    r"上下班的概念|24小时在线",
    r"根据(?:我的|我的能力)",
]
AI腔正则 = [re.compile(p) for p in AI腔短语]


def 计AI腔(文本: str) -> int:
    return sum(1 for r in AI腔正则 if r.search(文本))


def 潮汐生成(model, tokenizer, 提示, 种子, AI抑制=0.0):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    感知器 = 潮汐感知器()
    决策器 = 潮汐决策器(感知器)
    解码器 = 潮汐解码器(model, tokenizer, 感知器, 决策器, AI腔抑制强度=AI抑制)
    消息 = [{"role": "user", "content": 提示}]
    提示文 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(提示文, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = 解码器.生成(inputs.input_ids, max_new_tokens=64,
                          temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
                          eos_token_id=tokenizer.eos_token_id, 用户文本=提示)
    新token = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(新token, skip_special_tokens=True).strip()


def 主程序():
    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[加载] → {设备} ...", flush=True)
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)

    # ── Step1 诊断：AI 腔词切词 ──
    print("════ Step1 诊断：AI腔词 切词（红色=多token被过滤→抑制无效）", flush=True)
    诊断词 = ["AI", "助手", "模型", "智能", "作为", "提供", "帮助", "用户", "服务",
              "抱歉", "对不起", "请问", "回答", "信息", "可以", "如果", "需要", "根据"]
    for 词 in 诊断词:
        ids = 分词器.encode(词, add_special_tokens=False)
        ok = "✔单token" if len(ids) == 1 else f"✘{len(ids)}token {ids}"
        print(f"  {词:4s} → {ok}", flush=True)

    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16 if 设备 == "cuda" else torch.float32,
        trust_remote_code=True).to(设备)
    模型.eval()

    with open(样本路径, encoding="utf-8") as f:
        样本 = json.load(f)["样本"][:30]

    # ── Step2 强度扫描 ──
    import sys as _sys
    强度列表 = [float(x) for x in _sys.argv[1:]] or [0.0, 2.0, 4.0, 6.0, 8.0]
    print(f"\n════ Step2 强度扫描：{len(样本)} 条 × {len(强度列表)} 强度", flush=True)
    结果 = {}
    for 强度 in 强度列表:
        命中条数 = 0
        总命中 = 0
        总长 = 0
        示例 = []
        for i, r in enumerate(样本):
            回复 = 潮汐生成(模型, 分词器, r["user"], 42, AI抑制=强度)
            n = 计AI腔(回复)
            命中条数 += 1 if n > 0 else 0
            总命中 += n
            总长 += len(回复)
            if n > 0 and len(示例) < 3:
                示例.append(f"  [{r['user'][:12]}...] {回复[:70]}")
            print(f"  [强度{强度} {i+1}/30] 命中{n} | {回复[:45]}", flush=True)
        结果[强度] = {"命中条数": 命中条数, "总命中": 总命中, "平均长度": round(总长/30, 1)}
        print(f"  → 强度{强度}: 命中条数={命中条数}/30 总命中={总命中} 平均长={结果[强度]['平均长度']}", flush=True)

    print("\n════ 汇总 ════", flush=True)
    for 强度, d in 结果.items():
        print(f"  AI抑制={强度}: {d['命中条数']}/30 条含 AI 腔, 命中 {d['总命中']} 次, 均长 {d['平均长度']}", flush=True)
    模型.to("cpu")
    del 模型, 分词器
    torch.cuda.empty_cache()


if __name__ == "__main__":
    主程序()
