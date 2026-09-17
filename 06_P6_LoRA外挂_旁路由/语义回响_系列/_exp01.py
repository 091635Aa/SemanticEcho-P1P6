# -*- coding: utf-8 -*-
import json, io
p = r'f:\lora外挂\语义回响_系列\ep01_p1\script.json'
s = json.load(io.open(p, encoding='utf-8-sig'))
scenes = s['scenes']
# 在 index 7 (P1 wall) 之后插入 P1 补充场景
extra = [
    {"kicker":"P1 · RETENTION","title":"How long should an emotion echo?","points":["Decay beats sliding window","Sliding window beats global keep","Recency matters"],
     "en":"Semantic Echo also studies how long an emotion should echo. Three retention strategies were tested: decay, a sliding window, and keeping everything forever. Decay wins, sliding window second, global retention last. Recent feelings, weighted more.","zh":"语义回响还研究了情绪应该回响多久。测试了三种保留策略：衰减、滑动窗口、永久保留。衰减最优，滑动窗口其次，全局保留最差。越近的情绪，权重越高。"},
    {"kicker":"P1 · CHAIN OF THOUGHT","title":"Interrupting the collapse","points":["Qwen3 collapses under echo","CoT interruption rescues it","Repetition 0.84 → 0.0036"],
     "en":"There is a special failure mode: chain of thought models like Qwen3 can collapse under the echo. The fix interrupts the reasoning chain before it spins out. Repetition drops from 0.84 to 0.0036, from broken record back to normal.","zh":"有一种特殊的失效模式：Qwen3 这类思考链模型在回响下会坍缩。修复方式是在思考链失控前打断它。重复率从 0.84 降到 0.0036，从复读机恢复正常。"},
    {"kicker":"P1 · QUANTIZATION","title":"4-bit safe","points":["Quantization barely disturbs echo","Deviation under 18%","Runs on laptops"],
     "en":"And it survives compression. Four-bit quantization, the kind that makes models run on laptops, barely disturbs the echo, with a deviation under eighteen percent. The emotional boost survives the diet.","zh":"而且它经得起压缩。4bit 量化这种让模型能在笔记本上跑的压缩方式，对回响几乎无干扰，偏差低于 18%。情感增益在瘦身后依然成立。"},
    {"kicker":"P1.5 · BOUNDARY","title":"Honest limits","points":["It is a tuner, not a new mechanism","Still needs local weights","Long context still unsolved"],
     "en":"Honest part for P1.5. It is a smart tuner, not a new emotion mechanism. It still needs local weights, so closed APIs are out. And long-context collapse beyond five hundred tokens is not fully solved yet.","zh":"P1.5 的诚实部分：它是一个智能调参器，不是新的情感机制。它仍需要本地权重，闭源 API 用不了。而且超过五百 token 的长上下文坍缩还没有完全解决。"},
]
idx = 8  # 在 P1.5 fix 之前插入
scenes = scenes[:idx] + extra + scenes[idx:]
s['scenes'] = scenes
io.open(p, 'w', encoding='utf-8').write(json.dumps(s, ensure_ascii=False, indent=2))
print('ep01 scenes:', len(scenes))
