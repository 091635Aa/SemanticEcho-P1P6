# -*- coding: utf-8 -*-
import json, io
p = r'f:\lora外挂\语义回响_系列\ep02_etd\script.json'
s = json.load(io.open(p, encoding='utf-8-sig'))
scenes = s['scenes']
extra = [
    {"kicker":"ETD · SENSING","title":"VAD: the tide gauge","points":["Valence · arousal · dominance","Covers 1000+ emotion words","No internal access needed"],
     "en":"Let us look at the sensing layer. VAD stands for valence, arousal and dominance, three axes that cover over a thousand emotion words. It reads the surface text only, so it works even on closed APIs with no internal access.","zh":"我们来看感知层。VAD 即效价、唤醒度、支配度三个轴，覆盖一千多个情感词。它只读表面文本，所以在没有任何内部访问权限的闭源 API 上也能用。"},
    {"kicker":"ETD · BUDGET","title":"Emotion word budget","points":["Soft budget on emotion tokens","Keeps speech grounded","Prevents over-the-top drama"],
     "en":"The expression layer also uses an emotion word budget. It gently limits how many emotional words the model can use, keeping the speech grounded. Without it, small models tend to overact, going full drama queen.","zh":"表达层还用了情感词预算。它温和地限制模型能使用的情感词数量，让表达保持克制。没有它，小模型容易过度表演，变成\u201c戏精\u201d。"},
    {"kicker":"ETD · V6","title":"Identity & stop guards","points":["Identity rejection guard","Hard sentence stop","Stays in character"],
     "en":"Two safety guards in version six. The identity rejection guard stops the model from suddenly breaking character and giving a generic AI answer. The hard sentence stop prevents runaway generation at the end of a sentence.","zh":"v6 版本有两个安全护栏。身份拦截防止模型突然出戏、给出模板化 AI 回答。句子硬停止防止在句尾失控地继续生成。"},
    {"kicker":"ETD · V8","title":"Persona + self-pick","points":["Persona system prompt","Multiple candidates, self-select","0.5167 win rate with persona"],
     "en":"Version eight adds a persona system prompt and a self-pick trick. The model generates multiple candidate responses, then selects the best. With the persona, the win rate reaches 0.5167, above the coin-flip baseline.","zh":"v8 版本加了人设系统提示和自选技巧。模型生成多个候选回答，再自己选最好的。加上人设后，盲评胜率达到 0.5167，超过抛硬币的基线。"},
    {"kicker":"ETD · MODELS","title":"Transfers across sizes","points":["1.5B mix: +87.5%","3B tide: +33%","Qwen3 flat: -4%"],
     "en":"How does it transfer? At 1.5 billion, the mixed scheme gains 87.5 percent. At 3 billion, the tide gains 33 percent. On Qwen3 it stays flat, minus four percent, within variance. It is strongest on smaller chatty models.","zh":"迁移性如何？1.5B 上混合方案提升 87.5%。3B 上潮汐提升 33%。在 Qwen3 上持平，约 -4%，在方差内。它在更小的对话模型上最强。"},
    {"kicker":"ETD · BOUNDARY","title":"The honest list","points":["Lexicon misses internet slang","Mixed scheme limited to Qwen2 family","Not a general human-ness boost"],
     "en":"Final honest list. The lexicon misses internet slang. The full mixed scheme is currently tuned for the Qwen2 family. And ETD is an empathy specialist, not a general human-ness upgrade for every task.","zh":"最后的诚实清单：词库对网络流行语识别弱；完整混合方案目前针对 Qwen2 系调优；ETD 是共情专家，不是所有任务的通用人味升级包。"},
]
idx = 5
scenes = scenes[:idx] + extra + scenes[idx:]
s['scenes'] = scenes
io.open(p, 'w', encoding='utf-8').write(json.dumps(s, ensure_ascii=False, indent=2))
print('ep02 scenes:', len(scenes))
