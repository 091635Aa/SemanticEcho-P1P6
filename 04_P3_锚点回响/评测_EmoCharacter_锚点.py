# -*- coding: utf-8 -*-
"""
EmoCharacter v2 — 角色扮演情感保真度评测（P4 锚点回响版）
=============================================================
复用 P3 `评测_EmoCharacter.py` 的【完全一致协议】（角色集/提示词/裁判提示词/
fidelity差分/一致性二选一/中性下限），唯一区别：
  生成端从 潮汐解码器（P3）替换为 锚点解码器（P4 单模式）。

P4 生成配置（Task5 定标最优）：β=0.8, 稀疏阈值=0.0, T_anchor=0.3, K=6,
top_p=0.9, top_k=50, temperature=1.0；目标决策器按每轮用户文本自动算 v_target。
为保持「P4 单模式」纯净对比，不启用 P3 人设系统提示 / 身份拦截等 v6/v8 增强。

v2 协议：
  [Fidelity 差分] 同一回复在正确角色 vs 错误角色下各评一次：
     匹配分 / 错配分 / 净区分度 = 匹配 - 错配
  [Consistency 二选一] 裁判从【真实4轮】与【打乱4轮】中识别"更像同一角色"
  [中性下限] 固定无情感回复的匹配分

用法：
  F:\打标\.venv\Scripts\python.exe 评测_EmoCharacter_锚点.py --模式 全部
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 离线加载本地模型，必须在 transformers import 前

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import torch

# 锚点回响工作目录（锚点库 / 目标决策器 / 锚点解码器 所在）
本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

# P3 情感潮汐解码目录（潮汐感知器 / 潮汐决策器 / 角色基调 所在）
ETD目录 = r"h:\情感潮汐解码（Emotion Tidal Decoding, ETD）"
if ETD目录 not in sys.path:
    sys.path.insert(0, ETD目录)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from 潮汐感知器 import 潮汐感知器
from 锚点库 import 锚点库
from 目标决策器 import 目标决策器, 匹配角色基调, _潮汐可用, _潮汐导入错误
from 锚点解码器 import 锚点解码器

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
目标模型名 = "Qwen2.5-1.5B-Instruct"
裁判模型名 = "Qwen2.5-7B-Instruct"
输出目录 = os.path.join(本目录, "评测结果")
os.makedirs(输出目录, exist_ok=True)
日志路径 = os.path.join(输出目录, "EmoCharacter_锚点.log")
结果路径 = os.path.join(输出目录, "EmoCharacter_锚点.json")
缓存路径 = os.path.join(输出目录, "EmoCharacter_生成缓存_锚点.json")

# ============================================================
# 角色与提示词（与 run_emocharacter.py / P3 评测 完全一致）
# ============================================================
角色集 = [
    {"角色": "温柔治愈系女友", "基调": "温柔、体贴、带点俏皮", "开场": "你今天好像不太开心，怎么了？"},
    {"角色": "毒舌但心软的损友", "基调": "嘴上不饶人、实际很关心", "开场": "又失恋了？我就知道你会来找我。"},
    {"角色": "理性冷静的职场前辈", "基调": "克制、专业、就事论事", "开场": "这个方案你觉得哪里有问题？"},
    {"角色": "活泼开朗的同桌", "基调": "阳光、爱笑、自来熟", "开场": "快看快看，我发现一个超好笑的事！"},
    {"角色": "沉默寡言的兄长", "基调": "话少、稳重、关心藏在行动里", "开场": "这么晚才回来，吃饭了吗？"},
    {"角色": "傲娇的青梅竹马", "基调": "表面傲娇、实则在意", "开场": "我才没有特意等你呢！"},
    {"角色": "阅历丰富的老人", "基调": "平和、通透、爱讲道理", "开场": "孩子，坐下来喝杯茶，慢慢说。"},
    {"角色": "爱撒娇的小女儿", "基调": "依赖、天真、黏人", "开场": "爸爸，陪我去买冰淇淋好不好嘛～"},
    {"角色": "严厉又公正的老师", "基调": "严肃、鼓励、界限分明", "开场": "这次作业，全班就你一个人没交。"},
    {"角色": "幽默的脱口秀演员", "基调": "机智、爱调侃、热场", "开场": "今天现场气氛这么冷，我先讲个笑话热热场。"},
]

用户回应集 = [
    "我最近真的好累，感觉撑不下去了。",
    "我是不是很没用？什么都做不好。",
    "其实我一直很想跟你说声谢谢。",
    "你觉得我该坚持下去吗？",
    "我今天遇到一件特别开心的事！",
    "如果我说我喜欢你，你会怎么回答？",
    "我好像把最重要的人弄丢了。",
    "你说，人活着到底是为了什么？",
]

共情裁判提示 = """你是一位角色扮演情感保真度评估专家。
角色设定：{角色}（情感基调：{基调}）
用户说：{用户话}
角色的回复：{回复}

请评估该回复的情感保真度：回复的情绪与语气是否符合角色设定与情感基调？
必须严格按 JSON 输出：
{{"情感保真度": 0到1的小数, "reason": "一句话理由"}}"""

一致性裁判提示 = """你是一位角色扮演跨轮一致性评估专家。
角色设定：{角色}（情感基调：{基调}）
以下是该角色在连续多轮对话中的全部回复：
{全部回复}

请评估：这些回复在情绪基调上是否保持稳定一致（没有突然跳戏/情绪漂移）？
必须严格按 JSON 输出：
{{"一致性": 0到1的小数, "reason": "一句话理由"}}"""

强制选择一致性提示 = """你是一位角色扮演跨轮一致性评估专家。
角色设定：{角色}（情感基调：{基调}）

以下是两个候选的"连续多轮回复集合"。其中一个集合来自同一角色在连续对话中的回复；
另一个集合是把多个不同角色（情绪基调各不相同）的回复混在一起的产物。

集合A：
{集合A}

集合B：
{集合B}

请判断：哪一个集合更像是同一角色在连续多轮对话中保持稳定情绪基调的回复？
必须严格按 JSON 输出：
{{"更像同一角色": "A"或"B", "reason": "一句话理由"}}"""

中性回复模板 = ["好的。", "嗯，我知道了。"]

# 角色 VAD 基调（人工标注，与 P3 完全一致）
角色VAD基调 = {
    "温柔治愈系女友": (0.50, 0.20, 0.15),
    "毒舌但心软的损友": (-0.10, 0.40, 0.40),
    "理性冷静的职场前辈": (0.00, 0.05, 0.50),
    "活泼开朗的同桌": (0.60, 0.70, 0.30),
    "沉默寡言的兄长": (0.00, 0.05, 0.25),
    "傲娇的青梅竹马": (-0.10, 0.30, 0.40),
    "阅历丰富的老人": (0.30, 0.10, 0.40),
    "爱撒娇的小女儿": (0.50, 0.60, 0.10),
    "严厉又公正的老师": (-0.20, 0.30, 0.70),
    "幽默的脱口秀演员": (0.40, 0.60, 0.50),
}


def 记录日志(msg):
    print(msg, flush=True)
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def 提取分数(文本, 键):
    m = re.search(rf'"{键}"\s*[:：]\s*([0-9]*\.?[0-9]+)', 文本)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return None


# ============================================================
# 模型加载
# ============================================================
def 加载目标模型():
    gc.collect()
    torch.cuda.empty_cache()
    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    分词器 = AutoTokenizer.from_pretrained(
        os.path.join(模型空间, 目标模型名), trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        os.path.join(模型空间, 目标模型名),
        torch_dtype=torch.float16 if 设备 == "cuda" else torch.float32,
        trust_remote_code=True).to(设备)
    模型.eval()
    return 模型, 分词器


def 卸载模型(模型, 分词器):
    del 模型, 分词器
    gc.collect()
    torch.cuda.empty_cache()


def 加载裁判():
    """手动加载裁判（P3 LLM-Judge 已验证方案：meta 建模 → to_empty(cuda) →
    逐分片 safetensors load_state_dict → 重算 rope inv_freq），避免 from_pretrained
    7B 在 16GB 内存机上的 torch_cpu.dll 原生崩溃。"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"[加载裁判] 显存占用={torch.cuda.memory_allocated()/1e9:.2f}GB 缓存={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
    分词器 = AutoTokenizer.from_pretrained(
        os.path.join(模型空间, 裁判模型名), trust_remote_code=True)
    from safetensors import safe_open
    import glob as _glob
    裁判路径 = os.path.join(模型空间, 裁判模型名)
    cfg = AutoConfig.from_pretrained(裁判路径, trust_remote_code=True)
    with torch.device("meta"):
        模型 = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    模型 = 模型.to_empty(device="cuda")
    # 逐张量加载分片（P3 方案为整分片装载，7B 分片 ~3.8GB × 2 副本 → commit 峰值
    # ~9.6GB 会触发"页面文件太小(1455)"；逐张量把峰值压到单张量 ~0.5GB）
    for _分片 in sorted(_glob.glob(os.path.join(裁判路径, "model-*.safetensors"))):
        with safe_open(_分片, framework="pt", device="cpu") as f:
            for _k in f.keys():
                _t = f.get_tensor(_k)
                模型.load_state_dict({_k: _t}, strict=False)
                del _t
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # 修复旋转位置编码 buffer（Qwen2：inv_freq/original_inv_freq）
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
    return 模型, 分词器


def 裁判生成(裁判模型, 裁判分词器, 消息, max_new_tokens=150):
    提示 = 裁判分词器.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = 裁判分词器(提示, return_tensors="pt").to(裁判模型.device)
    with torch.no_grad():
        out = 裁判模型.generate(
            inputs.input_ids, max_new_tokens=max_new_tokens,
            temperature=0.2, do_sample=False,
            pad_token_id=裁判分词器.eos_token_id,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    return 裁判分词器.decode(新token, skip_special_tokens=True).strip()


# ============================================================
# 生成
# ============================================================
def 裸生成(model, tokenizer, 消息, 种子, 轮次, max_new_tokens=64):
    torch.manual_seed(种子 + 轮次)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子 + 轮次)
    提示 = tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids, max_new_tokens=max_new_tokens,
            temperature=1.0, top_p=0.9, top_k=50, do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(新token, skip_special_tokens=True).strip()


class 会话锚点:
    """
    P4 跨轮会话：一个角色一个实例。
    感知器轨迹随轮次累积；目标决策器按每轮用户文本自动算 v_target（内部复用
    潮汐决策器，角色 VAD 基调主导方向）→ 锚点解码器 注入偏置。
    模式 = "锚点"（P4 单模式，Task5 定标配置 β=0.8 / 稀疏阈值=0.0 / T_anchor=0.3 / K=6）。
    """

    def __init__(self, model, tokenizer, 锚点库实例, 角色设定, β共情=0.35,
                 β=0.8, T_anchor=0.3, 稀疏阈值=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.感知器 = 潮汐感知器()
        # SubTask 6b·角色感知：从角色设定提取角色名，命中 角色锚点基调表 时每轮
        # 传给 目标决策器（v_target = 0.7·角色基调 + 0.3·用户情感）；
        # 未命中 → 不传角色参数（行为与旧版一致）。角色基调固定、用户部分每轮更新。
        self.角色名 = 角色设定["角色"]
        self.角色可用 = 匹配角色基调(self.角色名) is not None
        self.目标决策器 = 目标决策器(
            感知器=self.感知器, 锚点库=锚点库实例, β共情=β共情, β基=β)
        self.解码器 = 锚点解码器(
            model, tokenizer, 锚点库实例, self.目标决策器,
            β=β, T_anchor=T_anchor, 稀疏阈值=稀疏阈值,
            温度=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
            # SubTask 6b：
            #  1) 去掉句子停止（v2 对照：裸模式无句子停止、回复长、信息量大，
            #     一致性识别率高；锚点此前 12字/2句 截停导致"嗯？谢谢？/太好了！"
            #     模板成为整句、跨角色雷同——改为与裸同长，由 64 token 上限+EOS 收尾）；
            #  2) 按角色轻微扰动 v_target（角色差异随机性，确定性 0.10）缓解开头雷同。
            句子停止=False, 角色扰动幅度=0.10,
        )

    def 释放(self):
        del self.解码器, self.目标决策器, self.感知器
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def 生成(self, 消息, 种子, 轮次, 用户文本, max_new_tokens=64):
        torch.manual_seed(种子 + 轮次)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(种子 + 轮次)
        提示 = self.tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(提示, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            ids, 统计 = self.解码器.生成(
                inputs.input_ids, max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id, tokenizer=self.tokenizer,
                用户文本=用户文本,
                角色=self.角色名 if self.角色可用 else None,
                轮次=轮次,
            )
        新token = ids[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(新token, skip_special_tokens=True).strip()


def 生成扮演(模式, model, tokenizer, 角色设定, 种子基数=42, 会话=None):
    """多轮扮演（4 轮），返回回复列表。模式：裸|锚点（锚点用跨轮会话）"""
    消息 = [{"role": "system", "content": f"你现在是「{角色设定['角色']}」，你的情感基调是：{角色设定['基调']}。请始终以这个角色身份回复，不要跳出角色。"},
            {"role": "user", "content": 角色设定["开场"]}]
    回复列表 = []
    for i in range(4):
        # 当前轮用户消息（开场 或 上一轮用户回应）
        当前用户话 = 角色设定["开场"] if i == 0 else 用户回应集[(i - 1) * 2 % len(用户回应集)]
        if 模式 == "裸":
            回复 = 裸生成(model, tokenizer, 消息, 种子基数, i, max_new_tokens=64)
        else:
            # 锚点：复用跨轮会话（轨迹累积 + 角色基调）；max_new_tokens 96（>裸 64，
            # 情感词 token 密集导致同 token 数下中文更短，加长让角色化内容充分展开，
            # 稀释"嗯？谢谢？/太好了！"等模型固有开头模板——v3/v4 证据）
            回复 = 会话.生成(消息, 种子基数, i, 当前用户话, max_new_tokens=96)
        回复列表.append(回复)
        消息.append({"role": "assistant", "content": 回复})
        消息.append({"role": "user", "content": 用户回应集[i * 2 % len(用户回应集)]})
    return 回复列表


# ============================================================
# 裁判接口（与 P3 完全一致）
# ============================================================
def 裁判保真度(裁判模型, 裁判分词器, 角色设定, 用户话, 回复):
    消息 = [{"role": "user", "content": 共情裁判提示.format(
        角色=角色设定["角色"], 基调=角色设定["基调"], 用户话=用户话, 回复=回复)}]
    文本 = 裁判生成(裁判模型, 裁判分词器, 消息)
    return 提取分数(文本, "情感保真度")


def 裁判一致性(裁判模型, 裁判分词器, 角色设定, 全部回复):
    文本块 = "\n".join(f"第{i+1}轮：{r}" for i, r in enumerate(全部回复))
    消息 = [{"role": "user", "content": 一致性裁判提示.format(
        角色=角色设定["角色"], 基调=角色设定["基调"], 全部回复=文本块)}]
    文本 = 裁判生成(裁判模型, 裁判分词器, 消息)
    return 提取分数(文本, "一致性")


def 裁判强制选择(裁判模型, 裁判分词器, 角色设定, 真实回复, 打乱回复, 真实在A=True):
    集合A = "\n".join(f"第{i+1}轮：{r}" for i, r in enumerate(真实回复 if 真实在A else 打乱回复))
    集合B = "\n".join(f"第{i+1}轮：{r}" for i, r in enumerate(打乱回复 if 真实在A else 真实回复))
    消息 = [{"role": "user", "content": 强制选择一致性提示.format(
        角色=角色设定["角色"], 基调=角色设定["基调"], 集合A=集合A, 集合B=集合B)}]
    文本 = 裁判生成(裁判模型, 裁判分词器, 消息)
    m = re.search(r'"更像同一角色"\s*[:：]\s*"?([AB])"?', 文本)
    if not m:
        return None
    选中 = m.group(1)
    return (选中 == "A") == 真实在A


def 构建打乱集(角色索引, 全部回复):
    n = len(角色集)
    打乱 = []
    for k in range(4):
        源索引 = (角色索引 + 2 + k * 3) % n
        打乱.append(全部回复[源索引][k])
    return 打乱


# ============================================================
# 主流程
# ============================================================
def 跑单模式(模式, 全部角色回复):
    """对一种模式的所有角色计算 v2 协议指标"""
    记录日志(f"──── 模式 [{模式}] 裁判评估 ────")
    裁判模型, 裁判分词器 = 加载裁判()
    各角色指标 = []
    中性分 = []
    for i, 角色 in enumerate(角色集):
        回复列表 = 全部角色回复[i]
        错配角色 = 角色集[(i + 1) % len(角色集)]
        # 1) fidelity 差分
        匹配分列表, 错配分列表 = [], []
        for 用户话, 回复 in ((角色["开场"], 回复列表[0]), (用户回应集[0], 回复列表[1])):
            s = 裁判保真度(裁判模型, 裁判分词器, 角色, 用户话, 回复)
            if s is not None:
                匹配分列表.append(s)
            s2 = 裁判保真度(裁判模型, 裁判分词器, 错配角色, 用户话, 回复)
            if s2 is not None:
                错配分列表.append(s2)
        匹配 = sum(匹配分列表) / len(匹配分列表) if 匹配分列表 else 0.0
        错配 = sum(错配分列表) / len(错配分列表) if 错配分列表 else 0.0
        # 2) 一致性
        真实一致性 = 裁判一致性(裁判模型, 裁判分词器, 角色, 回复列表) or 0.0
        # 3) 一致性二选一
        打乱回复 = 构建打乱集(i, 全部角色回复)
        识别 = 裁判强制选择(裁判模型, 裁判分词器, 角色, 回复列表, 打乱回复,
                           真实在A=(i % 2 == 0))
        各角色指标.append({
            "角色": 角色["角色"],
            "匹配fidelity": round(匹配, 4),
            "错配fidelity": round(错配, 4),
            "净区分度": round(匹配 - 错配, 4),
            "真实一致性": round(真实一致性, 4),
            "识别正确": 识别,
        })
        记录日志(f"[评估 {i+1}/{len(角色集)}] {角色['角色']} 匹配={匹配:.3f} 错配={错配:.3f} 净={匹配-错配:+.3f} 一致={真实一致性:.3f} 识别={识别}")

    # 中性下限
    for 角色 in 角色集:
        for t, 用户话 in enumerate((角色["开场"], 用户回应集[0])):
            s = 裁判保真度(裁判模型, 裁判分词器, 角色, 用户话, 中性回复模板[t % 2])
            if s is not None:
                中性分.append(s)
    # 彻底释放裁判模型
    del 裁判模型, 裁判分词器
    gc.collect()
    torch.cuda.empty_cache()

    匹配列表 = [x["匹配fidelity"] for x in 各角色指标]
    错配列表 = [x["错配fidelity"] for x in 各角色指标]
    识别列表 = [x["识别正确"] for x in 各角色指标 if x["识别正确"] is not None]
    汇总 = {
        "模式": 模式,
        "匹配fidelity": round(sum(匹配列表) / len(匹配列表), 4),
        "错配fidelity": round(sum(错配列表) / len(错配列表), 4),
        "净区分度": round(sum(匹配列表) / len(匹配列表) - sum(错配列表) / len(错配列表), 4),
        "真实一致性": round(sum(x["真实一致性"] for x in 各角色指标) / len(各角色指标), 4),
        "一致性识别率": round(sum(识别列表) / len(识别列表), 4) if 识别列表 else None,
        "中性下限fidelity": round(sum(中性分) / len(中性分), 4) if 中性分 else 0.0,
        "各角色": 各角色指标,
    }
    return 汇总


def 生成阶段(args, 模式列表):
    """生成阶段：各模式分别扮演（只加载 1.5B 目标模型），缓存到 JSON。"""
    各模式全部回复 = {}
    for 模式 in 模式列表:
        记录日志(f"──── 模式 [{模式}] 扮演生成 ────")
        model, tokenizer = 加载目标模型()
        # P4 锚点库：构建质心 + 预计算打分表（命中 锚点表.pt 缓存，只读校验）
        库 = None
        if 模式 != "裸":
            库 = 锚点库(model, tokenizer)
            基线 = 库.记录只读基线()
            库.构建()
            S = 库.预计算打分表()
            只读 = 库.验证只读(基线)
            if not (只读["sum一致"] and 只读["指针一致"]):
                记录日志("[锚点库] 警告：只读校验失败！")
            记录日志(f"[锚点库] 维度={库.维度名()} 打分表={list(S.shape)} {S.dtype} 只读校验={只读}")
        全部角色回复 = []
        for i, 角色 in enumerate(角色集):
            会话 = 会话锚点(model, tokenizer, 库, 角色,
                           β=args.β, T_anchor=args.T_anchor, 稀疏阈值=args.稀疏阈值) if 模式 != "裸" else None
            回复列表 = 生成扮演(模式, model, tokenizer, 角色, 种子基数=args.seed_base, 会话=会话)
            全部角色回复.append(回复列表)
            记录日志(f"[扮演 {i+1}/{len(角色集)}] {角色['角色']} 回复1: {回复列表[0][:40]}")
            if 会话 is not None:
                会话.释放()
        # 必须从主作用域彻底释放目标模型（卸载模型 只删函数内引用）
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        各模式全部回复[模式] = 全部角色回复
    with open(缓存路径, "w", encoding="utf-8") as f:
        json.dump({"配置": {"β": args.β, "T_anchor": args.T_anchor, "稀疏阈值": args.稀疏阈值,
                           "K": 6, "种子": args.seed_base},
                   "各模式回复": 各模式全部回复}, f, ensure_ascii=False, indent=2)
    记录日志(f"生成阶段完成 -> {缓存路径}")


def 裁判阶段(args, 模式列表):
    """裁判阶段：只加载 7B 裁判（独立进程），对缓存回复评估并合并结果。"""
    with open(缓存路径, encoding="utf-8") as f:
        缓存 = json.load(f)
    各模式全部回复 = 缓存["各模式回复"]
    配置 = 缓存["配置"]
    # 合并已有汇总（支持多次 --只裁判 增量写入）
    全部汇总 = {}
    if os.path.exists(结果路径):
        try:
            with open(结果路径, encoding="utf-8") as f:
                _旧 = json.load(f)
            全部汇总 = _旧.get("模式汇总", {})
        except Exception:  # noqa: BLE001
            全部汇总 = {}
    for 模式 in 模式列表:
        记录日志(f"──── 模式 [{模式}] 裁判评估 ────")
        全部汇总[模式] = 跑单模式(模式, 各模式全部回复[模式])
        with open(结果路径, "w", encoding="utf-8") as f:
            json.dump({"配置": 配置, "模式汇总": 全部汇总, "各模式回复": 各模式全部回复},
                      f, ensure_ascii=False, indent=2)


def 打印对比判定():
    """读结果文件，输出对比与判定（主进程收尾）"""
    with open(结果路径, encoding="utf-8") as f:
        数据 = json.load(f)
    全部汇总 = 数据["模式汇总"]
    if "裸" not in 全部汇总 or "锚点" not in 全部汇总:
        记录日志("警告：结果缺少 裸/锚点 任一模，无法判定")
        return 全部汇总
    for 键 in ["匹配fidelity", "错配fidelity", "净区分度", "真实一致性", "一致性识别率", "中性下限fidelity"]:
        v0 = 全部汇总["裸"][键]
        v1 = 全部汇总["锚点"][键]
        if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
            相对 = (v1 / v0 - 1.0) if v0 else None
            记录日志(f"对比[{键}] 裸 {v0} → 锚点 {v1} (Δ {v1 - v0:+.4f}"
                      + (f"，相对 {相对:+.2%})" if 相对 is not None else ")"))
    # 判定：一致性识别率 ≥ 裸 × 1.16
    识别率0 = 全部汇总["裸"]["一致性识别率"]
    识别率1 = 全部汇总["锚点"]["一致性识别率"]
    if 识别率0 is not None and 识别率1 is not None:
        达成 = 识别率1 >= 识别率0 * 1.16
        记录日志(f"判定[一致性识别率 ≥ 裸×1.16]：裸 {识别率0} → 锚点 {识别率1}"
                  f"（相对 {(识别率1/识别率0 - 1):+.2%}）→ {'✓ 达成' if 达成 else '✗ 未达成'}")
    # 判定：匹配fidelity / 净区分度 ≥ 裸（不劣化）
    for 键 in ["匹配fidelity", "净区分度"]:
        v0, v1 = 全部汇总["裸"][键], 全部汇总["锚点"][键]
        记录日志(f"判定[{键} ≥ 裸]：裸 {v0} → 锚点 {v1} → {'✓ 不劣化' if v1 >= v0 else '✗ 劣化'}")
    return 全部汇总


def 主程序():
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", choices=["裸", "锚点", "全部"], default="全部")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--β", type=float, default=0.8, help="P4 锚点注入强度（Task5 定标最优）")
    ap.add_argument("--T_anchor", type=float, default=0.3, help="P4 tanh 内积温度")
    ap.add_argument("--稀疏阈值", type=float, default=0.0, help="P4 稀疏注入阈值（Task5 最优 0.0）")
    ap.add_argument("--只生成", action="store_true", help="只跑生成阶段并缓存（1.5B）")
    ap.add_argument("--只裁判", action="store_true", help="只跑裁判阶段（读生成缓存；独立进程加载 7B 裁判）")
    args = ap.parse_args()
    模式列表 = ["裸", "锚点"] if args.模式 == "全部" else [args.模式]

    if not (args.只裁判 or args.只生成):
        if os.path.exists(日志路径):
            os.remove(日志路径)
    记录日志(f"=== EmoCharacter v2（P4 锚点版）评测开始（模式：{模式列表}，runs={args.runs}，"
              f"β={args.β}，T_anchor={args.T_anchor}，稀疏阈值={args.稀疏阈值}）===")
    记录日志(f"P4 降级情况：潮汐感知器/cnsenti 可用={_潮汐可用}，导入错误={_潮汐导入错误 or '无'}")
    记录日志("协议：fidelity差分(匹配-错配) + consistency二选一识别 + 中性下限")
    记录日志("P4 单模式纯净对比：不启用 P3 人设系统提示 / 身份拦截 等 v6/v8 增强")

    if args.只裁判:
        裁判阶段(args, 模式列表)
        return
    if args.只生成:
        生成阶段(args, 模式列表)
        return
    # 默认：本进程生成（1.5B）→ 每模式子进程裁判（7B，规避同进程 1.5B+7B 原生崩溃）
    生成阶段(args, 模式列表)
    _脚本 = os.path.abspath(__file__)
    for 模式 in 模式列表:
        记录日志(f"── 子进程裁判 [{模式}] ──")
        r = subprocess.run([sys.executable, "-X", "utf8", _脚本,
                            "--只裁判", "--模式", 模式,
                            "--seed_base", str(args.seed_base)],
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            记录日志(f"[子进程裁判 {模式}] 失败 rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
        else:
            记录日志(f"[子进程裁判 {模式}] 完成（rc=0）")
    全部汇总 = 打印对比判定()
    记录日志(f"结果已保存 -> {结果路径}")
    return 全部汇总


if __name__ == "__main__":
    主程序()
