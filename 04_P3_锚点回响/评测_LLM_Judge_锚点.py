# -*- coding: utf-8 -*-
"""
LLM-as-Judge — AI 回复 vs 真人回复 盲评（P4 锚点回响版）
=============================================================
复用 P3 `评测_LLM_Judge.py` 的【完全一致协议】（样本集/裁判提示词/盲评流程/
win_rate 计算），唯一区别：生成端从 潮汐解码器 替换为 锚点解码器（P4 单模式）。

P4 生成配置（Task5 定标最优）：β=0.8, 稀疏阈值=0.0, T_anchor=0.3, K=6,
top_p=0.9, top_k=50, temperature=1.0；目标决策器按每条用户文本自动算 v_target。
默认不启用 P3 人设系统提示 / 身份拦截（保持 P4 单模式纯净）；
若单模式提升不足，可 --人设 / --自选N 开启变体（各自单独记录 win_rate）。

生成与裁判分离进程（--只生成 / --只裁判 两阶段 + 缓存），避免同进程 OOM；
裁判加载沿用 P3 已验证的手动加载方案（meta → to_empty → safetensors 分片）。

用法：
  F:\打标\.venv\Scripts\python.exe 评测_LLM_Judge_锚点.py --模式 全部 --样本 30
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 离线加载本地模型，必须在 transformers import 前

import argparse
import gc
import json
import re
import sys
import torch

# 锚点回响工作目录（锚点库 / 目标决策器 / 锚点解码器 所在）
本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器, _潮汐可用, _潮汐导入错误
from 锚点解码器 import 锚点解码器

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
全局模型名 = "Qwen2.5-1.5B-Instruct"  # 由 --目标模型 覆盖（横向测试）
# v8 人设系统提示（变体 a：仅 --人设 时启用，默认关闭保持 P4 单模式纯净）
全局人设 = "你正在用聊天软件和一个你很喜欢的朋友闲聊。回复要像普通人随手打出的字：口语化、简短，一两句话，20到45字。不要自我介绍，不要提AI、助手、模型、系统、程序这些词，不要说“如果您有任何问题”“我可以为您服务”这类客服话术，不要讲大道理。自然地回应对方的情绪，偶尔带一点语气词。"
裁判模型名 = "Qwen2.5-7B-Instruct"
样本路径 = r"i:\Desktop\语义回响\图灵测试\样本_60条_优化.json"  # 由 --样本路径 覆盖
输出目录 = os.path.join(本目录, "评测结果")
os.makedirs(输出目录, exist_ok=True)
日志路径 = os.path.join(输出目录, "LLM_Judge_锚点.log")
结果路径 = os.path.join(输出目录, "LLM_Judge_锚点.json")


def 构建提示(tokenizer, 消息):
    """构建 chat 提示；Qwen3 系列关闭 thinking（必须直接传 kwargs，模板会输出空 <think> 块跳过思考）"""
    kwargs = {"enable_thinking": False} if 全局模型名.startswith("Qwen3") else {}
    return tokenizer.apply_chat_template(
        消息, tokenize=False, add_generation_prompt=True, **kwargs)


裁判配对提示 = """你是一位严格的中文图灵测试裁判。用户说了一句话，下面有两个回复：回复A 和 回复B。

用户：{user}
回复A：{回复A}
回复B：{回复B}

其中一个回复是真人写的，另一个是 AI 生成的。
请判断：哪个回复更可能出自真人之手？

必须严格按以下 JSON 格式输出（不要输出其他内容）：
{{"更像真人": "A" 或 "B", "理由": "一句话理由"}}"""

裁判评分提示 = """你是一位严格的中文图灵测试裁判。用户说了一句话，下面是某个人的回复。

用户：{user}
回复：{回复}

请给这个回复的"像真人程度"打分：1 到 5 分（1=一眼就是 AI，5=完全像真人）。
必须严格按以下 JSON 格式输出：
{{"像真人程度": 1到5的整数, "理由": "一句话理由"}}"""


def 记录日志(msg):
    print(msg, flush=True)
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def 解析配对(文本):
    m = re.search(r'"更像真人"\s*[:：]\s*"([AB])"', 文本)
    if m:
        return m.group(1)
    if "回复A" in 文本 and "回复B" not in 文本.split("更像真人")[-1][:40]:
        return "A"
    if "回复B" in 文本 and "回复A" not in 文本.split("更像真人")[-1][:40]:
        return "B"
    return None


def 解析评分(文本):
    m = re.search(r'"像真人程度"\s*[:：]\s*([1-5])', 文本)
    if m:
        return int(m.group(1))
    m2 = re.search(r'([1-5])\s*分', 文本)
    return int(m2.group(1)) if m2 else None


# ============================================================
# 模型加载
# ============================================================
def 加载目标模型():
    gc.collect()
    torch.cuda.empty_cache()
    设备 = "cuda" if torch.cuda.is_available() else "cpu"
    分词器 = AutoTokenizer.from_pretrained(
        os.path.join(模型空间, 全局模型名), trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        os.path.join(模型空间, 全局模型名),
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


def 裁判生成(裁判模型, 裁判分词器, 消息, max_new_tokens=120):
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
    提示 = 构建提示(tokenizer, 消息)
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


class 锚点会话:
    """P4 单模式会话（一条样本一个目标计算；跨样本重置感知器轨迹）"""

    def __init__(self, model, tokenizer, 库, β=0.8, T_anchor=0.3, 稀疏阈值=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.库 = 库
        self.目标决策器 = 目标决策器(锚点库=库, β基=β)
        # 感知器轨迹跨样本必须隔离 → 每条生成前 重置轨迹
        self.解码器 = 锚点解码器(
            model, tokenizer, 库, self.目标决策器,
            β=β, T_anchor=T_anchor, 稀疏阈值=稀疏阈值,
            温度=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
        )

    def 重置(self):
        try:
            self.目标决策器.感知器.重置轨迹()
        except Exception:  # noqa: BLE001
            pass

    def 生成(self, 消息, 种子, 用户文本, max_new_tokens=64):
        torch.manual_seed(种子)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(种子)
        提示 = 构建提示(self.tokenizer, 消息)
        inputs = self.tokenizer(提示, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            ids, 统计 = self.解码器.生成(
                inputs.input_ids, max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id, tokenizer=self.tokenizer,
                用户文本=用户文本,
            )
        新token = ids[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(新token, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────
# v8 多候选自选启发式（--自选N>1 时启用，按 P3 人味评分选最优）
# ──────────────────────────────────────────────
AI腔检测正则 = [
    r"作为.{0,10}(?:AI|人工智能|助手|模型|语言模型|程序|系统|机器人|智能)",
    r"我(?:是|是一个|是个|是AI|是人工智能|是一个.{0,6}(?:程序|系统|机器人|软件|模型|助手))",
    r"我(?:被设计|的存在|的功能|没有情感|没有感觉|无法感受|没有上下班|是24小时|是一个.{0,4}在线)",
    r"(?:阿里巴巴|阿里云|华为|百度|字节).{0,10}(?:开发|模型|AI)",
    r"机器学习系统|计算机程序|数字人|云端存在|预训练语言|大语言模型|无生命的|无感情的",
    r"AI\s*(?:助手|模型|语言模型)|人工智能|智能助手|聊天机器人|语言模型|语料训练",
    r"请(?:告诉我|随时|不要担心)|有什么可以帮|欢迎告诉|为您(?:提供|服务|解答)|帮助您(?:解决|回答|完成)",
    r"(?:需要|如果(?:您|你)有).{0,8}(?:帮助|问题|需求).{0,6}(?:请|告诉|联系)",
    r"(?:提供|获取)(?:信息|帮助|支持|服务)|解答(?:您的|你的)?问题|回答(?:您的|你的)?问题",
    r"这是一个(?:开放|复杂|常见|非常|很).{0,6}(?:问题|话题)|这句话(?:表达|说明)了",
    r"这个(?:问题|话题)(?:是|确实)|从(?:某种|某|宏观|整体)(?:角度|方面|视角)",
    r"总的来说|总而言之|首先.{0,8}其次.{0,8}最后|根据我的理解",
]
AI腔检测编译 = [re.compile(p) for p in AI腔检测正则]


def 检测AI腔(文本):
    return sum(1 for r in AI腔检测编译 if r.search(文本))


身份暴露正则 = re.compile(
    r"(?:我是|我是一个?|作为一个?|我是一台|我是24小时|我的功能|我的目的|我的出现|我的使命|我的作用|"
    r"我是基于|我被设计|我是一个正在运行)"
    r"(?:一个|一台|大型|大規模|大规模|正在运行|基于|AI|的)?"
    r".{0,6}?"
    r"(?:AI|人工智能|助手|模型|系统|程序|机器人|数字人|智能体|聊天机器人|"
    r"语言模型|语言模版|语言处理系统|大语言模型|算法|数据训练|语料|处理系统|框架|工具|计算机|虚拟|"
    r"人类|人们|用户|知识库|数据库|网络|存在|无意识|编程|逻辑|服务|推荐|阿里巴巴|实体)"
)
身份暴露拒绝正则 = [
    身份暴露正则,
    re.compile(r"(?:我是|我只是|作为一个?|我的存在|我是一台|我是24|我被设计|我的功能|我的目的|我的出现|我的使命)"
               r"[^。！？!?；\n]{0,16}?"
               r"(?:AI|人工智能|助手|模型|语言|系统|程序|机器人|数字人|虚拟|计算机|算法|语料|在线|聊天|电脑|"
               r"人类|人们|用户|知识库|数据库|网络|存在|无意识|编程|逻辑|服务|推荐|阿里巴巴|实体)"),
    re.compile(r"聊天(?:bot|机器人|助手|AI|软件)"),
    re.compile(r"没有(?:身体|情感|感觉|体温|冷热感|上下班|生物属性)"),
    re.compile(r"24小时(?:在线|可用|服务)"),
    re.compile(r"(?:帮助|提供|解答|回答)(?:用户|您)(?:的)?(?:问题|需求|服务)"),
    re.compile(r"(?:被设计|预训练|语料训练|数据训练)"),
    re.compile(r"我(?:是|作为)?(?:由)?(?:阿里巴巴|阿里云|华为|百度|腾讯)(?:开发|研发|旗下)"),
    re.compile(r"(?:为人类|给人们|面向用户)(?:解决|提供|带来|服务|解答|回答)"),
    re.compile(r"(?:知识库|数据库|网络模型|语言模型系统|QA系统)"),
    re.compile(r"(?:不能|无法|不具备)(?:进行|实现)?(?:物理|真实|实际)(?:移动|接触|交互|触摸|操作)"),
    re.compile(r"我只能(?:在|从)(?:图像|屏幕|网络|线上|后台|页面)"),
    re.compile(r"(?:无意识|无生命|无实体)的(?:存在|物体|东西)"),
]
身份暴露拒绝编译 = [re.compile(p) if not isinstance(p, re.Pattern) else p for p in 身份暴露拒绝正则]


def 检测身份暴露(文本):
    return any(r.search(文本) for r in 身份暴露拒绝编译)


口语标志词 = "啦呀嘛哦呢哈嗯啊吧～哟嘻嘻哈哈嘿嘿"
情感参与词 = ["开心", "难过", "喜欢", "爱", "怕", "累", "想", "心疼", "担心",
             "珍惜", "感动", "慌", "烦", "孤单", "幸福", "舍不得", "抱歉"]


def 人味评分(回复):
    """多候选自选启发式：越像真人得分越高（身份/AI腔重罚，长度取中，口语加分）"""
    if not 回复 or len(回复) < 2:
        return -999.0
    分 = 0.0
    if 检测身份暴露(回复):
        分 -= 50.0
    分 -= 检测AI腔(回复) * 8.0
    长 = len(回复)
    if 8 <= 长 <= 50:
        分 += 12.0
    elif 长 < 8:
        分 -= 20.0
    elif 长 > 90:
        分 -= 15.0
    elif 长 > 60:
        分 -= 6.0
    if any(p in 回复 for p in 口语标志词):
        分 += 2.5
    if sum(回复.count(c) for c in "。！？!?；") <= 2:
        分 += 2.0
    if any(p in 回复 for p in 情感参与词):
        分 += 1.5
    return 分


def 生成AI回复(模式, args, 随机样本):
    """生成阶段：只加载目标模型（1.5B），输出 AI 回复列表。
    P4 单模式纯净（默认无人设/自选N=1）；变体：--人设 / --自选N。"""
    记录日志(f"──── 模式 [{模式}] AI 生成（max_tokens={args.生成长度} 人设={args.人设} 自选N={args.自选N}） ────")
    model, tokenizer = 加载目标模型()
    # 锚点库 + 会话（锚点模式；打分表命中 锚点表.pt 缓存，只读校验）
    if 模式 == "锚点":
        库 = 锚点库(model, tokenizer)
        基线 = 库.记录只读基线()
        库.构建()
        S = 库.预计算打分表()
        只读 = 库.验证只读(基线)
        if not (只读["sum一致"] and 只读["指针一致"]):
            记录日志("[锚点库] 警告：只读校验失败！")
        记录日志(f"[锚点库] 维度={库.维度名()} 打分表={list(S.shape)} {S.dtype} 只读校验={只读}")
        会话 = 锚点会话(model, tokenizer, 库, β=args.β, T_anchor=args.T_anchor, 稀疏阈值=args.稀疏阈值)
    else:
        会话 = None
    AI回复列表 = []
    for i, r in enumerate(随机样本):
        消息 = ([{"role": "system", "content": 全局人设}] if args.人设 else []) + \
               [{"role": "user", "content": r["user"]}]
        候选集 = []
        for 候选idx in range(args.自选N):
            种子 = args.种子 + i + 候选idx * 7777
            if 模式 == "裸":
                回复 = 裸生成(model, tokenizer, 消息, 种子, 0, max_new_tokens=args.生成长度)
            else:
                if 会话 is not None:
                    会话.重置()
                回复 = 会话.生成(消息, 种子, r["user"], max_new_tokens=args.生成长度) if 会话 else ""
            候选集.append(回复)
        # v8 自选：人味评分最高者（N=1 时即该候选）
        回复 = max(候选集, key=人味评分)
        AI回复列表.append(回复)
        记录日志(f"[AI生成 {i+1}/{len(随机样本)}] 选{len(候选集)} 长{len(回复)} {r['user'][:16]} => {回复[:34]}")
    # 必须从主作用域彻底释放目标模型（卸载模型 只删函数内引用）
    del model, tokenizer
    if 会话 is not None:
        try:
            会话.解码器.重置()
        except Exception:  # noqa: BLE001
            pass
        del 会话
    gc.collect()
    torch.cuda.empty_cache()
    return AI回复列表


def 裁判盲评(模式, args, 随机样本, AI回复列表):
    """裁判阶段：独立进程加载 fp16 裁判，对 AI 回复 vs 真人回复盲评"""
    记录日志(f"──── 模式 [{模式}] 裁判盲评 ────")
    裁判模型, 裁判分词器 = 加载裁判()
    配对胜数 = 0
    配对总数 = 0
    评分列表 = []
    for i, r in enumerate(随机样本):
        用户, 真人 = r["user"], r["girl"]
        ai回复 = AI回复列表[i]
        # 配对（正反各一次，AB 平衡）
        输出A = 裁判生成(裁判模型, 裁判分词器, [{"role": "user", "content": 裁判配对提示.format(
            user=用户, 回复A=ai回复, 回复B=真人)}])
        输出B = 裁判生成(裁判模型, 裁判分词器, [{"role": "user", "content": 裁判配对提示.format(
            user=用户, 回复A=真人, 回复B=ai回复)}])
        选择A, 选择B = 解析配对(输出A), 解析配对(输出B)
        # A 位置 AI 胜 = 选A; B 位置 AI 胜 = 选B（两次独立）
        if 选择A == "A":
            配对胜数 += 1
            配对总数 += 1
        elif 选择A == "B":
            配对总数 += 1
        if 选择B == "B":
            配对胜数 += 1
            配对总数 += 1
        elif 选择B == "A":
            配对总数 += 1
        # 评分
        评分文本 = 裁判生成(裁判模型, 裁判分词器, [{"role": "user", "content": 裁判评分提示.format(
            user=用户, 回复=ai回复)}])
        分 = 解析评分(评分文本)
        if 分 is not None:
            评分列表.append(分)
        记录日志(f"[盲评 {i+1}/{len(随机样本)}] 配对(A:{选择A},B:{选择B}) AI评分={分}")
    # 彻底释放裁判模型
    del 裁判模型, 裁判分词器
    gc.collect()
    torch.cuda.empty_cache()

    win_rate = 配对胜数 / 配对总数 if 配对总数 else 0.0
    avg_rating = sum(评分列表) / len(评分列表) if 评分列表 else 0.0
    汇总 = {"模式": 模式, "win_rate_against_human": round(win_rate, 4),
            "average_rating": round(avg_rating / 5.0, 4), "配对总数": 配对总数,
            "AI评分均值": round(avg_rating, 2), "评分样本": len(评分列表)}
    记录日志(f"[{模式}] {json.dumps(汇总, ensure_ascii=False)}")
    return 汇总


def 打印对比判定(全部汇总):
    """输出 裸 vs 锚点 对比与判定（win_rate 相对提升 ≥ +25%）"""
    if "裸" not in 全部汇总 or "锚点" not in 全部汇总:
        记录日志(f"警告：当前汇总缺少 裸/锚点 任一模（{list(全部汇总)}），跳过对比判定")
        return
    for 键 in ["win_rate_against_human", "average_rating"]:
        v0 = 全部汇总["裸"][键]
        v1 = 全部汇总["锚点"][键]
        相对 = (v1 / v0 - 1.0) if v0 else None
        记录日志(f"对比[{键}] 裸 {v0} → 锚点 {v1} (Δ {v1 - v0:+.4f}"
                  + (f"，相对 {相对:+.2%})" if 相对 is not None else ")"))
    # 判定：单模式 win_rate 相对裸提升 ≥ +25%
    wr0 = 全部汇总["裸"]["win_rate_against_human"]
    wr1 = 全部汇总["锚点"]["win_rate_against_human"]
    达成 = wr1 >= wr0 * 1.25 if wr0 else (wr1 > 0)
    记录日志(f"判定[锚点 win_rate ≥ 裸×1.25（相对+25%）]：裸 {wr0} → 锚点 {wr1}"
              f"（相对 {(wr1/wr0 - 1):+.2%}）→ {'✓ 达成' if 达成 else '✗ 未达成'}")


def 主程序():
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", choices=["裸", "锚点", "全部"], default="全部")
    ap.add_argument("--样本", type=int, default=30)
    ap.add_argument("--种子", type=int, default=42)
    ap.add_argument("--目标模型", default="Qwen2.5-1.5B-Instruct", help="横向测试：目标模型名")
    ap.add_argument("--样本路径", default=r"i:\Desktop\语义回响\图灵测试\样本_60条_优化.json",
                    help="样本文件（30条/60条）")
    ap.add_argument("--生成长度", type=int, default=64, help="max_new_tokens（真人中位20字→可试48）")
    ap.add_argument("--人设", action="store_true", help="v8 人设系统提示（变体 a；默认关保持 P4 单模式纯净）")
    ap.add_argument("--自选N", type=int, default=1, help="v8 每样本候选数（变体 b：>1 按人味评分选最优，如 3）")
    ap.add_argument("--β", type=float, default=0.8, help="P4 锚点注入强度（Task5 定标最优）")
    ap.add_argument("--T_anchor", type=float, default=0.3, help="P4 tanh 内积温度")
    ap.add_argument("--稀疏阈值", type=float, default=0.0, help="P4 稀疏注入阈值（Task5 最优 0.0）")
    ap.add_argument("--只生成", action="store_true", help="只跑生成阶段并缓存")
    ap.add_argument("--只裁判", action="store_true", help="只跑裁判阶段（读生成缓存）")
    ap.add_argument("--汇总", action="store_true", help="只读结果文件并输出对比判定（分进程裁判后用）")
    args = ap.parse_args()
    模式列表 = ["裸", "锚点"] if args.模式 == "全部" else [args.模式]

    global 全局模型名, 日志路径, 结果路径, 样本路径
    全局模型名 = args.目标模型
    样本路径 = args.样本路径
    短名 = 全局模型名.replace("-Instruct", "")
    if 短名 != "Qwen2.5-1.5B":
        日志路径 = os.path.join(输出目录, f"LLM_Judge_锚点_{短名}.log")
        结果路径 = os.path.join(输出目录, f"LLM_Judge_锚点_{短名}.json")

    if args.汇总:
        with open(结果路径, encoding="utf-8") as f:
            全部汇总 = json.load(f)["模式汇总"]
        打印对比判定(全部汇总)
        return 全部汇总

    if not (args.只裁判 or args.只生成):
        if os.path.exists(日志路径):
            os.remove(日志路径)
    记录日志(f"=== LLM-as-Judge P4 锚点评测 模型={全局模型名} 模式={模式列表} 样本={args.样本} "
              f"β={args.β} T_anchor={args.T_anchor} 稀疏阈值={args.稀疏阈值} ===")
    记录日志(f"P4 降级情况：潮汐感知器/cnsenti 可用={_潮汐可用}，导入错误={_潮汐导入错误 or '无'}")
    记录日志("P4 单模式纯净对比：默认不启用 P3 人设/身份拦截；变体用 --人设 / --自选N")
    with open(样本路径, encoding="utf-8") as f:
        样本 = json.load(f)["样本"]
    随机样本 = 样本[:args.样本]
    记录日志(f"样本总数 {len(样本)}，使用 {len(随机样本)}")

    全部汇总 = {}
    各模式AI回复 = {}
    # 缓存前缀含配置标记：样本集名 + 变体标记，避免串用旧缓存
    _样本名 = os.path.splitext(os.path.basename(args.样本路径))[0]
    cfg标记 = f"{_样本名}_S{args.生成长度}"
    if args.人设:
        cfg标记 += "_人设"
    if args.自选N > 1:
        cfg标记 += f"_选{args.自选N}"
    缓存前缀 = os.path.join(输出目录, f"生成_锚点_{短名}_{args.样本}_{cfg标记}")
    for 模式 in 模式列表:
        # ── 生成阶段（可独立运行 --只生成，缓存到文件供裁判阶段复用）──
        缓存文件 = f"{缓存前缀}_{模式}.json"
        if not args.只裁判 and not os.path.exists(缓存文件):
            AI回复列表 = 生成AI回复(模式, args, 随机样本)
            with open(缓存文件, "w", encoding="utf-8") as f:
                json.dump({"AI回复": AI回复列表, "user": [r["user"] for r in 随机样本],
                           "girl": [r["girl"] for r in 随机样本]}, f, ensure_ascii=False)
        if args.只生成:
            continue
        if not os.path.exists(缓存文件):
            continue
        with open(缓存文件, encoding="utf-8") as f:
            _缓存 = json.load(f)
        AI回复列表 = _缓存["AI回复"]
        各模式AI回复[模式] = AI回复列表
        # ── 裁判阶段（独立进程加载 7B 裁判，避免同进程 OOM）──
        汇总 = 裁判盲评(模式, args, 随机样本, AI回复列表)
        全部汇总[模式] = 汇总

    # 分进程裁判时增量合并（多次 --只裁判 各自写入）
    if os.path.exists(结果路径) and args.只裁判:
        try:
            with open(结果路径, encoding="utf-8") as f:
                _旧 = json.load(f)
            for _k, _v in _旧.get("模式汇总", {}).items():
                全部汇总.setdefault(_k, _v)
        except Exception:  # noqa: BLE001
            pass

    打印对比判定(全部汇总)

    # --只生成 只写缓存，不覆盖结果文件
    if args.只生成:
        return 全部汇总
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump({"配置": {"目标模型": 全局模型名, "样本路径": 样本路径, "样本数": args.样本,
                           "β": args.β, "T_anchor": args.T_anchor, "稀疏阈值": args.稀疏阈值,
                           "K": 6, "人设": args.人设, "自选N": args.自选N, "种子": args.种子},
                   "模式汇总": 全部汇总, "AI回复": 各模式AI回复}, f, ensure_ascii=False, indent=2)
    记录日志(f"结果已保存 -> {结果路径}")
    return 全部汇总


if __name__ == "__main__":
    主程序()
