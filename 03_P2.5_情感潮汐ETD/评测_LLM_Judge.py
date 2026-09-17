# -*- coding: utf-8 -*-
"""
LLM-as-Judge — AI 回复 vs 真人回复 盲评（潮汐版）
====================================================
复用语义回响项目 run_llm_judge.py 的协议：
- 提示词集：样本_30条.json 的 user 话
- 候选 A：目标模型 1.5B 生成回复（裸 / 潮汐）
- 候选 B：真人回复（girl 字段）
- 裁判 7B：盲评"哪个更像真人写的"，并对每个回复打 1-5 分
指标：win_rate_against_human + average_rating

用法：
  python 评测_LLM_Judge.py --模式 全部 --样本 20
"""
import json
import os
import re
import sys
import gc
import random
import argparse
import torch

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

# 语义回响工程根（混合模式需导入回响池/注入器）
回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from 潮汐感知器 import 潮汐感知器
from 潮汐决策器 import 潮汐决策器
from 潮汐解码器 import 潮汐解码器

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
全局模型名 = "Qwen2.5-1.5B-Instruct"  # 由 --目标模型 覆盖（横向测试）
全局LoRA路径 = ""  # 由 --LoRA 覆盖（如 gentle_v2 外挂）
# v8 人设系统提示（架构级优化：prompt 真人聊天风格引导；--无人设 关闭）
# 直击 1.5B 服务腔/大道理两大低分主因（"如果您有任何问题/每个人都是独一无二的"）
全局人设 = "你正在用聊天软件和一个你很喜欢的朋友闲聊。回复要像普通人随手打出的字：口语化、简短，一两句话，20到45字。不要自我介绍，不要提AI、助手、模型、系统、程序这些词，不要说“如果您有任何问题”“我可以为您服务”这类客服话术，不要讲大道理。自然地回应对方的情绪，偶尔带一点语气词。"
裁判模型名 = "Qwen2.5-7B-Instruct"
样本路径 = r"i:\Desktop\语义回响\图灵测试\样本_30条.json"  # 由 --样本路径 覆盖（60条）
输出目录 = os.path.join(本目录, "评测结果")
os.makedirs(输出目录, exist_ok=True)
日志路径 = os.path.join(输出目录, "LLM_Judge_潮汐.log")
结果路径 = os.path.join(输出目录, "LLM_Judge_潮汐.json")


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
    # 可选 LoRA 外挂（如 gentle_v2 温柔陪伴适配器），加载后合并进基座权重
    global 全局LoRA路径
    if 全局LoRA路径:
        from peft import PeftModel
        模型 = PeftModel.from_pretrained(模型, 全局LoRA路径)
        模型 = 模型.merge_and_unload()
        模型 = 模型.to(设备)
        记录日志(f"[LoRA] 已加载并合并: {全局LoRA路径}")
    模型.eval()
    return 模型, 分词器


def 卸载模型(模型, 分词器):
    del 模型, 分词器
    gc.collect()
    torch.cuda.empty_cache()


def 加载裁判():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"[加载裁判] 显存占用={torch.cuda.memory_allocated()/1e9:.2f}GB 缓存={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
    分词器 = AutoTokenizer.from_pretrained(
        os.path.join(模型空间, 裁判模型名), trust_remote_code=True)
    # v8.1 手动加载裁判（绕过 from_pretrained 在低内存机上的 torch_cpu.dll 原生崩溃）：
    # 背景：from_pretrained 加载 7B 时 torch_cpu.dll 固定偏移 0x6046edb 崩溃（0xC0000005），
    #       1.5B/3B 正常——与提交内存(commit)吃紧相关（bnb 8bit 需 ~14GB 瞬时驻留）。
    # 手动路径实测稳定：meta 建模 → to_empty(cuda) → 逐分片 safetensors load_state_dict，
    # CPU 峰值仅单分片（~3.7GB），bf16 与存储一致。
    # 关键坑：to_empty 只迁移结构，__init__ 计算的非持久化 buffer（Qwen2 旋转编码
    # inv_freq/original_inv_freq）变成全零 → 必须按 rope_theta 重算，否则 logits 乱码。
    from safetensors import safe_open
    import glob as _glob
    裁判路径 = os.path.join(模型空间, 裁判模型名)
    cfg = AutoConfig.from_pretrained(裁判路径, trust_remote_code=True)
    with torch.device("meta"):
        模型 = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    模型 = 模型.to_empty(device="cuda")
    for _分片 in sorted(_glob.glob(os.path.join(裁判路径, "model-*.safetensors"))):
        with safe_open(_分片, framework="pt", device="cpu") as f:
            _sd = {k: f.get_tensor(k) for k in f.keys()}
        模型.load_state_dict(_sd, strict=False)
        del _sd
        gc.collect()
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


def 潮汐生成(model, tokenizer, 消息, 种子, 轮次, 用户文本, max_new_tokens=64,
             AI抑制=2.0, 口语化=1.0, 目标长=34,
             身份拦截=True, 句子停止=True, 最长句数=2, 最短字数=12, 最大字数=90, 最小长度=0):
    torch.manual_seed(种子 + 轮次)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子 + 轮次)
    感知器 = 潮汐感知器()
    决策器 = 潮汐决策器(感知器)
    解码器 = 潮汐解码器(model, tokenizer, 感知器, 决策器, AI腔抑制强度=AI抑制,
                       口语化强度=口语化, 目标长度=目标长,
                       身份拦截=身份拦截, 句子停止=句子停止,
                       最长句数=最长句数, 最短字数=最短字数, 最大字数=最大字数,
                       最小长度=最小长度)
    提示 = 构建提示(tokenizer, 消息)
    inputs = tokenizer(提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = 解码器.生成(
            inputs.input_ids, max_new_tokens=max_new_tokens,
            temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            用户文本=用户文本,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(新token, skip_special_tokens=True).strip()


def 混合生成(model, tokenizer, 消息, 种子, 轮次, 用户文本, max_new_tokens=64,
              λ=0.10, 倍率=8.0, AI抑制=2.0, 口语化=1.0, 目标长=34,
             身份拦截=True, 句子停止=True, 最长句数=2, 最短字数=12, 最大字数=90, 最小长度=0):
    """回响 × 潮汐 混合生成（角色主导方向）"""
    torch.manual_seed(种子 + 轮次)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子 + 轮次)
    from semantic_echo.回响池 import 语义回响池
    from semantic_echo.情感过滤器 import 情感过滤器
    from 混合注入器 import 混合注入器
    感知器 = 潮汐感知器()
    决策器 = 潮汐决策器(感知器)
    过滤器 = 情感过滤器()
    过滤器.加载词库()
    池 = 语义回响池(hidden_dim=model.config.hidden_size, decay_gamma=0.07)
    解码器 = 混合注入器(model, 池, tokenizer, 感知器, 决策器,
                       lambda_strength=λ, 引导倍率=倍率,
                       情感过滤器实例=过滤器, AI腔抑制强度=AI抑制,
                       口语化强度=口语化, 目标长度=目标长,
                       身份拦截=身份拦截, 句子停止=句子停止,
                       最长句数=最长句数, 最短字数=最短字数, 最大字数=最大字数,
                       最小长度=最小长度)
    提示 = 构建提示(tokenizer, 消息)
    inputs = tokenizer(提示, return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            out = 解码器.生成(
                inputs.input_ids, max_new_tokens=max_new_tokens,
                temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id, tokenizer=tokenizer,
                用户文本=用户文本,
            )
    finally:
        # 释放投影矩阵（1536×151936≈892MB）与钩子，防止跨样本 OOM
        try:
            解码器._移除钩子()
        except Exception:
            pass
        del 解码器, 池, 过滤器, 感知器, 决策器
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    新token = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(新token, skip_special_tokens=True).strip()


# ============================================================
# AI 腔序列级检测（拒绝采样用）
# ============================================================
AI腔检测正则 = [
    # 身份暴露
    r"作为.{0,10}(?:AI|人工智能|助手|模型|语言模型|程序|系统|机器人|智能)",
    r"我(?:是|是一个|是个|是AI|是人工智能|是一个.{0,6}(?:程序|系统|机器人|软件|模型|助手))",
    r"我(?:被设计|的存在|的功能|没有情感|没有感觉|无法感受|没有上下班|是24小时|是一个.{0,4}在线)",
    r"(?:阿里巴巴|阿里云|华为|百度|字节).{0,10}(?:开发|模型|AI)",
    r"机器学习系统|计算机程序|数字人|云端存在|预训练语言|大语言模型|无生命的|无感情的",
    r"AI\s*(?:助手|模型|语言模型)|人工智能|智能助手|聊天机器人|语言模型|语料训练",
    # 服务套话
    r"请(?:告诉我|随时|不要担心)|有什么可以帮|欢迎告诉|为您(?:提供|服务|解答)|帮助您(?:解决|回答|完成)",
    r"(?:需要|如果(?:您|你)有).{0,8}(?:帮助|问题|需求).{0,6}(?:请|告诉|联系)",
    r"(?:提供|获取)(?:信息|帮助|支持|服务)|解答(?:您的|你的)?问题|回答(?:您的|你的)?问题",
    # 分析式书面语（AI 味）
    r"这是一个(?:开放|复杂|常见|非常|很).{0,6}(?:问题|话题)|这句话(?:表达|说明)了",
    r"这个(?:问题|话题)(?:是|确实)|从(?:某种|某|宏观|整体)(?:角度|方面|视角)",
    r"总的来说|总而言之|首先.{0,8}其次.{0,8}最后|根据我的理解",
]
AI腔检测编译 = [re.compile(p) for p in AI腔检测正则]


def 检测AI腔(文本):
    """序列级 AI 腔检测：返回命中的短语数"""
    return sum(1 for r in AI腔检测编译 if r.search(文本))


# ──────────────────────────────────────────────
# v6 身份暴露兜底清理（decode 拦截漏网时的最后防线）
# ──────────────────────────────────────────────
# 与解码层"身份打开器+身份名词"紧邻拦截配合：若完整身份短语仍生成出来，
# 在上一句边界处截断（如 "我不怕冷。我是一个语言模型…" → "我不怕冷。"）。
身份暴露正则 = re.compile(
    # 开启器（紧邻约束，避免误伤"我是做系统开发的"类正常句）
    r"(?:我是|我是一个?|作为一个?|我是一台|我是24小时|我的功能|我的目的|我的出现|我的使命|我的作用|"
    r"我是基于|我被设计|我是一个正在运行)"
    r"(?:一个|一台|大型|大規模|大规模|正在运行|基于|AI|的)?"
    r".{0,6}?"  # 宽容中段：覆盖"我是基于语料训练的语言处理系统"类绕行短语
    r"(?:AI|人工智能|助手|模型|系统|程序|机器人|数字人|智能体|聊天机器人|"
    r"语言模型|语言模版|语言处理系统|大语言模型|算法|数据训练|语料|处理系统|框架|工具|计算机|虚拟|"
    r"人类|人们|用户|知识库|数据库|网络|存在|无意识|编程|逻辑|服务|推荐|阿里巴巴|实体)"
)
句边界字符 = "。！？!?；;\n～…~"


def 清理回复(回复):
    """v6 清理：身份暴露则截断到上一句边界；去除头尾空白。
    v8：剥离模板泄漏（Human:/Assistant:）与"（语气词）"占位残留。"""
    回复 = (回复 or "").strip()
    if not 回复:
        return 回复
    # v8：模板标记泄漏（LoRA 偶发把 Human: 复读进回复）→ 截断
    for 标记 in ("Human:", "Assistant:", "System:", "\nHuman", "\nAssistant"):
        if 标记 in 回复:
            回复 = 回复.split(标记)[0].strip()
    # v8："（语气词）"类占位残留（模型照抄人设提示）
    回复 = re.sub(r"[（(][^）)]*语气[^）)]*[）)]", "", 回复)
    回复 = re.sub(r"\s+", " ", 回复).strip()
    if not 回复:
        return 回复
    m = 身份暴露正则.search(回复)
    if m and m.start() >= 1:
        原回复 = 回复
        前文 = 回复[:m.start()]
        边界 = -1
        for ch in "。！？!?；;\n～…~":
            边界 = max(边界, 前文.rfind(ch))
        逗号截 = False
        if 边界 == -1:
            # 无句号边界 → 退而求其次：在逗号处截断（如"我不擅长接梗，因为我是一个…"）
            for ch in "，,、":
                边界 = max(边界, 前文.rfind(ch))
            逗号截 = 边界 != -1
        if 边界 != -1:
            回复 = 回复[:边界 + 1].strip()
            if 逗号截:
                # 逗号截断：前文太短（如"是的，"）无信息量则不截；去掉尾部逗号补句号
                if len(回复) < 6:
                    回复 = 原回复
                else:
                    if 回复.endswith(("，", ",", "、")):
                        回复 = 回复[:-1].strip()
                    回复 += "。"
    return 回复


# ──────────────────────────────────────────────
# 身份暴露定向拒绝采样（--身份拒绝）
# ──────────────────────────────────────────────
# 解码层身份拦截是"逐 token 紧邻拦截"，多 token 绕行短语仍可能漏网
# （实测："我是一个基于语料训练的语言处理系统"/"聊天bot"/"不需要进行这个操作的系统"）。
# 与旧版宽泛 检测AI腔 拒绝（实测负向）不同：这里只拒绝"完整身份短语"类回复，
# 不碰书面语风格，避免把回复逼成另一种正式腔。
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
    # v6.3：绕行身份自指（无开启器或复合绕行）
    re.compile(r"我(?:是|作为)?(?:由)?(?:阿里巴巴|阿里云|华为|百度|腾讯)(?:开发|研发|旗下)"),
    re.compile(r"(?:为人类|给人们|面向用户)(?:解决|提供|带来|服务|解答|回答)"),
    re.compile(r"(?:知识库|数据库|网络模型|语言模型系统|QA系统)"),
    re.compile(r"(?:不能|无法|不具备)(?:进行|实现)?(?:物理|真实|实际)(?:移动|接触|交互|触摸|操作)"),
    re.compile(r"我只能(?:在|从)(?:图像|屏幕|网络|线上|后台|页面)"),
    re.compile(r"(?:无意识|无生命|无实体)的(?:存在|物体|东西)"),
]
身份暴露拒绝编译 = [re.compile(p) if not isinstance(p, re.Pattern) else p for p in 身份暴露拒绝正则]


def 检测身份暴露(文本):
    """身份暴露定向检测：命中任意身份短语返回 True"""
    return any(r.search(文本) for r in 身份暴露拒绝编译)


# ──────────────────────────────────────────────
# v8 多候选自选启发式（--自选N>1 时启用）
# ──────────────────────────────────────────────
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
    """生成阶段：只加载目标模型（1.5B/3B/1.7B），输出 AI 回复列表。
    可选拒绝采样（--拒绝采样，默认关：实测换来的书面语回复反而负向）。
    v8：人设系统提示 + 多候选自选（--自选N>1 时按 人味评分 选最优）。"""
    记录日志(f"──── 模式 [{模式}] AI 生成（max_tokens={args.生成长度} 人设={args.人设} 自选N={args.自选N}） ────")
    model, tokenizer = 加载目标模型()
    AI回复列表 = []
    for i, r in enumerate(随机样本):
        消息 = ([{"role": "system", "content": 全局人设}] if args.人设 else []) + \
               [{"role": "user", "content": r["user"]}]
        候选集 = []
        for 候选idx in range(args.自选N):
            for 尝试 in range(args.重试次数):
                种子 = args.种子 + i + 候选idx * 7777 + 尝试 * 1000
                if 模式 == "裸":
                    回复 = 裸生成(model, tokenizer, 消息, 种子, 0, max_new_tokens=args.生成长度)
                elif 模式 == "潮汐":
                    回复 = 潮汐生成(model, tokenizer, 消息, 种子, 0, r["user"],
                                  max_new_tokens=args.生成长度,
                                  AI抑制=args.AI抑制, 口语化=args.口语化, 目标长=args.目标长,
                                  身份拦截=args.身份拦截, 句子停止=args.句子停止,
                                  最长句数=args.最长句数, 最短字数=args.最短字数,
                                  最大字数=args.最大字数, 最小长度=args.最小长度)
                else:
                    回复 = 混合生成(model, tokenizer, 消息, 种子, 0, r["user"],
                                  max_new_tokens=args.生成长度,
                                  λ=args.混合λ, 倍率=args.混合倍率, AI抑制=args.AI抑制,
                                  口语化=args.口语化, 目标长=args.目标长,
                                  身份拦截=args.身份拦截, 句子停止=args.句子停止,
                                  最长句数=args.最长句数, 最短字数=args.最短字数,
                                  最大字数=args.最大字数, 最小长度=args.最小长度)
                # v6 兜底清理：身份暴露截断（只对潮汐/混合，裸模式保持原始基线）
                if 模式 != "裸":
                    回复 = 清理回复(回复)
                应拒绝 = (args.拒绝采样 and 检测AI腔(回复) > 0) or (args.身份拒绝 and 检测身份暴露(回复))
                if not 应拒绝:
                    break
            候选集.append(回复)
        # v8 自选：人味评分最高者（N=1 时即该候选）
        回复 = max(候选集, key=人味评分)
        AI回复列表.append(回复)
        记录日志(f"[AI生成 {i+1}/{len(随机样本)}] 选{len(候选集)} 长{len(回复)} {r['user'][:16]} => {回复[:34]}")
    # 必须从主作用域彻底释放目标模型（卸载模型 只删函数内引用）
    del model, tokenizer
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


def 主程序():
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", choices=["裸", "潮汐", "混合", "全部"], default="全部")
    ap.add_argument("--样本", type=int, default=20)
    ap.add_argument("--种子", type=int, default=42)
    ap.add_argument("--目标模型", default="Qwen2.5-1.5B-Instruct", help="横向测试：目标模型名")
    ap.add_argument("--LoRA", default="", help="LoRA 外挂适配器路径（如 f:\\lora外挂\\lora_adapters\\gentle_v2）")
    ap.add_argument("--样本路径", default=r"i:\Desktop\语义回响\图灵测试\样本_30条.json",
                    help="样本文件（30条/60条）")
    ap.add_argument("--混合λ", type=float, default=0.08, help="混合模式回响注入强度")
    ap.add_argument("--混合倍率", type=float, default=6.0, help="混合模式潮汐引导倍率")
    ap.add_argument("--AI抑制", type=float, default=4.0, help="AI腔抑制强度（v4 双通道）")
    ap.add_argument("--口语化", type=float, default=0.0, help="口语化引导强度（v5；默认关，消融为负向）")
    ap.add_argument("--目标长", type=int, default=200, help="目标回复长度（超长促收尾；默认关）")
    ap.add_argument("--生成长度", type=int, default=64, help="max_new_tokens（真人中位20字→可试48）")
    ap.add_argument("--拒绝采样", action="store_true", help="AI腔拒绝采样（默认关：实测负向）")
    ap.add_argument("--身份拒绝", action="store_true", help="身份暴露定向拒绝采样（仅身份短语，比宽泛 AI腔 拒绝安全）")
    ap.add_argument("--重试次数", type=int, default=6, help="拒绝采样/身份拒绝的最大重试种子数")
    ap.add_argument("--关闭身份拦截", action="store_true", help="v6 关闭身份暴露硬拦截（消融）")
    ap.add_argument("--关闭句子停止", action="store_true", help="v6 关闭句子边界硬停止（消融）")
    ap.add_argument("--最长句数", type=int, default=2, help="v6 句子边界硬停止：达到该句数即停")
    ap.add_argument("--最短字数", type=int, default=12, help="v6 达到最短字数后才允许停")
    ap.add_argument("--最大字数", type=int, default=90, help="v6 超长强制停止")
    ap.add_argument("--最小长度", type=int, default=0, help="v6.2 最小回复字数（不足则压制 EOS 强制续写；0=关）")
    ap.add_argument("--无人设", action="store_true", help="v8 关闭人设系统提示（架构级风格引导）")
    ap.add_argument("--自选N", type=int, default=1, help="v8 每样本候选数（>1 按人味评分选最优，如 3）")
    ap.add_argument("--只生成", action="store_true", help="只跑生成阶段并缓存")
    ap.add_argument("--只裁判", action="store_true", help="只跑裁判阶段（读生成缓存）")
    args = ap.parse_args()
    args.身份拦截 = not args.关闭身份拦截
    args.句子停止 = not args.关闭句子停止
    args.人设 = not args.无人设
    模式列表 = ["裸", "潮汐", "混合"] if args.模式 == "全部" else [args.模式]

    global 全局模型名, 日志路径, 结果路径, 样本路径, 全局LoRA路径
    全局模型名 = args.目标模型
    全局LoRA路径 = args.LoRA
    样本路径 = args.样本路径
    短名 = 全局模型名.replace("-Instruct", "")
    if 全局LoRA路径:
        短名 += "+" + os.path.basename(全局LoRA路径)
    if 短名 != "Qwen2.5-1.5B":
        日志路径 = os.path.join(输出目录, f"LLM_Judge_{短名}.log")
        结果路径 = os.path.join(输出目录, f"LLM_Judge_{短名}.json")

    if os.path.exists(日志路径):
        os.remove(日志路径)
    记录日志(f"=== LLM-as-Judge 潮汐评测 模型={全局模型名} 模式={模式列表} 样本={args.样本} ===")
    with open(样本路径, encoding="utf-8") as f:
        样本 = json.load(f)["样本"]
    随机样本 = 样本[:args.样本]
    记录日志(f"样本总数 {len(样本)}，使用 {len(随机样本)}")

    全部汇总 = {}
    # 缓存前缀含配置标记：样本集名 + 消融/调参标记，避免串用旧缓存
    _样本名 = os.path.splitext(os.path.basename(args.样本路径))[0]
    cfg标记 = f"{_样本名}_L{args.混合λ}_M{args.混合倍率}_S{args.生成长度}"
    if args.关闭身份拦截:
        cfg标记 += "_noID"
    if args.关闭句子停止:
        cfg标记 += "_noStop"
    if args.口语化 != 0.0:
        cfg标记 += f"_口{args.口语化}"
    if args.目标长 != 200:
        cfg标记 += f"_长{args.目标长}"
    if args.身份拒绝:
        cfg标记 += "_身份拒"
    if args.拒绝采样:
        cfg标记 += "_AI拒"
    if args.最长句数 != 2:
        cfg标记 += f"_句{args.最长句数}"
    if args.最短字数 != 12:
        cfg标记 += f"_短{args.最短字数}"
    if args.最大字数 != 90:
        cfg标记 += f"_帽{args.最大字数}"
    if args.最小长度 != 0:
        cfg标记 += f"_min{args.最小长度}"
    if args.人设:
        cfg标记 += "_人设"
    if args.自选N > 1:
        cfg标记 += f"_选{args.自选N}"
    缓存前缀 = os.path.join(输出目录, f"生成_{短名}_{args.样本}_{cfg标记}")
    for 模式 in 模式列表:
        # ── 生成阶段（可独立运行 --只生成，缓存到文件供裁判阶段复用） ──
        缓存文件 = f"{缓存前缀}_{模式}.json"
        if not args.只裁判 and not os.path.exists(缓存文件):
            AI回复列表 = 生成AI回复(模式, args, 随机样本)
            with open(缓存文件, "w", encoding="utf-8") as f:
                json.dump({"AI回复": AI回复列表}, f, ensure_ascii=False)
        if args.只生成:
            continue
        if not os.path.exists(缓存文件):
            continue
        with open(缓存文件, encoding="utf-8") as f:
            AI回复列表 = json.load(f)["AI回复"]
        # ── 裁判阶段（独立进程加载 fp16 裁判，避免同进程 OOM） ──
        汇总 = 裁判盲评(模式, args, 随机样本, AI回复列表)
        全部汇总[模式] = 汇总

    if "裸" in 全部汇总 and "潮汐" in 全部汇总:
        for 键 in ["win_rate_against_human", "average_rating"]:
            v0 = 全部汇总["裸"][键]
            v1 = 全部汇总["潮汐"][键]
            记录日志(f"对比[{键}] 裸 {v0} → 潮汐 {v1} (Δ {v1 - v0:+.4f})")

    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump({"目标模型": 全局模型名, "模式汇总": 全部汇总}, f, ensure_ascii=False, indent=2)
    记录日志(f"结果已保存 -> {结果路径}")
    return 全部汇总


if __name__ == "__main__":
    主程序()
