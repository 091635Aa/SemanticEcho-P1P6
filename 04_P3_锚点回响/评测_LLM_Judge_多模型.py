# -*- coding: utf-8 -*-
"""
Task7 多模型泛化验证 —— LLM-as-Judge 盲评（裸 vs P4 锚点，Qwen2.5-3B / Qwen3-1.7B / gemma-2-2b）
====================================================================================================
基于 P4 `评测_LLM_Judge_锚点.py` 改造（同一裁判协议：配对盲评 AB 正反各一次 + 1-5 分评分，
win_rate_against_human / average_rating 口径完全一致），核心差异：

① 目标模型可配（--目标模型），模型路径查表；每模型独立锚点库 + 独立打分表缓存；
② β / T_anchor 直接采用 `目标决策器.自动适配(model, 量化类型)` 的返回值（含来源说明），
   不沿用 Task5 定标 β=0.8（验证 P2 扫描表/公式/兜底路径在多模型上的效果）；
③ Qwen3 系列：enable_thinking=False 直接传 kwargs（chat_template_kwargs 无效，P3 已验证），
   生成后剥离残留 <think>...</think> 块（P3 图灵测试框架同款正则）；
④ 裸模式 = 锚点解码器 β=0（不注入），与锚点模式同一采样循环/句子停止/统计口径，
   保证"唯一变量是锚点注入"；生成阶段记录 熵/重复率/情感命中率/触发兜底次数（健康度）；
⑤ 生成与裁判分离进程（--只生成 / --只裁判 两阶段 + 缓存），裁判沿用 P3 手动加载方案
   （meta → to_empty → safetensors 逐张量 → 重算 rope inv_freq）；
⑥ --β倍率：锚点模式坍缩/退化时降 β（×0.5）重跑锚点生成（默认 1.0）。

用法：
  F:\打标\.venv\Scripts\python.exe 评测_LLM_Judge_多模型.py --目标模型 Qwen2.5-3B-Instruct --只生成
  F:\打标\.venv\Scripts\python.exe 评测_LLM_Judge_多模型.py --目标模型 Qwen2.5-3B-Instruct --只裁判
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 离线加载本地模型，必须在 transformers import 前

import argparse
import gc
import json
import re
import sys
import time
import datetime
import torch

# 锚点回响工作目录（锚点库 / 目标决策器 / 锚点解码器 所在）
本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器, 自动适配, _潮汐可用, _潮汐导入错误
from 锚点解码器 import 锚点解码器

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
# 多模型路径表（Task7：Qwen2.5-3B / Qwen3-1.7B / gemma-2-2b，gemma 在 l:\模型空间）
模型路径表 = {
    "Qwen2.5-1.5B-Instruct": os.path.join(模型空间, "Qwen2.5-1.5B-Instruct"),
    "Qwen2.5-3B-Instruct": os.path.join(模型空间, "Qwen2.5-3B-Instruct"),
    "Qwen3-1.7B-Instruct": os.path.join(模型空间, "Qwen3-1.7B-Instruct"),
    "gemma-2-2b-it": r"l:\模型空间\gemma-2-2b-it",
}
裁判模型名 = "Qwen2.5-7B-Instruct"
裁判路径 = os.path.join(模型空间, 裁判模型名)
样本路径 = r"i:\Desktop\语义回响\图灵测试\样本_30条.json"  # 由 --样本路径 覆盖；三模型必须同一份
输出目录 = os.path.join(本目录, "评测结果")
os.makedirs(输出目录, exist_ok=True)

全局模型名 = "Qwen2.5-3B-Instruct"
日志路径 = os.path.join(输出目录, "LLM_Judge_多模型.log")


# Qwen3 思考链剥离（P3 图灵测试框架同款：剥离 <think>...</think> 及未闭合残留）
_think剥离正则 = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def 清洗回复(文本):
    """剥离 Qwen3 <think> 块（enable_thinking=False 兜底）并去除首尾空白"""
    文本 = _think剥离正则.sub("", 文本 or "")
    return 文本.strip()


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
    _路径 = 模型路径表[全局模型名]
    # gemma 原生 bf16；Qwen 系列 fp16（3B/1.7B 在 16GB 无压力）
    _dtype = torch.bfloat16 if 全局模型名 == "gemma-2-2b-it" else torch.float16
    分词器 = AutoTokenizer.from_pretrained(_路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        _路径, torch_dtype=_dtype if 设备 == "cuda" else torch.float32,
        trust_remote_code=True).to(设备)
    模型.eval()
    return 模型, 分词器


def 卸载模型(模型, 分词器):
    del 模型, 分词器
    gc.collect()
    torch.cuda.empty_cache()


def 加载裁判():
    """手动加载裁判（P3/P4 已验证方案：meta 建模 → to_empty(cuda) →
    逐分片 safetensors 逐张量 load_state_dict → 重算 rope inv_freq），
    避免 from_pretrained 7B 在 16GB 内存机上的 torch_cpu.dll 原生崩溃。"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"[加载裁判] 显存占用={torch.cuda.memory_allocated()/1e9:.2f}GB 缓存={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
    分词器 = AutoTokenizer.from_pretrained(裁判路径, trust_remote_code=True)
    from safetensors import safe_open
    import glob as _glob
    cfg = AutoConfig.from_pretrained(裁判路径, trust_remote_code=True)
    with torch.device("meta"):
        模型 = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    模型 = 模型.to_empty(device="cuda")
    # 逐张量加载分片（P3 方案：把 commit 峰值压到单张量 ~0.5GB，避免"页面文件太小(1455)"）
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
# 生成（裸 = 解码器 β=0 不注入；锚点 = 解码器 β=自动适配值；同一统计口径）
# ============================================================
class 多模型锚点会话:
    """P4 会话：β 来自 自动适配()；--β倍率 用于坍缩重跑（β×0.5）；
    裸模式传入 β=0（不注入，等价纯采样，统计照常）。"""

    def __init__(self, model, tokenizer, 库, 适配, β, T_anchor, 生成长度=64):
        self.model = model
        self.tokenizer = tokenizer
        self.库 = 库
        self.适配 = 适配
        self.目标决策器 = 目标决策器(锚点库=库, β基=β)
        self.解码器 = 锚点解码器(
            model, tokenizer, 库, self.目标决策器,
            β=β, T_anchor=T_anchor, 稀疏阈值=0.0,
            温度=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
            打分表缓存路径=os.path.join(输出目录, f"锚点表_多模型_{全局模型名}.pt"),
        )
        self.生成长度 = 生成长度

    def 重置(self):
        try:
            self.目标决策器.感知器.重置轨迹()
        except Exception:  # noqa: BLE001
            pass

    def 生成(self, 消息, 种子, 用户文本):
        torch.manual_seed(种子)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(种子)
        提示 = 构建提示(self.tokenizer, 消息)
        inputs = self.tokenizer(提示, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            ids, 统计 = self.解码器.生成(
                inputs.input_ids, max_new_tokens=self.生成长度,
                eos_token_id=self.tokenizer.eos_token_id, tokenizer=self.tokenizer,
                用户文本=用户文本,
            )
        新token = ids[0, inputs.input_ids.shape[1]:]
        回复 = self.tokenizer.decode(新token, skip_special_tokens=True).strip()
        return 清洗回复(回复), 统计


def 文本级情感命中率(回复, 库):
    """文本级情感种子词命中率（每维词集子串命中计数 / 回复长度），
    作为 token 级命中率（受多 token 中文分词影响）的补充口径。"""
    if not 回复:
        return 0.0
    命中 = 0
    for 词列表 in 库.词集.values():
        for 词 in 词列表:
            if 词 and 词 in 回复:
                命中 += 1
    return round(命中 / max(len(回复), 1), 4)


def 汇总健康度(统计列表, 回复列表, 库):
    """生成健康度汇总：熵均值 / 2-gram 重复率均值 / 情感命中率均值 / 触发兜底均值"""
    键 = ["平均熵", "重复率", "情感命中率"]
    汇总 = {}
    for k in 键:
        值列表 = [s.get(k, 0.0) for s in 统计列表]
        汇总[k] = round(sum(值列表) / max(len(值列表), 1), 4)
    汇总["触发兜底次数均值"] = round(
        sum(s.get("触发兜底次数", 0) for s in 统计列表) / max(len(统计列表), 1), 4)
    # 文本级情感命中（补充口径）
    文本命中 = [文本级情感命中率(r, 库) for r in 回复列表]
    汇总["文本级情感命中率"] = round(sum(文本命中) / max(len(文本命中), 1), 4)
    # 长度统计
    长度 = [len(r) for r in 回复列表]
    汇总["平均长度"] = round(sum(长度) / max(len(长度), 1), 2)
    汇总["空回复数"] = sum(1 for r in 回复列表 if not r)
    return 汇总


def 生成AI回复(模式, args, 随机样本, 适配):
    """生成阶段：只加载目标模型（3B/1.7B/2B），输出 AI 回复 + 健康度统计。
    裸 = β×倍率后仍为 0（不注入）；锚点 = 适配β × args.β倍率。"""
    记录日志(f"──── 模型[{全局模型名}] 模式 [{模式}] AI 生成（max_tokens={args.生成长度} β倍率={args.β倍率}） ────")
    model, tokenizer = 加载目标模型()
    库 = 锚点库(model, tokenizer)
    基线 = 库.记录只读基线()
    库.构建()
    S = 库.预计算打分表(缓存路径=os.path.join(输出目录, f"锚点表_多模型_{全局模型名}.pt"))
    只读 = 库.验证只读(基线)
    if not (只读["sum一致"] and 只读["指针一致"]):
        记录日志("[锚点库] 警告：只读校验失败！")
    记录日志(f"[锚点库] 维度={库.维度名()} 打分表={list(S.shape)} {S.dtype} 只读校验={只读}")

    β = (适配["β"] * args.β倍率) if 模式 == "锚点" else 0.0
    T_anchor = 适配["T_anchor"]
    记录日志(f"[{模式}] β={β} T_anchor={T_anchor}（适配来源：{适配.get('来源', '')}）")
    会话 = 多模型锚点会话(model, tokenizer, 库, 适配, β=β, T_anchor=T_anchor,
                          生成长度=args.生成长度)
    AI回复列表 = []
    统计列表 = []
    for i, r in enumerate(随机样本):
        消息 = [{"role": "user", "content": r["user"]}]
        种子 = args.种子 + i
        会话.重置()
        回复, 统计 = 会话.生成(消息, 种子, r["user"])
        AI回复列表.append(回复)
        统计列表.append(统计)
        记录日志(f"[AI生成 {i+1}/{len(随机样本)}] 长{len(回复)} 熵{统计['平均熵']} "
                  f"重{统计['重复率']} 情{统计['情感命中率']} 兜{统计['触发兜底次数']} "
                  f"{r['user'][:12]} => {回复[:30]}")
    # 彻底释放目标模型
    del model, tokenizer, 会话
    gc.collect()
    torch.cuda.empty_cache()
    健康度 = 汇总健康度(统计列表, AI回复列表, 库)
    记录日志(f"[{模式}] 生成健康度（无坍缩检查）: {json.dumps(健康度, ensure_ascii=False)}")
    return AI回复列表, 健康度


def 裁判盲评(模式, args, 随机样本, AI回复列表, 起始序号=0):
    """裁判阶段：独立进程加载 fp16 裁判，对 AI 回复 vs 真人回复盲评。
    返回 (汇总, 已完成条数)。"""
    记录日志(f"──── 模型[{全局模型名}] 模式 [{模式}] 裁判盲评 ────")
    裁判模型, 裁判分词器 = 加载裁判()
    配对胜数 = 0
    配对总数 = 0
    评分列表 = []
    for i in range(起始序号, len(随机样本)):
        r = 随机样本[i]
        用户, 真人 = r["user"], r["girl"]
        ai回复 = AI回复列表[i]
        # 配对（正反各一次，AB 平衡）
        输出A = 裁判生成(裁判模型, 裁判分词器, [{"role": "user", "content": 裁判配对提示.format(
            user=用户, 回复A=ai回复, 回复B=真人)}])
        输出B = 裁判生成(裁判模型, 裁判分词器, [{"role": "user", "content": 裁判配对提示.format(
            user=用户, 回复A=真人, 回复B=ai回复)}])
        选择A, 选择B = 解析配对(输出A), 解析配对(输出B)
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
        # 每 5 条增量写盘（进程被停/超时后保留部分结果）
        if (i + 1) % 5 == 0 or i == len(随机样本) - 1:
            _partial = _汇总(模式, 配对胜数, 配对总数, 评分列表)
            _partial["已完成条数"] = i + 1
            _partial["已完成条数总计"] = len(随机样本)
            _写部分结果(模式, _partial)
    del 裁判模型, 裁判分词器
    gc.collect()
    torch.cuda.empty_cache()
    汇总 = _汇总(模式, 配对胜数, 配对总数, 评分列表)
    汇总["已完成条数"] = len(随机样本)
    return 汇总, len(随机样本)


def _汇总(模式, 配对胜数, 配对总数, 评分列表):
    win_rate = 配对胜数 / 配对总数 if 配对总数 else 0.0
    avg_rating = sum(评分列表) / len(评分列表) if 评分列表 else 0.0
    return {"模式": 模式, "win_rate_against_human": round(win_rate, 4),
            "average_rating": round(avg_rating / 5.0, 4), "配对总数": 配对总数,
            "AI评分均值": round(avg_rating, 2), "评分样本": len(评分列表)}


def _写部分结果(模式, 汇总):
    """把当前模式的部分结果写入 结果路径（--只裁判 增量合并）"""
    try:
        _内容 = {}
        if os.path.exists(结果路径):
            with open(结果路径, encoding="utf-8") as f:
                _内容 = json.load(f)
        模型段 = _内容.setdefault("模型", {}).setdefault(全局模型名, {})
        模型段.setdefault("LLM_Judge", {})[模式] = 汇总
        with open(结果路径, "w", encoding="utf-8") as f:
            json.dump(_内容, f, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001
        记录日志(f"[部分结果写入失败] {e}")


def 打印对比判定(模型段):
    """输出 裸 vs 锚点 对比与判定（Task7 标准：相对提升为正即达成）"""
    if "裸" not in 模型段 or "锚点" not in 模型段:
        记录日志(f"警告：模型段缺少 裸/锚点 任一模（{list(模型段)}），跳过对比判定")
        return None
    for 键 in ["win_rate_against_human", "average_rating"]:
        v0 = 模型段["裸"].get(键, 0.0)
        v1 = 模型段["锚点"].get(键, 0.0)
        相对 = (v1 / v0 - 1.0) if v0 else None
        记录日志(f"对比[{键}] 裸 {v0} → 锚点 {v1} (Δ {v1 - v0:+.4f}"
                  + (f"，相对 {相对:+.2%})" if 相对 is not None else ")"))
    wr0 = 模型段["裸"].get("win_rate_against_human", 0.0)
    wr1 = 模型段["锚点"].get("win_rate_against_human", 0.0)
    达成 = wr1 > wr0
    记录日志(f"判定[锚点 win_rate > 裸 win_rate]：裸 {wr0} → 锚点 {wr1}"
              f"（相对 {(wr1 / wr0 - 1):+.2%}）→ {'✓ 提升为正' if 达成 else '✗ 未提升'}")
    return 达成


def 主程序():
    global 全局模型名, 日志路径, 结果路径, 样本路径
    ap = argparse.ArgumentParser()
    ap.add_argument("--目标模型", default="Qwen2.5-3B-Instruct",
                    choices=list(模型路径表.keys()))
    ap.add_argument("--样本", type=int, default=30)
    ap.add_argument("--种子", type=int, default=42)
    ap.add_argument("--样本路径", default=样本路径)
    ap.add_argument("--生成长度", type=int, default=64)
    ap.add_argument("--β倍率", type=float, default=1.0, help="锚点 β 倍率（坍缩重跑 ×0.5）")
    ap.add_argument("--只生成", action="store_true")
    ap.add_argument("--只裁判", action="store_true")
    ap.add_argument("--模式", choices=["裸", "锚点", "全部"], default="全部",
                    help="只裁判/只生成时限定单个模式（每模式独立进程，时间可控）")
    ap.add_argument("--汇总", action="store_true", help="只读结果文件输出三模型对照判定")
    ap.add_argument("--起始序号", type=int, default=0, help="裁判断点续跑起始条（从 0 开始，部分结果续跑用）")
    args = ap.parse_args()

    全局模型名 = args.目标模型
    样本路径 = args.样本路径
    短名 = 全局模型名.replace("-Instruct", "").replace("gemma-2-2b-it", "gemma-2-2b")
    日志路径 = os.path.join(输出目录, f"LLM_Judge_多模型_{短名}.log")
    结果路径 = os.path.join(输出目录, f"LLM_Judge_多模型_{短名}.json")

    if args.汇总:
        with open(结果路径, encoding="utf-8") as f:
            _全部 = json.load(f)
        for _模型名, _段 in _全部.get("模型", {}).items():
            记录日志(f"\n=== 模型 {_模型名} ===")
            打印对比判定(_段)
        return 0

    if not (args.只裁判 or args.只生成):
        if os.path.exists(日志路径):
            os.remove(日志路径)
    记录日志(f"=== Task7 多模型泛化验证：LLM-as-Judge P4 锚点 模型={全局模型名} "
              f"样本={args.样本} 种子={args.种子} 生成长度={args.生成长度} β倍率={args.β倍率} ===")
    记录日志(f"P4 降级情况：潮汐感知器/cnsenti 可用={_潮汐可用}，导入错误={_潮汐导入错误 or '无'}")
    with open(样本路径, encoding="utf-8") as f:
        样本 = json.load(f)["样本"]
    随机样本 = 样本[:args.样本]
    记录日志(f"样本总数 {len(样本)}，使用 {len(随机样本)}（三模型同一份样本集）")

    # 缓存前缀：样本集名 + 生成长度 + β倍率（避免坍缩重跑串用旧缓存）
    _样本名 = os.path.splitext(os.path.basename(样本路径))[0]
    cfg标记 = f"{_样本名}_S{args.生成长度}"
    if args.β倍率 != 1.0:
        cfg标记 += f"_β{args.β倍率}"
    缓存前缀 = os.path.join(输出目录, f"生成_多模型_{短名}_{args.样本}_{cfg标记}")

    # 适配记录文件：--只生成 时计算并写入；--只裁判 时读取（不加载目标模型）
    适配记录路径 = os.path.join(输出目录, f"多模型_适配_{短名}.json")
    适配 = None
    if args.只裁判 or args.只生成:
        if os.path.exists(适配记录路径) and not args.只生成:
            with open(适配记录路径, encoding="utf-8") as f:
                适配 = json.load(f)
    模式列表 = ["裸", "锚点"] if args.模式 == "全部" else [args.模式]
    各模式AI回复 = {}
    各模式健康度 = {}
    全部汇总 = {}
    for 模式 in 模式列表:
        # ── 生成阶段 ──
        缓存文件 = f"{缓存前缀}_{模式}.json"
        if not args.只裁判 and not os.path.exists(缓存文件):
            if 适配 is None:
                # 首轮：先加载模型取 自动适配，再生成
                _m, _t = 加载目标模型()
                适配 = 自动适配(_m, "fp16" if 全局模型名 != "gemma-2-2b-it" else "bf16")
                with open(适配记录路径, "w", encoding="utf-8") as f:
                    json.dump(适配, f, ensure_ascii=False, indent=2)
                卸载模型(_m, _t)
                记录日志(f"[自动适配] {全局模型名} → {json.dumps(适配, ensure_ascii=False)}")
            AI回复列表, 健康度 = 生成AI回复(模式, args, 随机样本, 适配)
            各模式健康度[模式] = 健康度
            with open(缓存文件, "w", encoding="utf-8") as f:
                json.dump({"AI回复": AI回复列表, "user": [r["user"] for r in 随机样本],
                           "girl": [r["girl"] for r in 随机样本],
                           "健康度": 健康度, "适配": 适配}, f, ensure_ascii=False)
        if args.只生成:
            continue
        if not os.path.exists(缓存文件):
            continue
        with open(缓存文件, encoding="utf-8") as f:
            _缓存 = json.load(f)
        AI回复列表 = _缓存["AI回复"]
        各模式AI回复[模式] = AI回复列表
        各模式健康度[模式] = _缓存.get("健康度", {})
        if 适配 is None:
            适配 = _缓存.get("适配")
        # ── 裁判阶段（独立进程加载 7B 裁判，避免同进程 OOM）──
        汇总, 完成 = 裁判盲评(模式, args, 随机样本, AI回复列表, 起始序号=args.起始序号)
        全部汇总[模式] = 汇总
        if 完成 < len(随机样本):
            记录日志(f"[裁判] 模型 {全局模型名} 模式 {模式} 仅完成 {完成}/{len(随机样本)} 条（限时/中断），记录部分结果后继续")
            break  # 本模型超时/中断：不再评下一个模式

    # 分进程裁判时增量合并
    if os.path.exists(结果路径) and args.只裁判:
        try:
            with open(结果路径, encoding="utf-8") as f:
                _旧 = json.load(f)
            for _k, _v in _旧.get("模型", {}).get(全局模型名, {}).get("LLM_Judge", {}).items():
                全部汇总.setdefault(_k, _v)
        except Exception:  # noqa: BLE001
            pass

    if args.只生成:
        return 0

    # 组装并写最终结果
    模型段 = {"适配": 适配, "生成健康度": 各模式健康度, "LLM_Judge": 全部汇总}
    if "裸" in 全部汇总 and "锚点" in 全部汇总:
        wr0 = 全部汇总["裸"].get("win_rate_against_human", 0.0)
        wr1 = 全部汇总["锚点"].get("win_rate_against_human", 0.0)
        模型段["提升"] = {
            "win_rate_Δ": round(wr1 - wr0, 4),
            "win_rate_相对": f"{((wr1 / wr0 - 1) * 100):+.2f}%" if wr0 else "N/A(裸=0)",
            "达成(>0)": bool(wr1 > wr0),
        }
        模型段["判定"] = 打印对比判定(全部汇总)
    时间戳 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _最终结果 = {"任务": "Task7 多模型泛化验证（P4 锚点回响）",
                "时间戳": 时间戳, "样本路径": 样本路径, "样本数": len(随机样本),
                "种子": args.种子, "生成长度": args.生成长度, "裁判模型": 裁判模型名,
                "模型": {全局模型名: 模型段}, "AI回复": 各模式AI回复}
    最终路径 = os.path.join(输出目录, f"LLM_Judge_多模型_{短名}.json")
    with open(最终路径, "w", encoding="utf-8") as f:
        json.dump(_最终结果, f, ensure_ascii=False, indent=2)
    记录日志(f"结果已保存 -> {最终路径}")
    return 0


if __name__ == "__main__":
    主程序()
