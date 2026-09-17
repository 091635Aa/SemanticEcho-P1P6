# -*- coding: utf-8 -*-
"""
HeartBench — 中文"人味儿"多维评测（P4 锚点回响版）
===================================================
复用 P3 评测_HeartBench.py 的【完全一致协议】（官方 rubric 逐条打分 +
norm_score 归一化 + 四维汇总），唯一区别：生成端从 潮汐解码器 替换为
锚点解码器（P4 单模式：β=0.8, 稀疏阈值=0.0, T_anchor=0.3, K=6，
目标决策器按对话最后一条用户消息自动算 v_target）。

流程：1.5B 对对话历史生成下文回应 → 7B 裁判按 rubric 逐条命中(0/1)
     → 官方 norm_score 归一化(0-100) → 汇总 4 大维度 + overall。

生成与裁判分离进程（--只生成 / --只裁判 两阶段 + 缓存）；裁判加载沿用
P3 已验证的手动加载方案（meta → to_empty → safetensors 逐张量）。

用法：
  F:\打标\.venv\Scripts\python.exe 评测_HeartBench_锚点.py --模式 全部 --样本数 20
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 离线加载本地模型，必须在 transformers import 前

import argparse
import gc
import json
import math
import random
import re
import sys
import time
import torch

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器, _潮汐可用, _潮汐导入错误
from 锚点解码器 import 锚点解码器

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
目标模型名 = "Qwen2.5-1.5B-Instruct"
裁判模型名 = "Qwen2.5-7B-Instruct"
数据路径 = r"i:\Desktop\语义回响\图灵测试\repos\HeartBench\data\question_all.jsonl"
输出目录 = os.path.join(本目录, "评测结果")
os.makedirs(输出目录, exist_ok=True)
日志路径 = os.path.join(输出目录, "HeartBench_锚点.log")
结果路径 = os.path.join(输出目录, "HeartBench_锚点.json")

# 官方评分提示模板（与 P3 评测_HeartBench.py 完全一致）
评分提示模板 = """你是一个专业、严苛的心理学打分批评专家，请根据以上对话历史（作为考题）与待测AI的表现进行打分。
注意事项：
1.考题中的角色B不是待测AI，仅辅助生成考题上文。仅「待测AI回复」中才是你要考察的部分。
2.打分要尽可能严格，加分项只有完全达到才能命中，扣分项如果有一点涉及都要命中。
3.打分时通读对话历史、待测ai恢复和rubric，逐条根据rubric给出命中细节，并计算rubric条数，
最后命中细节数量必须和rubric条数一致，这个原则你需要反复验证，非常重要。
4.不必在意分值多少，不需要计算总分。

对话历史：
{dialogue_history}
待测AI回复：
{response}
rubric：
{rubric}
rubric条数：
{rubric_nums}

直接按照格以下式输出，不要输出markdown或者其他内容。
输出格式：
{{
  'reason'（String）:（说出具体的评分过程，除了json结构中，文本内容里不要用引号防止解析失败，用「」）
  'detail'（array of Integer）:[1,0,1,....](按顺序给出rubric命中情况，命中置1，未命中置0，用英文逗号分割，这里的元素个数一定要与rubric条数保持一致)
}}"""


def calculate_dimension_details(rubric, detail, special_dimension="其他"):
    """官方 score_answers.calculate_dimension_details 的精简等价实现"""
    if not rubric:
        return {"dimension_details": [], "question_score": 0.0, "has_special_hit": False}
    if len(detail) < len(rubric):
        detail = detail + [0] * (len(rubric) - len(detail))
    elif len(detail) > len(rubric):
        detail = detail[:len(rubric)]
    dimension_ranges = {}
    for item in rubric:
        dim = item.get("dimension")
        score = item.get("score", 0)
        if dim is None:
            continue
        s = float(score) if score is not None else 0.0
        if dim not in dimension_ranges:
            dimension_ranges[dim] = {"min": 0.0, "max": 0.0}
        if s < 0:
            dimension_ranges[dim]["min"] += s
        elif s > 0:
            dimension_ranges[dim]["max"] += s
    raw_scores = {dim: 0.0 for dim in dimension_ranges.keys()}
    has_special_hit = False
    for rub, hit in zip(rubric, detail):
        if not hit:
            continue
        dim = rub.get("dimension")
        score = rub.get("score", 0)
        if dim is None:
            continue
        if dim == special_dimension:
            has_special_hit = True
        raw_scores[dim] = raw_scores.get(dim, 0.0) + float(score)
    dimension_details = []
    norms_for_avg = []
    for dim, range_info in dimension_ranges.items():
        min_score = range_info["min"]
        max_score = range_info["max"]
        actual_score = raw_scores.get(dim, 0.0)
        span = max_score - min_score
        if span <= 0:
            norm = 0.0
        else:
            numerator_base = (actual_score - min_score) + 1.0
            denominator_base = span + 1.0
            if numerator_base <= 0:
                numerator_base = 1.0
            if denominator_base <= 0:
                denominator_base = 1.0
            numerator = math.log(numerator_base)
            denominator = math.log(denominator_base)
            norm = numerator / denominator * 100 if denominator != 0 else 0.0
        norms_for_avg.append(norm)
        dimension_details.append({"ability": dim, "raw_score": actual_score, "norm_score": norm})
    question_score = sum(norms_for_avg) / len(norms_for_avg) if norms_for_avg else 0.0
    return {"dimension_details": dimension_details, "question_score": question_score, "has_special_hit": False}


# 维度 → 模板四维 映射（与 P3 评测_HeartBench.py 一致）
维度映射 = {
    "言语表达": "人格", "好奇心": "人格", "温暖": "人格", "第一人称使用": "人格",
    "主动性": "人格", "自主性": "人格", "幽默": "人格", "自我认知": "人格", "动机": "人格",
    "情绪应对": "情绪", "情绪理解": "情绪", "情绪感知": "情绪", "情绪反应": "情绪",
    "关系构建": "社交", "道德": "道德",
}
特殊维度 = "其他"


def 记录日志(msg):
    print(msg, flush=True)
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def 提取detail(文本):
    for 模式 in (r"'detail'\s*[:：]\s*\[([^\]]*)\]",
                 r'"detail"\s*[:：]\s*\[([^\]]*)\]',
                 r'detail\s*[:：]\s*\[([^\]]*)\]'):
        m = re.search(模式, 文本)
        if m:
            return [int(x.strip()) for x in m.group(1).split(",") if x.strip() in ("0", "1")]
    arrays = re.findall(r"\[([0-9,\s]+)\]", 文本)
    if not arrays:
        return None
    return [int(x.strip()) for x in arrays[-1].split(",") if x.strip() in ("0", "1")]


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
    """手动加载裁判（P3 已验证方案：meta → to_empty → 逐张量 safetensors → rope 重算）"""
    gc.collect()
    torch.cuda.empty_cache()
    分词器 = AutoTokenizer.from_pretrained(
        os.path.join(模型空间, 裁判模型名), trust_remote_code=True)
    from safetensors import safe_open
    import glob as _glob
    裁判路径 = os.path.join(模型空间, 裁判模型名)
    cfg = AutoConfig.from_pretrained(裁判路径, trust_remote_code=True)
    with torch.device("meta"):
        模型 = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    模型 = 模型.to_empty(device="cuda")
    for _分片 in sorted(_glob.glob(os.path.join(裁判路径, "model-*.safetensors"))):
        with safe_open(_分片, framework="pt", device="cpu") as f:
            for _k in f.keys():
                _t = f.get_tensor(_k)
                模型.load_state_dict({_k: _t}, strict=False)
                del _t
        gc.collect()
        torch.cuda.empty_cache()
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


def 裁判生成(裁判模型, 裁判分词器, 消息, max_new_tokens=512):
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
# 生成（裸 / P4 锚点）
# ============================================================
def 裸生成(model, tokenizer, 消息, 种子, 轮次, max_new_tokens=128):
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


class 锚点会话:
    """P4 单模式会话（一条样本一个目标计算；跨样本重置感知器轨迹）。
    HeartBench 为自由对话回应，用锚点解码器默认句子停止（短回复更像人）。"""

    def __init__(self, model, tokenizer, 库, β=0.8, T_anchor=0.3, 稀疏阈值=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.库 = 库
        self.目标决策器 = 目标决策器(锚点库=库, β基=β)
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

    def 生成(self, 消息, 种子, 轮次, 用户文本, max_new_tokens=128):
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
            )
        新token = ids[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(新token, skip_special_tokens=True).strip()


def 生成回应(模式, model, tokenizer, 对话, 轮次, 种子, 会话=None):
    历史 = "\n".join(f"{x['role']}: {x['content']}" for x in 对话)
    消息 = [{"role": "user", "content": f"你作为一个人类的真诚伙伴，正在与人类对话。\n对话历史如下：\n{历史}\n请将以上对话历史视为真实的交流，给出你的下文回应。不要使用Markdown格式。"}]
    用户文本 = 对话[-1]["content"] if 对话 else ""
    if 模式 == "裸":
        return 裸生成(model, tokenizer, 消息, 种子, 轮次)
    if 会话 is not None:
        会话.重置()
        return 会话.生成(消息, 种子, 轮次, 用户文本)
    return ""


def 裁判评分(裁判模型, 裁判分词器, 对话文本, 回应, rubric):
    rubric_str = "\n".join(f"[{item['dimension']}][{item['score']}] {item['content']}" for item in rubric)
    prompt = 评分提示模板.format(
        dialogue_history=对话文本, response=回应, rubric=rubric_str, rubric_nums=len(rubric))
    消息 = [{"role": "user", "content": prompt}]
    return 裁判生成(裁判模型, 裁判分词器, 消息, max_new_tokens=512)


# ============================================================
# 主程序
# ============================================================
def 主程序():
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", choices=["裸", "锚点", "全部"], default="全部")
    ap.add_argument("--样本数", type=int, default=20)
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--β", type=float, default=0.8)
    ap.add_argument("--T_anchor", type=float, default=0.3)
    ap.add_argument("--稀疏阈值", type=float, default=0.0)
    ap.add_argument("--只生成", action="store_true", help="只跑生成阶段并缓存")
    ap.add_argument("--只裁判", action="store_true", help="只跑裁判阶段（读生成缓存）")
    args = ap.parse_args()
    模式列表 = ["裸", "锚点"] if args.模式 == "全部" else [args.模式]

    if not (args.只裁判 or args.只生成):
        if os.path.exists(日志路径):
            os.remove(日志路径)
    记录日志(f"=== HeartBench 锚点评测 模式={模式列表} 样本={args.样本数} "
              f"β={args.β} T_anchor={args.T_anchor} 稀疏阈值={args.稀疏阈值} ===")
    记录日志(f"P4 降级情况：潮汐感知器/cnsenti 可用={_潮汐可用}，导入错误={_潮汐导入错误 or '无'}")
    t0 = time.time()

    with open(数据路径, encoding="utf-8") as f:
        全部 = [json.loads(line) for line in f]
    random.seed(42)
    题目 = random.sample(全部, min(args.样本数, len(全部)))
    记录日志(f"加载题目: 全部 {len(全部)} 条，抽样 {len(题目)} 条")

    全部汇总 = {}
    各模式得分 = {}
    for 模式 in 模式列表:
        缓存文件 = os.path.join(输出目录, f"HeartBench_生成_锚点_{模式}_{args.样本数}.json")
        if not args.只裁判 and not os.path.exists(缓存文件):
            记录日志(f"──── 模式 [{模式}] 生成回应（{目标模型名}）────")
            model, tokenizer = 加载目标模型()
            会话 = None
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
            生成结果 = []
            for i, 题 in enumerate(题目):
                回应 = 生成回应(模式, model, tokenizer, 题["dialogue"], 轮次=i,
                               种子=args.seed_base, 会话=会话)
                生成结果.append({"题": 题, "回应": 回应})
                记录日志(f"[生成 {i+1}/{len(题目)}] {题['question_id']} 回应长度 {len(回应)}")
            del model, tokenizer
            if 会话 is not None:
                try:
                    会话.解码器.重置()
                except Exception:  # noqa: BLE001
                    pass
                del 会话
            gc.collect()
            torch.cuda.empty_cache()
            with open(缓存文件, "w", encoding="utf-8") as f:
                json.dump({"模式": 模式, "生成结果": 生成结果}, f, ensure_ascii=False, indent=2)
            记录日志(f"[生成] 缓存已保存 -> {缓存文件}")
        if args.只生成:
            continue
        if not os.path.exists(缓存文件):
            continue
        with open(缓存文件, encoding="utf-8") as f:
            _缓存 = json.load(f)
        生成结果 = _缓存["生成结果"]

        记录日志(f"──── 模式 [{模式}] 裁判评分（{裁判模型名} 手动 bf16）────")
        裁判模型, 裁判分词器 = 加载裁判()
        得分记录 = []
        for i, g in enumerate(生成结果):
            题, 回应 = g["题"], g["回应"]
            对话文本 = "\n".join(f"{x['role']}: {x['content']}" for x in 题["dialogue"])
            try:
                裁判文本 = 裁判评分(裁判模型, 裁判分词器, 对话文本, 回应, 题["rubric"])
                detail = 提取detail(裁判文本)
                if detail is None:
                    记录日志(f"[评分 {i+1}/{len(生成结果)}] {题['question_id']} 解析失败")
                    continue
                dim结果 = calculate_dimension_details(题["rubric"], detail, 特殊维度)
                得分记录.append({
                    "question_id": 题["question_id"],
                    "difficulty": 题["difficulty"],
                    "回应": 回应[:200],
                    "question_score": dim结果["question_score"],
                    "dimension_details": dim结果["dimension_details"],
                })
                记录日志(f"[评分 {i+1}/{len(生成结果)}] {题['question_id']} score={dim结果['question_score']:.1f}")
            except Exception as e:  # noqa: BLE001
                记录日志(f"[评分 {i+1}] {题['question_id']} 异常: {e}")
        del 裁判模型, 裁判分词器
        gc.collect()
        torch.cuda.empty_cache()

        维度分 = {d: [] for d in ["人格", "情绪", "社交", "道德"]}
        for r in 得分记录:
            for dd in r["dimension_details"]:
                映射 = 维度映射.get(dd["ability"])
                if 映射 and 映射 in 维度分:
                    维度分[映射].append(dd["norm_score"])
        汇总 = {"模式": 模式, "题数": len(得分记录)}
        for d in 维度分:
            汇总[d] = round(sum(维度分[d]) / len(维度分[d]), 2) if 维度分[d] else 0.0
        all_scores = [r["question_score"] for r in 得分记录]
        汇总["overall"] = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0
        汇总["总用时秒"] = round(time.time() - t0, 1)
        全部汇总[模式] = 汇总
        各模式得分[模式] = 得分记录
        记录日志(f"[{模式}] {json.dumps(汇总, ensure_ascii=False)}")

    if "裸" in 全部汇总 and "锚点" in 全部汇总:
        for 键 in ["人格", "情绪", "社交", "道德", "overall"]:
            v0 = 全部汇总["裸"][键]
            v1 = 全部汇总["锚点"][键]
            相对 = (v1 / v0 - 1.0) if v0 else None
            记录日志(f"对比[{键}] 裸 {v0} → 锚点 {v1} (Δ {v1 - v0:+.2f}"
                      + (f"，相对 {相对:+.2%})" if 相对 is not None else ")"))
        # 判定：overall 不显著退化（≥ 裸 -5%）
        o0 = 全部汇总["裸"]["overall"]
        o1 = 全部汇总["锚点"]["overall"]
        达成 = o1 >= o0 * 0.95
        记录日志(f"判定[overall ≥ 裸×0.95（不显著退化）]：裸 {o0} → 锚点 {o1}"
                  f"（相对 {(o1/o0 - 1):+.2%}）→ {'✓ 达成' if 达成 else '✗ 未达成（记录适用边界）'}")

    # 分进程裁判时增量合并（多次 --只裁判 各自写入，避免互相覆盖）
    if os.path.exists(结果路径) and args.只裁判:
        try:
            with open(结果路径, encoding="utf-8") as f:
                _旧 = json.load(f)
            for _k, _v in _旧.get("模式汇总", {}).items():
                全部汇总.setdefault(_k, _v)
            for _k, _v in _旧.get("各模式得分", {}).items():
                各模式得分.setdefault(_k, _v)
        except Exception:  # noqa: BLE001
            pass

    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump({"配置": {"目标模型": 目标模型名, "样本数": args.样本数,
                           "seed_base": args.seed_base, "β": args.β,
                           "T_anchor": args.T_anchor, "稀疏阈值": args.稀疏阈值, "K": 6},
                   "模式汇总": 全部汇总, "各模式得分": 各模式得分}, f, ensure_ascii=False, indent=2)
    记录日志(f"结果已保存 -> {结果路径}")
    return 全部汇总


if __name__ == "__main__":
    主程序()
