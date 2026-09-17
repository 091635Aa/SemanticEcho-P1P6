# -*- coding: utf-8 -*-
"""
LLM-as-Judge — 四模式三正交叠加验证（P4 Task8：锚点回响 × 回响 × 潮汐）
========================================================================
在 Qwen2.5-1.5B 上跑四模式 LLM-Judge（同种子 42、样本_30条.json 30 条、每模式 60 配对）：

  模式1 裸   —— 纯模型采样循环（三通道全关 + 句子停止关，等价 model.generate 同参采样）
  模式2 锚点 —— A 通道单开（锚点β=0.8, 稀疏阈值=0.0, T_anchor=0.3, K=6，Task5 最优）
  模式3 回响 —— B 通道单开（回响λ=0.08，复用 P1 回响注入器钩子/池/投影，GPU 直分配）
  模式4 混合 —— A+B+C 三通道全开（锚点β=0.8, 回响λ=0.08, 潮汐倍率=6.0，
                 α 由 目标决策器 共享输出，锚点解码器 v_target 同源）

判定（SubTask 8.1/8.2）：
  ① 三通道混合 win_rate ≥ 裸 + 18%（绝对提升，对齐 P3 混合 v4.2 与 spec ≥+18% 底线）
  ② 无坍缩：四模式重复率均 <0.6、熵>0.6、无空回复
  ③ 正交可叠加：三通道混合 win_rate ≥ 任一单通道（锚点/回响）
记录各模式 熵/重复率/情感命中率/显存；参数敏感性（--β / --潮汐倍率 变体）可选。

协议完全复用 P3/P4 LLM-Judge（裁判提示词/盲评流程/win_rate 计算）；生成与裁判
分进程（--只生成 / --只裁判 + 缓存）；裁判 7B 手动逐张量加载（P3 已验证方案）。
本任务只做三通道正交叠加，不加 v8 人设/自选N，保持纯净对照。

用法：
  F:\打标\.venv\Scripts\python.exe 评测_LLM_Judge_混合.py --模式 全部 --只生成
  F:\打标\.venv\Scripts\python.exe 评测_LLM_Judge_混合.py --模式 全部 --只裁判
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 离线加载本地模型，必须在 transformers import 前

import argparse
import gc
import json
import re
import sys
import time
import torch

# 锚点回响工作目录（锚点库 / 目标决策器 / 混合锚点器 所在）
本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器, _潮汐可用, _潮汐导入错误
from 混合锚点器 import 混合锚点器, _回响可用, _回响错误

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
全局模型名 = "Qwen2.5-1.5B-Instruct"  # 由 --目标模型 覆盖
裁判模型名 = "Qwen2.5-7B-Instruct"
样本路径 = r"i:\Desktop\语义回响\图灵测试\样本_30条.json"  # 由 --样本路径 覆盖
输出目录 = os.path.join(本目录, "评测结果")
os.makedirs(输出目录, exist_ok=True)
日志路径 = os.path.join(输出目录, "LLM_Judge_混合.log")
结果路径 = os.path.join(输出目录, "LLM_Judge_四模式.json")

# 四模式通道配置（正交消融：A 锚点 / B 回响 / C 潮汐）
模式配置 = {
    "裸":   {"A": False, "B": False, "C": False, "句子停止": False, "名称": "裸（纯模型）"},
    "锚点": {"A": True,  "B": False, "C": False, "句子停止": True,  "名称": "锚点单模式（A）"},
    "回响": {"A": False, "B": True,  "C": False, "句子停止": True,  "名称": "回响单模式（B）"},
    "混合": {"A": True,  "B": True,  "C": True,  "句子停止": True,  "名称": "锚点+回响+潮汐（A+B+C）"},
}
默认模式列表 = ["裸", "锚点", "回响", "混合"]


def 构建提示(tokenizer, 消息):
    """构建 chat 提示；Qwen3 系列关闭 thinking（直接传 kwargs，模板输出空 <think> 块跳过思考）"""
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


def 加载裁判():
    """加载 7B 裁判：优先 bitsandbytes 8bit（GPU ~8.7GB，16GB 内存机稳定）。

    诊断记录（2026-08-10）：bf16 meta→to_empty→逐张量 手动加载 在 to_empty 后
    GPU 已 15.28GB 满载，H2D 拷贝阶段触发 torch_cpu.dll 原生崩溃 0xC0000005
    （确定性复现，05:45 前同机曾成功）；8bit 加载仅占 ~8.7GB 稳定通过。
    8bit 不可用时回退 bf16 手动加载（P3 已验证方案）。
    """
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"[加载裁判] 显存占用={torch.cuda.memory_allocated()/1e9:.2f}GB 缓存={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
    分词器 = AutoTokenizer.from_pretrained(
        os.path.join(模型空间, 裁判模型名), trust_remote_code=True)
    try:
        from transformers import BitsAndBytesConfig
        量化配置 = BitsAndBytesConfig(load_in_8bit=True)
        模型 = AutoModelForCausalLM.from_pretrained(
            os.path.join(模型空间, 裁判模型名), quantization_config=量化配置,
            device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
        print(f"[加载裁判] 8bit 加载成功 显存={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    except Exception as e:  # noqa: BLE001 —— 回退 bf16 手动加载
        print(f"[加载裁判] 8bit 加载失败（{e}），回退 bf16 手动加载", flush=True)
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
# 生成（四模式统一走 混合锚点器，通道开关 = 模式配置；裸 = 全关）
# ============================================================
class 混合会话:
    """四模式统一会话：混合锚点器 + 通道开关；裸模式 = A/B/C 全关 + 句子停止关。"""

    def __init__(self, model, tokenizer, 库, β=0.8, T_anchor=0.3, 回响λ=0.08,
                 潮汐倍率=6.0, 开A=True, 开B=True, 开C=True, 句子停止=True):
        self.model = model
        self.tokenizer = tokenizer
        self.目标决策器 = 目标决策器(锚点库=库, β基=β)
        self.混合 = 混合锚点器(
            model, tokenizer, 库, self.目标决策器,
            锚点β=β, 锚点T=T_anchor, 回响λ=回响λ, 潮汐倍率=潮汐倍率,
            开启A=开A, 开启B=开B, 开启C=开C,
            温度=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
            句子停止=句子停止, 最短字数=12, 最大字数=90, 最长句数=2, 最小长度=0,
        )

    def 重置(self):
        """感知器轨迹跨样本必须隔离 → 每条生成前重置轨迹（混合锚点器.重置 不重置决策器轨迹）"""
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
            ids, 统计 = self.混合.生成(
                inputs.input_ids, max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id, tokenizer=self.tokenizer,
                用户文本=用户文本,
            )
        新token = ids[0, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(新token, skip_special_tokens=True).strip(), 统计

    def 清理(self):
        """移除回响钩子并释放投影矩阵（B 通道开时），防跨模式/跨样本 OOM"""
        if self.混合.开启B and self.混合.回响 is not None:
            try:
                self.混合.回响._移除钩子()
            except Exception:  # noqa: BLE001
                pass


def 生成AI回复(模式, args, 随机样本):
    """生成阶段：只加载目标模型（1.5B）+ 锚点库 + 混合锚点器，输出 AI 回复列表与模式级统计。"""
    记录日志(f"──── 模式 [{模式}] {模式配置[模式]['名称']} AI 生成（max_tokens={args.生成长度} "
              f"β={args.β} T={args.锚点T} 回响λ={args.回响λ} 潮汐倍率={args.潮汐倍率}） ────")
    model, tokenizer = 加载目标模型()
    库 = 锚点库(model, tokenizer)
    基线 = 库.记录只读基线()
    库.构建()
    S = 库.预计算打分表()
    只读 = 库.验证只读(基线)
    if not (只读["sum一致"] and 只读["指针一致"]):
        记录日志("[锚点库] 警告：只读校验失败！")
    记录日志(f"[锚点库] 维度={库.维度名()} 打分表={list(S.shape)} {S.dtype} 只读校验={只读}")
    配置 = 模式配置[模式]
    会话 = 混合会话(model, tokenizer, 库, β=args.β, T_anchor=args.锚点T, 回响λ=args.回响λ,
                     潮汐倍率=args.潮汐倍率, 开A=配置["A"], 开B=配置["B"], 开C=配置["C"],
                     句子停止=配置["句子停止"])
    显存构建后 = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    记录日志(f"[{模式}] 会话构建完成 显存={显存构建后:.2f}GB 回响通道可用={会话.混合.开启B}")
    AI回复列表, 统计列表 = [], []
    for i, r in enumerate(随机样本):
        消息 = [{"role": "user", "content": r["user"]}]
        会话.重置()
        try:
            回复, 统计 = 会话.生成(消息, args.种子 + i, r["user"], max_new_tokens=args.生成长度)
        except Exception as e:  # noqa: BLE001 —— 单样本异常不中断，记录空回复
            记录日志(f"[AI生成 {i+1}/{len(随机样本)}] 异常：{e}")
            回复, 统计 = "", {"平均熵": 0.0, "重复率": 1.0, "情感命中率": 0.0,
                              "触发兜底次数": 0, "β": args.β}
        AI回复列表.append(回复)
        统计列表.append(统计)
        记录日志(f"[AI生成 {i+1}/{len(随机样本)}] 长{len(回复)} 熵{统计['平均熵']} "
                  f"重复{统计['重复率']} 命中{统计['情感命中率']} 兜底{统计.get('触发兜底次数', 0)} "
                  f"{r['user'][:16]} => {回复[:34]}")
    显存峰值 = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    会话.清理()
    # 从主作用域彻底释放目标模型/锚点库/会话（含回响投影矩阵）
    del 会话, 库, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    # ── 模式级统计（均值）──
    空回复数 = sum(1 for r in AI回复列表 if not (r or "").strip())
    模式统计 = {
        "平均熵": round(sum(s["平均熵"] for s in 统计列表) / len(统计列表), 4) if 统计列表 else 0.0,
        "重复率": round(sum(s["重复率"] for s in 统计列表) / len(统计列表), 4) if 统计列表 else 0.0,
        "情感命中率": round(sum(s["情感命中率"] for s in 统计列表) / len(统计列表), 4) if 统计列表 else 0.0,
        "兜底触发总数": int(sum(s.get("触发兜底次数", 0) for s in 统计列表)),
        "空回复数": 空回复数,
        "显存GB": round(max(显存构建后, 显存峰值), 2),
    }
    记录日志(f"[{模式}] 模式统计：{json.dumps(模式统计, ensure_ascii=False)}")
    return AI回复列表, 模式统计


# ============================================================
# 裁判
# ============================================================
def 裁判盲评(模式, args, 随机样本, AI回复列表):
    """裁判阶段：独立进程加载 7B 裁判，对 AI 回复 vs 真人回复盲评（60 配对 + 30 评分）"""
    记录日志(f"──── 模式 [{模式}] 裁判盲评（{len(随机样本)} 样本 → 60 配对 + 30 评分） ────")
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
    """输出四模式对照与三项判定（① 混合≥裸+18% ② 无坍缩 ③ 正交可叠加）"""
    记录日志("\n" + "=" * 72)
    记录日志("四模式对照表")
    记录日志("=" * 72)
    记录日志(f"{'模式':<6}{'win_rate':>10}{'rating':>9}{'熵':>8}{'重复率':>9}{'命中率':>9}{'显存GB':>9}{'空回复':>7}")
    for 模式 in 默认模式列表:
        if 模式 not in 全部汇总:
            continue
        s = 全部汇总[模式]
        记录日志(f"{模式:<6}{s['win_rate_against_human']:>10.4f}{s['average_rating']:>9.4f}"
                  f"{s['平均熵']:>8.4f}{s['重复率']:>9.4f}{s['情感命中率']:>9.4f}"
                  f"{s['显存GB']:>9.2f}{s['空回复数']:>7d}")
    # ① 无坍缩
    记录日志("-" * 72)
    for 模式 in 默认模式列表:
        if 模式 not in 全部汇总:
            continue
        s = 全部汇总[模式]
        无坍缩 = s["平均熵"] > 0.6 and s["重复率"] < 0.6 and s["空回复数"] == 0
        记录日志(f"无坍缩[{模式}] 熵={s['平均熵']}(>0.6:{s['平均熵'] > 0.6}) "
                  f"重复率={s['重复率']}(<0.6:{s['重复率'] < 0.6}) "
                  f"空回复={s['空回复数']}(=0:{s['空回复数'] == 0}) → {'✓' if 无坍缩 else '✗'}")
    # ② 混合 vs 裸
    if "混合" in 全部汇总 and "裸" in 全部汇总:
        wr0 = 全部汇总["裸"]["win_rate_against_human"]
        wr1 = 全部汇总["混合"]["win_rate_against_human"]
        Δ = wr1 - wr0
        达成 = Δ >= 0.18
        记录日志(f"判定[三通道混合 win_rate ≥ 裸+18%]：裸 {wr0:.4f} → 混合 {wr1:.4f} "
                  f"(Δ {Δ:+.4f}) → {'✓ 达成' if 达成 else '✗ 未达成'}")
    # ③ 正交可叠加（混合 ≥ 任一单通道）
    if "混合" in 全部汇总:
        单通道列表 = [全部汇总[m]["win_rate_against_human"] for m in ("锚点", "回响")
                    if m in 全部汇总]
        if 单通道列表:
            单通道max = max(单通道列表)
            可叠加 = 全部汇总["混合"]["win_rate_against_human"] >= 单通道max
            记录日志(f"判定[三通道叠加 ≥ 任一单通道]：单通道max={单通道max:.4f} "
                      f"(锚点={全部汇总.get('锚点', {}).get('win_rate_against_human', float('nan')):.4f}, "
                      f"回响={全部汇总.get('回响', {}).get('win_rate_against_human', float('nan')):.4f}) "
                      f"→ 混合={全部汇总['混合']['win_rate_against_human']:.4f} → "
                      f"{'✓ 可叠加' if 可叠加 else '✗ 不可叠加'}")
    # ④ 单通道 vs 裸（补充）
    for m in ("锚点", "回响"):
        if m in 全部汇总 and "裸" in 全部汇总:
            wr0 = 全部汇总["裸"]["win_rate_against_human"]
            wr1 = 全部汇总[m]["win_rate_against_human"]
            记录日志(f"对比[{m} vs 裸] {wr0:.4f} → {wr1:.4f} (Δ {wr1 - wr0:+.4f})")
    记录日志("=" * 72)


def 主程序():
    ap = argparse.ArgumentParser()
    ap.add_argument("--模式", choices=["裸", "锚点", "回响", "混合", "全部"], default="全部")
    ap.add_argument("--样本", type=int, default=30)
    ap.add_argument("--种子", type=int, default=42)
    ap.add_argument("--目标模型", default="Qwen2.5-1.5B-Instruct", help="横向测试：目标模型名")
    ap.add_argument("--样本路径", default=r"i:\Desktop\语义回响\图灵测试\样本_30条.json",
                    help="样本文件（30条/60条）")
    ap.add_argument("--生成长度", type=int, default=64, help="max_new_tokens")
    ap.add_argument("--β", type=float, default=0.8, help="锚点通道注入强度（Task5 最优 0.8；敏感性可试 1.2）")
    ap.add_argument("--锚点T", type=float, default=0.3, help="P4 tanh 内积温度")
    ap.add_argument("--回响λ", type=float, default=0.08, help="回响通道注入强度（P1 定标 0.08）")
    ap.add_argument("--潮汐倍率", type=float, default=6.0, help="潮汐通道引导倍率（P3 混合 v4.2 定标 6.0）")
    ap.add_argument("--只生成", action="store_true", help="只跑生成阶段并缓存")
    ap.add_argument("--只裁判", action="store_true", help="只跑裁判阶段（读生成缓存）")
    args = ap.parse_args()
    模式列表 = 默认模式列表 if args.模式 == "全部" else [args.模式]

    global 全局模型名, 日志路径, 结果路径, 样本路径
    全局模型名 = args.目标模型
    样本路径 = args.样本路径
    短名 = 全局模型名.replace("-Instruct", "")
    if 短名 != "Qwen2.5-1.5B":
        日志路径 = os.path.join(输出目录, f"LLM_Judge_混合_{短名}.log")
        结果路径 = os.path.join(输出目录, f"LLM_Judge_混合_{短名}.json")
    # 非默认通道参数（强注入边界等变体）→ 结果文件加配置后缀，避免覆盖主四模式结果
    if args.β != 0.8 or args.回响λ != 0.08 or args.潮汐倍率 != 6.0:
        结果路径 = os.path.join(输出目录, f"LLM_Judge_四模式_β{args.β}_L{args.回响λ}_M{args.潮汐倍率}.json")

    if not (args.只裁判 or args.只生成):
        if os.path.exists(日志路径):
            os.remove(日志路径)
    记录日志(f"=== LLM-as-Judge 四模式三正交叠加 模型={全局模型名} 模式={模式列表} 样本={args.样本} "
              f"β={args.β} T={args.锚点T} 回响λ={args.回响λ} 潮汐倍率={args.潮汐倍率} 种子={args.种子} ===")
    记录日志(f"P4 降级情况：潮汐感知器/cnsenti 可用={_潮汐可用}，导入错误={_潮汐导入错误 or '无'}；"
              f"回响注入器可用={_回响可用}，导入错误={_回响错误 or '无'}")
    记录日志("纯净对照：不加 v8 人设 / 自选N / 身份拦截，只做三通道正交叠加")
    with open(样本路径, encoding="utf-8") as f:
        样本 = json.load(f)["样本"]
    随机样本 = 样本[:args.样本]
    记录日志(f"样本总数 {len(样本)}，使用 {len(随机样本)}")

    全部汇总 = {}
    各模式AI回复 = {}
    # 缓存前缀含配置标记（样本集 + 通道参数），避免串用旧缓存
    _样本名 = os.path.splitext(os.path.basename(args.样本路径))[0]
    cfg标记 = f"{_样本名}_S{args.生成长度}_β{args.β}_M{args.潮汐倍率}"
    if args.回响λ != 0.08:
        cfg标记 += f"_L{args.回响λ}"
    缓存前缀 = os.path.join(输出目录, f"生成_混合_{短名}_{args.样本}_{cfg标记}")
    for 模式 in 模式列表:
        # ── 生成阶段（--只生成 独立运行，缓存到文件供裁判阶段复用）──
        缓存文件 = f"{缓存前缀}_{模式}.json"
        if not args.只裁判 and not os.path.exists(缓存文件):
            AI回复列表, 模式统计 = 生成AI回复(模式, args, 随机样本)
            with open(缓存文件, "w", encoding="utf-8") as f:
                json.dump({"模式": 模式, "AI回复": AI回复列表, "模式统计": 模式统计,
                           "user": [r["user"] for r in 随机样本],
                           "girl": [r["girl"] for r in 随机样本]},
                          f, ensure_ascii=False, indent=2)
            记录日志(f"[{模式}] 生成缓存已保存 -> {缓存文件}")
        if args.只生成:
            continue
        if not os.path.exists(缓存文件):
            记录日志(f"[{模式}] 无生成缓存，跳过裁判：{缓存文件}")
            continue
        with open(缓存文件, encoding="utf-8") as f:
            _缓存 = json.load(f)
        AI回复列表 = _缓存["AI回复"]
        各模式AI回复[模式] = AI回复列表
        # ── 裁判阶段（独立进程加载 7B 裁判，避免同进程 OOM）──
        汇总 = 裁判盲评(模式, args, 随机样本, AI回复列表)
        汇总.update(_缓存.get("模式统计", {}))   # 合并生成侧统计（熵/重复率/命中率/显存）
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
    时间戳 = time.strftime("%Y%m%d_%H%M%S")
    _cfg后缀 = os.path.basename(结果路径).replace("LLM_Judge_四模式", "").replace(".json", "")
    时间戳结果路径 = os.path.join(输出目录, f"LLM_Judge_四模式{_cfg后缀}_{时间戳}.json")
    with open(时间戳结果路径, "w", encoding="utf-8") as f:
        json.dump({"配置": {"目标模型": 全局模型名, "样本路径": 样本路径, "样本数": args.样本,
                            "β": args.β, "锚点T": args.锚点T, "回响λ": args.回响λ,
                            "潮汐倍率": args.潮汐倍率, "种子": args.种子, "生成长度": args.生成长度,
                            "潮汐可用": _潮汐可用, "回响可用": _回响可用},
                   "模式汇总": 全部汇总, "AI回复": 各模式AI回复},
                  f, ensure_ascii=False, indent=2)
    # 同步覆盖固定名结果文件（便于汇总查找），并保留时间戳版本
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump({"配置": {"目标模型": 全局模型名, "样本路径": 样本路径, "样本数": args.样本,
                            "β": args.β, "锚点T": args.锚点T, "回响λ": args.回响λ,
                            "潮汐倍率": args.潮汐倍率, "种子": args.种子, "生成长度": args.生成长度},
                   "模式汇总": 全部汇总, "AI回复": 各模式AI回复},
                  f, ensure_ascii=False, indent=2)
    记录日志(f"结果已保存 -> {时间戳结果路径}")
    return 全部汇总


if __name__ == "__main__":
    主程序()
