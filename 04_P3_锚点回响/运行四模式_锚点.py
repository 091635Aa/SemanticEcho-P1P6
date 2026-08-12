# -*- coding: utf-8 -*-
"""
Task8 步骤1：四模式生成（Qwen2.5-1.5B，种子 42，样本_30条.json 30 条）
======================================================================
四模式 × 30 条样本，统一 max_new_tokens=256 / temperature=1.0 / top_p=0.9 /
top_k=50 / repetition_penalty=1.05，目标决策器按每条用户文本自动算 v_target：

  裸   ：原生 model.generate（无注入）
  锚点 ：仅通道 A（锚点解码器 β=0.8, T_anchor=0.3, 稀疏阈值=0.0）
  回响 ：仅通道 B（P1 回响注入器 λ=0.08，GPU 直分配投影 = _GPU回响注入器）
  全开 ：A 锚点β=0.8 + B 回响λ=0.08 + C 潮汐倍率=6（混合锚点器三通道全开）

每模式记录 平均熵 / 重复率 / 情感命中率 / 显存；生成结果 JSON 存
评测结果\\四模式生成_<cfg>.json 可缓存复用（--重新生成 强制重跑，--只模式 单模式跑）。
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 离线加载本地模型，必须在 transformers import 前

import argparse
import gc
import json
import sys
import torch

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)
回响工程根 = r"i:\Desktop\语义回响"
if 回响工程根 not in sys.path:
    sys.path.insert(0, 回响工程根)

from transformers import AutoModelForCausalLM, AutoTokenizer

from 锚点库 import 锚点库
from 目标决策器 import 目标决策器
from 锚点解码器 import 锚点解码器, 计算熵, 计算重复率
from 混合锚点器 import 混合锚点器, _GPU回响注入器

模型路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-1.5B-Instruct"
样本路径 = r"i:\Desktop\语义回响\图灵测试\样本_30条.json"
输出目录 = os.path.join(本目录, "评测结果")
os.makedirs(输出目录, exist_ok=True)
日志路径 = os.path.join(输出目录, "四模式_锚点.log")


def 记录日志(msg):
    print(msg, flush=True)
    with open(日志路径, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def 构建提示(tokenizer, 消息):
    return tokenizer.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)


def 加载目标模型():
    gc.collect()
    torch.cuda.empty_cache()
    分词器 = AutoTokenizer.from_pretrained(模型路径, trust_remote_code=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        模型路径, torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
    模型.eval()
    return 模型, 分词器


def 文本级情感命中率(回复, 库):
    """文本级情感种子词命中率：锚点库词集子串命中数 / 回复长度"""
    if not 回复:
        return 0.0
    命中 = 0
    for 词列表 in 库.词集.values():
        for 词 in 词列表:
            if 词 and 词 in 回复:
                命中 += 1
    return round(命中 / max(len(回复), 1), 4)


# ──────────────────────────────────────────────
# 裸（原生生成 + 再前向熵，KV cache 加速）
# ──────────────────────────────────────────────
def 裸生成(model, tokenizer, 消息, 种子, max_new_tokens):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    提示 = 构建提示(tokenizer, 消息)
    inputs = tokenizer(提示, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids, max_new_tokens=max_new_tokens,
            temperature=1.0, top_p=0.9, top_k=50, do_sample=True,
            repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    回复 = tokenizer.decode(新token, skip_special_tokens=True).strip()
    # 熵：温度 1.0 下再前向分布 = 采样时分布（裸无注入），KV cache 加速
    熵列表 = []
    ids = inputs.input_ids
    pkv = None
    with torch.no_grad():
        for _ in range(len(新token)):
            模型输入 = ids[:, -1:] if pkv is not None else ids
            out2 = model(模型输入, past_key_values=pkv, use_cache=True)
            pkv = out2.past_key_values
            熵列表.append(计算熵(out2.logits[0, -1, :]))
            ids = torch.cat([ids, 新token[_: _ + 1].unsqueeze(0)], dim=-1)
    统计 = {
        "平均熵": round(sum(熵列表) / max(len(熵列表), 1), 4),
        "重复率": 计算重复率(新token.tolist()),
        "情感命中率": 0.0,  # 汇总时按文本级补算
        "β": 0.0, "T_anchor": 0.0, "触发兜底次数": 0,
        "token数": int(len(新token)),
    }
    return 回复, 统计


# ──────────────────────────────────────────────
# 锚点 / 全开 会话（锚点解码器 / 混合锚点器）
# ──────────────────────────────────────────────
class 锚点会话:
    """锚点（仅 A）与 全开（A+B+C）共享会话：一条样本一个目标计算"""

    def __init__(self, model, tokenizer, 库, 模式, β=0.8, T_anchor=0.3, 稀疏阈值=0.0,
                 λ=0.08, 潮汐倍率=6.0, 生成长度=256):
        self.model = model
        self.tokenizer = tokenizer
        self.模式 = 模式
        self.生成长度 = 生成长度
        self.目标决策器 = 目标决策器(锚点库=库, β基=β)
        if 模式 == "锚点":
            self.解码器 = 锚点解码器(
                model, tokenizer, 库, self.目标决策器,
                β=β, T_anchor=T_anchor, 稀疏阈值=稀疏阈值,
                温度=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
            )
        elif 模式 == "全开":
            self.解码器 = 混合锚点器(
                model, tokenizer, 库, self.目标决策器,
                锚点β=β, 锚点T=T_anchor, 回响λ=λ, 潮汐倍率=潮汐倍率,
                开启A=True, 开启B=True, 开启C=True,
                温度=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
            )
        else:
            raise ValueError(f"未知会话模式：{模式}")

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
        统计["token数"] = int(len(新token))
        return 回复, 统计


# ──────────────────────────────────────────────
# 回响（仅通道 B，P1 回响注入器 λ=0.08；GPU 投影防 CPU 每步转移 892MB 慢路径）
# ──────────────────────────────────────────────
def 回响生成(model, tokenizer, 消息, 种子, max_new_tokens):
    torch.manual_seed(种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(种子)
    from semantic_echo.回响池 import 语义回响池
    from semantic_echo.情感过滤器 import 情感过滤器
    过滤器 = 情感过滤器()
    过滤器.加载词库()
    池 = 语义回响池(hidden_dim=model.config.hidden_size, decay_gamma=0.07)
    注入器 = _GPU回响注入器(model, 池, lambda_strength=0.08, 情感过滤器实例=过滤器)
    提示 = 构建提示(tokenizer, 消息)
    inputs = tokenizer(提示, return_tensors="pt").to(model.device)
    熵列表 = []

    def cb(_步, logits):
        熵列表.append(计算熵(logits))

    try:
        with torch.no_grad():
            out = 注入器.生成(
                inputs.input_ids, max_new_tokens=max_new_tokens,
                temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id, tokenizer=tokenizer,
                logits_callback=cb,
            )
    finally:
        # 释放投影矩阵（1536×151936≈892MB）与钩子，防止跨样本 OOM
        try:
            注入器._移除钩子()
        except Exception:  # noqa: BLE001
            pass
        del 注入器, 池, 过滤器
        gc.collect()
        torch.cuda.empty_cache()
    新token = out[0, inputs.input_ids.shape[1]:]
    回复 = tokenizer.decode(新token, skip_special_tokens=True).strip()
    统计 = {
        "平均熵": round(sum(熵列表) / max(len(熵列表), 1), 4),
        "重复率": 计算重复率(新token.tolist()),
        "情感命中率": 0.0,  # 汇总时按文本级补算
        "β": 0.0, "T_anchor": 0.0, "触发兜底次数": 0,
        "token数": int(len(新token)),
    }
    return 回复, 统计


def 汇总健康度(统计列表, 回复列表, 库):
    """生成健康度汇总（无坍缩检查口径）：熵/重复率/情感命中率均值 + 长度/空回复"""
    汇总 = {}
    for k in ("平均熵", "重复率", "情感命中率"):
        值列表 = [s.get(k, 0.0) for s in 统计列表]
        汇总[k] = round(sum(值列表) / max(len(值列表), 1), 4)
    汇总["触发兜底次数均值"] = round(
        sum(s.get("触发兜底次数", 0) for s in 统计列表) / max(len(统计列表), 1), 4)
    文本命中 = [文本级情感命中率(r, 库) for r in 回复列表]
    汇总["文本级情感命中率"] = round(sum(文本命中) / max(len(文本命中), 1), 4)
    长度 = [len(r) for r in 回复列表]
    汇总["平均长度"] = round(sum(长度) / max(len(长度), 1), 2)
    汇总["平均token数"] = round(sum(s.get("token数", 0) for s in 统计列表)
                                / max(len(统计列表), 1), 1)
    汇总["空回复数"] = sum(1 for r in 回复列表 if not r.strip())
    汇总["最小熵"] = min(s["平均熵"] for s in 统计列表)
    汇总["最大重复率"] = max(s["重复率"] for s in 统计列表)
    return 汇总


def 主程序():
    ap = argparse.ArgumentParser()
    ap.add_argument("--样本", type=int, default=30)
    ap.add_argument("--种子", type=int, default=42)
    ap.add_argument("--生成长度", type=int, default=256)
    ap.add_argument("--重新生成", action="store_true", help="忽略已有缓存强制重跑")
    ap.add_argument("--只模式", choices=["裸", "锚点", "回响", "全开"], default=None)
    args = ap.parse_args()

    if os.path.exists(日志路径):
        os.remove(日志路径)
    记录日志(f"=== Task8 步骤1 四模式生成 模型=Qwen2.5-1.5B 样本={args.样本} "
              f"种子={args.种子} 生成长度={args.生成长度} ===")
    with open(样本路径, encoding="utf-8") as f:
        样本 = json.load(f)["样本"]
    随机样本 = 样本[:args.样本]
    模式列表 = ["裸", "锚点", "回响", "全开"] if not args.只模式 else [args.只模式]

    缓存文件 = os.path.join(输出目录,
                            f"四模式生成_样本{args.样本}_S{args.生成长度}_种子{args.种子}.json")
    已有 = {}
    if os.path.exists(缓存文件) and not args.重新生成:
        with open(缓存文件, encoding="utf-8") as f:
            已有 = json.load(f)
    结果 = {"配置": {"目标模型": "Qwen2.5-1.5B-Instruct", "样本路径": 样本路径,
                    "样本数": args.样本, "种子": args.种子, "生成长度": args.生成长度,
                    "裸": "原生 generate 无注入", "锚点": "A:β=0.8 T=0.3 稀疏=0.0",
                    "回响": "B:λ=0.08 (P1)", "全开": "A:β=0.8 + B:λ=0.08 + C:倍率=6",
                    "采样": "T=1.0 top_p=0.9 top_k=50 rep_pen=1.05"},
            "样本": {"user": [r["user"] for r in 随机样本],
                    "girl": [r["girl"] for r in 随机样本]},
            "模式": 已有.get("模式", {})}

    for 模式 in 模式列表:
        if 模式 in 结果["模式"] and not args.重新生成:
            记录日志(f"[{模式}] 缓存已存在，跳过（--重新生成 强制重跑）")
            continue
        记录日志(f"──── 模式 [{模式}] 生成开始 ────")
        model, tokenizer = 加载目标模型()
        torch.cuda.empty_cache()
        显存前 = torch.cuda.memory_allocated() / 1e9
        库 = 锚点库(model, tokenizer)
        基线 = 库.记录只读基线()
        库.构建()
        S = 库.预计算打分表()
        只读 = 库.验证只读(基线)
        if not (只读["sum一致"] and 只读["指针一致"]):
            记录日志("[锚点库] 警告：只读校验失败！")
        记录日志(f"[锚点库] 维度={库.维度名()} 打分表={list(S.shape)} {S.dtype} 只读校验={只读}")

        会话 = None
        if 模式 in ("锚点", "全开"):
            会话 = 锚点会话(model, tokenizer, 库, 模式, 生成长度=args.生成长度)
        回复列表, 统计列表 = [], []
        for i, r in enumerate(随机样本):
            消息 = [{"role": "user", "content": r["user"]}]
            种子 = args.种子 + i
            if 模式 == "裸":
                回复, 统计 = 裸生成(model, tokenizer, 消息, 种子, args.生成长度)
            elif 模式 == "回响":
                回复, 统计 = 回响生成(model, tokenizer, 消息, 种子, args.生成长度)
            else:
                会话.重置()
                回复, 统计 = 会话.生成(消息, 种子, r["user"])
            回复列表.append(回复)
            统计列表.append(统计)
            记录日志(f"[{模式} {i+1}/{len(随机样本)}] 长{len(回复)} 熵{统计['平均熵']} "
                      f"重{统计['重复率']} tok{统计['token数']} {r['user'][:12]} => {回复[:30]}")

        # 文本级情感命中率补算 + 健康度
        for j, r in enumerate(回复列表):
            统计列表[j]["情感命中率"] = 文本级情感命中率(r, 库)
        健康度 = 汇总健康度(统计列表, 回复列表, 库)
        显存后 = torch.cuda.memory_allocated() / 1e9
        结果["模式"][模式] = {"回复": 回复列表, "统计": 统计列表, "健康度": 健康度,
                            "显存GB": {"前": round(显存前, 2), "后": round(显存后, 2)}}
        with open(缓存文件, "w", encoding="utf-8") as f:
            json.dump(结果, f, ensure_ascii=False, indent=2)
        记录日志(f"[{模式}] 健康度 {json.dumps(健康度, ensure_ascii=False)} "
                  f"显存 前{显存前:.2f}GB 后{显存后:.2f}GB")

        # 彻底释放
        del model, tokenizer, 库
        if 会话 is not None:
            try:
                会话.解码器.重置()
            except Exception:  # noqa: BLE001
                pass
            del 会话
        gc.collect()
        torch.cuda.empty_cache()
    记录日志(f"结果已保存 -> {缓存文件}")


if __name__ == "__main__":
    主程序()
