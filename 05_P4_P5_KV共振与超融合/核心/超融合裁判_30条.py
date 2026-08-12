# -*- coding: utf-8 -*-
"""超融合解码器（UFD）LLM-Judge —— 裁判阶段（本地 7B 手动 bf16 加载，v2 断点续跑）

协议（对齐 P4 三正交叠加报告）：配对盲评 AB 正反各一次，win_rate = 该模式
回复胜过真人回复的配对比例；裁判提示词/加载路径与 ETD 评测_LLM_Judge.py 一致。

v2 增强：每调用异常重试 2 次；每 5 样本 gc + 清显存；逐样本增量保存，支持断点续跑
（已完成的模式自动跳过）。
"""
import os
import sys
import json
import gc
import glob
import re
import time
import torch
from datetime import datetime

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors import safe_open

工作目录 = os.path.dirname(os.path.abspath(__file__))
if 工作目录 not in sys.path:
    sys.path.insert(0, 工作目录)

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
裁判模型名 = "Qwen2.5-7B-Instruct"
输出目录 = os.path.join(工作目录, "..", "评测结果")
生成路径 = os.path.join(输出目录, "超融合_生成_30.json")
结果路径 = os.path.join(输出目录, "超融合_LLMJudge_30.json")

裁判配对提示 = """你是一位严格的中文图灵测试裁判。用户说了一句话，下面有两个回复：回复A 和 回复B。

用户：{user}
回复A：{回复A}
回复B：{回复B}

其中一个回复是真人写的，另一个是 AI 生成的。
请判断：哪个回复更可能出自真人之手？

必须严格按以下 JSON 格式输出（不要输出其他内容）：
{{"更像真人": "A" 或 "B", "理由": "一句话理由"}}"""


def 加载裁判():
    gc.collect()
    torch.cuda.empty_cache()
    裁判路径 = os.path.join(模型空间, 裁判模型名)
    cfg = AutoConfig.from_pretrained(裁判路径, trust_remote_code=True)
    with torch.device("meta"):
        模型 = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    模型 = 模型.to_empty(device="cuda")
    for _分片 in sorted(glob.glob(os.path.join(裁判路径, "model-*.safetensors"))):
        with safe_open(_分片, framework="pt", device="cpu") as f:
            _sd = {k: f.get_tensor(k) for k in f.keys()}
        模型.load_state_dict(_sd, strict=False)
        del _sd
        gc.collect()
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
    return 模型, AutoTokenizer.from_pretrained(裁判路径, trust_remote_code=True)


def 裁判生成(裁判模型, 裁判分词器, 消息, max_new_tokens=120):
    提示 = 裁判分词器.apply_chat_template(消息, tokenize=False, add_generation_prompt=True)
    inputs = 裁判分词器(提示, return_tensors="pt").to(裁判模型.device)
    with torch.no_grad():
        out = 裁判模型.generate(
            inputs.input_ids, max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=裁判分词器.eos_token_id,
        )
    新token = out[0, inputs.input_ids.shape[1]:]
    return 裁判分词器.decode(新token, skip_special_tokens=True).strip()


def 解析配对(文本):
    m = re.search(r'"更像真人"\s*[:：]\s*"([AB])"', 文本)
    if m:
        return m.group(1)
    if "回复A" in 文本 and "回复B" not in 文本.split("更像真人")[-1][:40]:
        return "A"
    if "回复B" in 文本 and "回复A" not in 文本.split("更像真人")[-1][:40]:
        return "B"
    return None


def 安全裁判生成(裁判模型, 裁判分词器, user, A, B):
    """单次配对判定，失败重试 2 次；仍失败返回 None（不计入统计）"""
    for _尝试 in range(3):
        try:
            文本 = 裁判生成(裁判模型, 裁判分词器, [{"role": "user", "content":
                 裁判配对提示.format(user=user, 回复A=A, 回复B=B)}])
            return 解析配对(文本)
        except Exception as e:  # noqa: BLE001
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)
            if _尝试 == 2:
                print(f"    裁判调用失败3次：{e}", flush=True)
    return None


def main():
    print(f"=== UFD LLM-Judge 裁判阶段(v2 断点续跑) {datetime.now().strftime('%H:%M:%S')} ===", flush=True)
    with open(生成路径, "r", encoding="utf-8") as f:
        数据 = json.load(f)
    样本 = 数据["回复"]
    模式列表 = 数据["模式"]
    print(f"样本数={len(样本)} 模式={模式列表}", flush=True)

    # 断点续跑：加载已有结果（若存在）
    结果 = None
    if os.path.exists(结果路径):
        try:
            with open(结果路径, "r", encoding="utf-8") as f:
                结果 = json.load(f)
            print(f"检测到已有结果，断点续跑：已完成的模式={list(结果.get('配对', {}).keys())}", flush=True)
        except Exception:
            结果 = None
    if 结果 is None:
        结果 = {"模型": 数据["模型"], "裁判": 裁判模型名, "样本数": len(样本),
                "模式": 模式列表, "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "配对": {}}

    裁判模型, 裁判分词器 = 加载裁判()
    print("裁判模型加载完成", flush=True)

    for 模式 in 模式列表:
        if 模式 in 结果["配对"]:
            print(f"[{模式}] 已完成，跳过", flush=True)
            continue
        胜 = 0
        总 = 0
        配对明细 = []
        for i, 项 in enumerate(样本):
            user = 项["user"]
            真人 = 项["girl"]
            AI = 项["回复"][模式]["文本"]
            if not AI.strip():
                AI = "（空回复）"
            for AI在前 in (True, False):
                A, B = (AI, 真人) if AI在前 else (真人, AI)
                选择 = 安全裁判生成(裁判模型, 裁判分词器, user, A, B)
                if 选择 is None:
                    continue
                AI胜 = (选择 == "A") if AI在前 else (选择 == "B")
                胜 += 1 if AI胜 else 0
                总 += 1
                配对明细.append({"序号": 项["序号"], "AI在前": AI在前, "裁判选择": 选择,
                                "AI胜": AI胜, "裁判原文": ""})
            if (i + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            if (i + 1) % 10 == 0:
                print(f"  [{模式}] {i+1}/{len(样本)} 暂win_rate={胜/max(总,1):.4f}", flush=True)
                # 增量保存
                结果["配对"][模式] = {"win_rate": round(胜 / max(总, 1), 4),
                                    "胜": 胜, "总": 总, "明细": 配对明细,
                                    "未完成": True}
                with open(结果路径, "w", encoding="utf-8") as f:
                    json.dump(结果, f, ensure_ascii=False, indent=2)
        win_rate = 胜 / max(总, 1)
        结果["配对"][模式] = {"win_rate": round(win_rate, 4), "胜": 胜, "总": 总,
                            "明细": 配对明细}
        with open(结果路径, "w", encoding="utf-8") as f:
            json.dump(结果, f, ensure_ascii=False, indent=2)
        print(f"  [{模式}] win_rate={win_rate:.4f} ({胜}/{总})", flush=True)

    print("\n================ 汇总 ================", flush=True)
    for 模式 in 模式列表:
        wr = 结果["配对"][模式]["win_rate"]
        print(f"{模式:<8} win_rate={wr:.4f}", flush=True)
    print(f"\n裁判完成，已保存：{结果路径}", flush=True)


if __name__ == "__main__":
    main()
