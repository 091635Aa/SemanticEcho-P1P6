# -*- coding: utf-8 -*-
"""
Task8 步骤2：三裁判交叉验证（本地 7B / deepseek-v4-flash / deepseek-v4-pro）
============================================================================
对同一批四模式回复（评测结果\\四模式生成_样本30_S256_种子42.json）做 AB 盲评
（AI vs 真人，配对方式与 评测_LLM_Judge_锚点.py 完全一致：AB 正反各一次，
win_rate = 配对胜数/配对总数），三个裁判分别打分：

  裁判①本地7B ：Qwen2.5-7B-Instruct（手动加载，温度0.2 贪心）——沿用既有加载与提示词
  裁判②flash  ：deepseek-v4-flash（OpenAI API，same 提示词模板，思考关）
  裁判③pro    ：deepseek-v4-pro  （OpenAI API，same 提示词模板，思考关）
  + 思考开关对比：裁判③ pro 再跑一组 思考开（裸 vs 全开 两模式），看思考是否改变判定

控制变量：同一回复文件、同一配对顺序、同一裁判提示词模板（仅 API 适配），只变裁判。

用法：
  F:\打标\.venv\Scripts\python.exe 评测_双裁判.py --裁判 本地7B            # 四模式
  F:\打标\.venv\Scripts\python.exe 评测_双裁判.py --裁判 flash --模式 裸,锚点,回响,全开
  F:\打标\.venv\Scripts\python.exe 评测_双裁判.py --裁判 pro --思考开       # 追加思考开对比
  F:\打标\.venv\Scripts\python.exe 评测_双裁判.py --汇总                    # 只读结果输出总表
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

本目录 = os.path.dirname(os.path.abspath(__file__))
if 本目录 not in sys.path:
    sys.path.insert(0, 本目录)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
裁判模型名 = "Qwen2.5-7B-Instruct"
裁判路径 = os.path.join(模型空间, 裁判模型名)
生成文件 = os.path.join(本目录, "评测结果", "四模式生成_样本30_S256_种子42.json")
输出目录 = os.path.join(本目录, "评测结果")
os.makedirs(输出目录, exist_ok=True)
时间戳 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 固定文件名：三个裁判/多轮续跑共用同一结果文件（进程内增量合并 + 断点续跑）
结果路径 = os.path.join(输出目录, "双裁判_三正交.json")
日志路径 = os.path.join(输出目录, "双裁判_三正交.log")

# ⚠️ 仅本地脚本使用，绝不推送任何 GitHub 仓库；报告用 sk-**** 脱敏
云API_KEY = "sk-ecd61cf28d244628837f7f87d0d8a96f"
云BASE_URL = "https://api.deepseek.com"

模式顺序 = ["裸", "锚点", "回响", "全开"]

# ── 裁判提示词模板（与 评测_LLM_Judge_锚点.py 完全一致，三裁判共用）──
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


def 汇总(配对胜数, 配对总数, 评分列表):
    win_rate = 配对胜数 / 配对总数 if 配对总数 else 0.0
    avg_rating = sum(评分列表) / len(评分列表) if 评分列表 else 0.0
    return {"win_rate_against_human": round(win_rate, 4),
            "average_rating": round(avg_rating / 5.0, 4), "配对总数": 配对总数,
            "AI评分均值": round(avg_rating, 2), "评分样本": len(评分列表)}


# ══════════════════════════════════════════════════
# 裁判①：本地 7B（P3/P4 已验证手动加载方案）
# ══════════════════════════════════════════════════
def 加载裁判():
    """8bit 手动加载裁判（bitsandbytes，诊断_裁判8bit.py 已验证可行；
    bf16 7B≈15GB 在本机 16GB 显存会 OOM，故用 load_in_8bit + device_map=auto）"""
    gc.collect()
    torch.cuda.empty_cache()
    from transformers import BitsAndBytesConfig
    print("[加载裁判] 8bit 加载 Qwen2.5-7B-Instruct ...", flush=True)
    分词器 = AutoTokenizer.from_pretrained(裁判路径, trust_remote_code=True)
    量化配置 = BitsAndBytesConfig(load_in_8bit=True)
    模型 = AutoModelForCausalLM.from_pretrained(
        裁判路径, quantization_config=量化配置, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True)
    模型.eval()
    if torch.cuda.is_available():
        print(f"[加载裁判] 显存占用={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"缓存={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
    return 模型, 分词器


def 本地裁判生成(裁判模型, 裁判分词器, 消息, max_new_tokens=120):
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


def 裁判_本地7B(模式列表, 随机样本, AI回复字典, 结果):
    裁判模型, 裁判分词器 = 加载裁判()
    裁判段 = 结果["裁判"].setdefault("本地7B", {"说明": "Qwen2.5-7B-Instruct 8bit 贪心(温度0.2)"})
    裁判段.setdefault("模式", {})
    裁判段.setdefault("明细", {})
    for 模式 in 模式列表:
        if 裁判段["模式"].get(模式, {}).get("已完成条数", 0) >= len(随机样本):
            记录日志(f"[本地7B] 模式 [{模式}] 已完成({len(随机样本)}条)，跳过续跑")
            continue
        记录日志(f"──── 裁判[本地7B] 模式 [{模式}] ────")
        AI回复列表 = AI回复字典[模式]
        配对胜数 = 配对总数 = 0
        评分列表 = []
        明细 = []
        for i, r in enumerate(随机样本):
            用户, 真人 = r["user"], r["girl"]
            ai回复 = AI回复列表[i]
            输出A = 本地裁判生成(裁判模型, 裁判分词器, [{"role": "user", "content": 裁判配对提示.format(
                user=用户, 回复A=ai回复, 回复B=真人)}])
            输出B = 本地裁判生成(裁判模型, 裁判分词器, [{"role": "user", "content": 裁判配对提示.format(
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
            评分文本 = 本地裁判生成(裁判模型, 裁判分词器, [{"role": "user", "content": 裁判评分提示.format(
                user=用户, 回复=ai回复)}])
            分 = 解析评分(评分文本)
            if 分 is not None:
                评分列表.append(分)
            明细.append({"样本": i + 1, "选择A": 选择A, "选择B": 选择B, "评分": 分})
            记录日志(f"[本地7B {模式} {i+1}/{len(随机样本)}] 配对(A:{选择A},B:{选择B}) AI评分={分}")
        _s = 汇总(配对胜数, 配对总数, 评分列表)
        _s["已完成条数"] = len(随机样本)
        裁判段["模式"][模式] = _s
        裁判段["明细"][模式] = 明细
        记录日志(f"[本地7B {模式}] {json.dumps(_s, ensure_ascii=False)}")
        _写盘(结果)
    del 裁判模型, 裁判分词器
    gc.collect()
    torch.cuda.empty_cache()
    return 结果


# ══════════════════════════════════════════════════
# 裁判②/③：云裁判（DeepSeek OpenAI 兼容接口）
# ══════════════════════════════════════════════════
def 云调用(client, 模型名, 提示, 思考开关="disabled", 重试=2):
    """单次 chat completion 调用；失败重试 重试 次，返回 (ok, 文本或错误)"""
    kwargs = {"model": 模型名, "messages": [{"role": "user", "content": 提示}],
              "max_tokens": 300, "temperature": 0.0}
    if 思考开关 is not None:
        kwargs["extra_body"] = {"thinking": {"type": 思考开关}}
    for 尝试 in range(重试 + 1):
        try:
            r = client.chat.completions.create(**kwargs)
            内容 = r.choices[0].message.content or ""
            return True, 内容.strip()
        except Exception as e:  # noqa: BLE001
            if 尝试 < 重试:
                time.sleep(3)
            else:
                return False, str(e)[:200]
    return False, "未知错误"


def 裁判_云(裁判键, 模型名, 模式列表, 随机样本, AI回复字典, 结果, 思考开关="disabled"):
    from openai import OpenAI
    client = OpenAI(api_key=云API_KEY, base_url=云BASE_URL)
    if 思考开关 == "enabled":
        裁判段 = 结果["裁判"].setdefault(f"{裁判键}_思考开",
                                        {"说明": f"{模型名} 思考开", "思考": "enabled"})
    else:
        裁判段 = 结果["裁判"].setdefault(裁判键, {"说明": f"{模型名} 思考关", "思考": "disabled"})
    裁判段.setdefault("模式", {})
    裁判段.setdefault("明细", {})
    裁判段.setdefault("失败样本", [])
    调用计数 = [0]

    for 模式 in 模式列表:
        if 裁判段["模式"].get(模式, {}).get("已完成条数", 0) >= len(随机样本):
            记录日志(f"[{裁判键}] 模式 [{模式}] 已完成({len(随机样本)}条)，跳过续跑")
            continue
        记录日志(f"──── 裁判[{裁判键}{'(思考开)' if 思考开关=='enabled' else ''}] 模式 [{模式}] ────")
        AI回复列表 = AI回复字典[模式]
        配对胜数 = 配对总数 = 0
        评分列表 = []
        明细 = []
        for i, r in enumerate(随机样本):
            用户, 真人 = r["user"], r["girl"]
            ai回复 = AI回复列表[i]
            # 配对 A（AI 在 A 位）
            okA, 输出A = 云调用(client, 模型名, 裁判配对提示.format(
                user=用户, 回复A=ai回复, 回复B=真人), 思考开关)
            # 配对 B（AI 在 B 位）
            okB, 输出B = 云调用(client, 模型名, 裁判配对提示.format(
                user=用户, 回复A=真人, 回复B=ai回复), 思考开关)
            # 评分
            okS, 评分文本 = 云调用(client, 模型名, 裁判评分提示.format(
                user=用户, 回复=ai回复), 思考开关)
            选择A = 解析配对(输出A) if okA else None
            选择B = 解析配对(输出B) if okB else None
            分 = 解析评分(评分文本) if okS else None
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
            if 分 is not None:
                评分列表.append(分)
            if not (okA and okB and okS):
                裁判段["失败样本"].append({"模式": 模式, "样本": i + 1,
                                          "okA": okA, "okB": okB, "okS": okS,
                                          "错误A": 输出A if not okA else None})
            明细.append({"样本": i + 1, "选择A": 选择A, "选择B": 选择B, "评分": 分})
            # 防限流：每 50 次 sleep 2s
            调用计数[0] += 3
            if 调用计数[0] >= 50:
                time.sleep(2)
                调用计数[0] = 0
            记录日志(f"[{裁判键} {模式} {i+1}/{len(随机样本)}] 配对(A:{选择A},B:{选择B}) AI评分={分}")
            # 每 5 条增量写盘（中断/限流后保留部分结果）
            if (i + 1) % 5 == 0 or i == len(随机样本) - 1:
                _s = 汇总(配对胜数, 配对总数, 评分列表)
                _s["已完成条数"] = i + 1
                裁判段["模式"][模式] = _s
                裁判段["明细"][模式] = 明细
                _写盘(结果)
        _s = 汇总(配对胜数, 配对总数, 评分列表)
        _s["已完成条数"] = len(随机样本)
        裁判段["模式"][模式] = _s
        裁判段["明细"][模式] = 明细
        记录日志(f"[{裁判键} {模式}] {json.dumps(_s, ensure_ascii=False)}")
        _写盘(结果)
    return 结果


def _写盘(结果):
    try:
        with open(结果路径, "w", encoding="utf-8") as f:
            json.dump(结果, f, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001
        记录日志(f"[写盘失败] {e}")


def 打印总表(结果):
    """输出 四模式 × 裁判 win_rate 表 + 判定"""
    记录日志("\n════════ 四模式 × 三裁判 win_rate 总表 ════════")
    for 裁判键, 段 in 结果["裁判"].items():
        if "模式" not in 段:
            continue
        记录日志(f"\n[{裁判键}] {段.get('说明', '')}")
        for 模式 in 模式顺序:
            if 模式 in 段["模式"]:
                m = 段["模式"][模式]
                记录日志(f"  {模式:4s} win_rate={m['win_rate_against_human']:.4f} "
                          f"rating={m['average_rating']:.4f} "
                          f"(配对{m['配对总数']} 评分{m['评分样本']})")
    # 判定：全开 vs 裸（各裁判独立看 ≥ +18%）
    记录日志("\n════════ 判定：全开 win_rate ≥ 裸+18%（各裁判独立） ════════")
    for 裁判键, 段 in 结果["裁判"].items():
        if "模式" not in 段 or "裸" not in 段["模式"] or "全开" not in 段["模式"]:
            continue
        v0 = 段["模式"]["裸"]["win_rate_against_human"]
        v1 = 段["模式"]["全开"]["win_rate_against_human"]
        Δ = v1 - v0
        达成 = Δ >= 0.18
        记录日志(f"[{裁判键}] 裸 {v0} → 全开 {v1} (Δ {Δ:+.4f}) "
                  f"→ {'✓ 达成(≥+18%)' if 达成 else '✗ 未达成'}")


def 主程序():
    ap = argparse.ArgumentParser()
    ap.add_argument("--裁判", choices=["本地7B", "flash", "pro"], required=True)
    ap.add_argument("--模式", default="全部", help="逗号分隔如 裸,锚点,回响,全开 或 全部")
    ap.add_argument("--思考开", action="store_true",
                    help="仅 pro：追加一组 思考开 对比（裸 vs 全开）")
    ap.add_argument("--汇总", action="store_true", help="只读最近结果文件输出总表")
    args = ap.parse_args()

    if args.汇总:
        if not os.path.exists(结果路径):
            print(f"无结果文件：{结果路径}")
            return
        with open(结果路径, encoding="utf-8") as f:
            _结果 = json.load(f)
        打印总表(_结果)
        return

    if not os.path.exists(生成文件):
        记录日志(f"生成缓存不存在：{生成文件}（先运行 运行四模式_锚点.py）")
        return
    with open(生成文件, encoding="utf-8") as f:
        生成数据 = json.load(f)
    随机样本 = [{"user": u, "girl": g} for u, g in
                zip(生成数据["样本"]["user"], 生成数据["样本"]["girl"])]
    AI回复字典 = {模式: 生成数据["模式"][模式]["回复"] for 模式 in 模式顺序
                 if 模式 in 生成数据["模式"]}

    模式列表 = 模式顺序 if args.模式 == "全部" else [m for m in 模式顺序
                                            if m in args.模式.split(",")]
    记录日志(f"=== Task8 步骤2 三裁判交叉验证 裁判={args.裁判} 模式={模式列表} "
              f"生成文件={os.path.basename(生成文件)} ===")
    记录日志(f"结果 -> {结果路径}")

    # 结果骨架（若同文件已存在则合并）
    结果 = {"任务": "Task8 三裁判交叉验证（控制变量：同回复/同配对/同提示词，只变裁判）",
            "时间戳": 时间戳, "生成文件": 生成文件, "裁判": {}}
    if os.path.exists(结果路径):
        try:
            with open(结果路径, encoding="utf-8") as f:
                旧 = json.load(f)
            结果["裁判"] = 旧.get("裁判", {})
        except Exception:  # noqa: BLE001
            pass

    if args.裁判 == "本地7B":
        if "本地7B" in 结果["裁判"]:
            裁判段 = 结果["裁判"]["本地7B"]
        else:
            裁判段 = {"说明": "Qwen2.5-7B-Instruct 手动加载 bf16 贪心(温度0.2)"}
        裁判段.setdefault("模式", {})
        裁判段.setdefault("明细", {})
        结果["裁判"]["本地7B"] = 裁判段
        裁判_本地7B(模式列表, 随机样本, AI回复字典, 结果)
    elif args.裁判 in ("flash", "pro"):
        模型名 = f"deepseek-v4-{args.裁判}"
        裁判_云(args.裁判, 模型名, 模式列表, 随机样本, AI回复字典, 结果,
               思考开关="disabled")
        if args.思考开 and args.裁判 == "pro":
            对比模式 = [m for m in ("裸", "全开") if m in AI回复字典]
            if 对比模式:
                裁判_云("pro", "deepseek-v4-pro", 对比模式, 随机样本, AI回复字典,
                       结果, 思考开关="enabled")
    _写盘(结果)
    打印总表(结果)
    记录日志(f"结果已保存 -> {结果路径}")


if __name__ == "__main__":
    主程序()
