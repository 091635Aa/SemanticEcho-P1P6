# -*- coding: utf-8 -*-
"""
步骤0：云裁判连通性探测（Task8 三正交叠加验证）
================================================
- 两个模型（deepseek-v4-flash / deepseek-v4-pro）各发 1 条简单 chat completion，
  确认 OpenAI 兼容接口连通与返回格式；
- 探测"思考"开关：尝试 extra_body={"thinking": ...} 若干写法 + 观察返回消息对象
  是否含推理字段（reasoning_content / reasoning / thinking_content / thinking），
  记录哪种调用方式能得到"开/关思考"两种行为；API 不支持则记录"不支持"。
- 结果写入 h:\锚点回响（Anchor Echo）\云裁判探测结果.json
"""
import datetime
import json
from openai import OpenAI

BASE_URL = "https://api.deepseek.com"
# ⚠️ 仅本地脚本使用，绝不推送任何 GitHub 仓库；报告用 sk-**** 脱敏
API_KEY = "sk-ecd61cf28d244628837f7f87d0d8a96f"
输出路径 = r"h:\锚点回响（Anchor Echo）\云裁判探测结果.json"

思考开关尝试 = [
    ("thinking_enabled", {"thinking": {"type": "enabled"}}),
    ("thinking_disabled", {"thinking": {"type": "disabled"}}),
    ("thinking_bool_true", {"thinking": True}),
    ("thinking_bool_false", {"thinking": False}),
    ("reasoning_effort_low", {"reasoning_effort": "low"}),
]

推理字段名 = ("reasoning_content", "reasoning", "thinking_content", "thinking")


def 抓推理字段(消息):
    """返回消息对象上可见的推理字段内容（截断）"""
    字段 = {}
    for attr in 推理字段名:
        try:
            v = getattr(消息, attr, None)
            if v is not None:
                字段[attr] = str(v)[:300]
        except Exception:  # noqa: BLE001
            pass
    return 字段 or "无"


def 探测():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    结果 = {"时间戳": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "说明": "云裁判（DeepSeek OpenAI 兼容接口）连通性与思考开关探测",
            "模型": {}}
    for 模型名 in ("deepseek-v4-flash", "deepseek-v4-pro"):
        m = {}
        # ── 1. 基础连通（1 条简单调用）──
        try:
            r = client.chat.completions.create(
                model=模型名,
                messages=[{"role": "user", "content": "只回复两个字：你好"}],
                max_tokens=64, temperature=0.0,
            )
            ch = r.choices[0]
            m["基础连通"] = {
                "ok": True,
                "回复": ch.message.content,
                "finish_reason": ch.finish_reason,
                "usage": r.usage.model_dump() if r.usage else None,
                "推理字段": 抓推理字段(ch.message),
            }
            try:
                m["消息对象键"] = list(ch.message.model_dump().keys())
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            m["基础连通"] = {"ok": False, "错误": str(e)[:400]}
        # ── 2. 思考开关探测（多种写法逐个尝试）──
        for 标签, body in 思考开关尝试:
            try:
                r2 = client.chat.completions.create(
                    model=模型名,
                    messages=[{"role": "user", "content": "用一句话解释什么是爱。"}],
                    max_tokens=64, temperature=0.0,
                    extra_body=body,
                )
                ch2 = r2.choices[0]
                m[标签] = {"ok": True, "回复": ch2.message.content,
                           "finish_reason": ch2.finish_reason,
                           "usage": r2.usage.model_dump() if r2.usage else None,
                           "推理字段": 抓推理字段(ch2.message)}
            except Exception as e:  # noqa: BLE001
                m[标签] = {"ok": False, "错误": str(e)[:400]}
        结果["模型"][模型名] = m
    return 结果


if __name__ == "__main__":
    r = 探测()
    with open(输出路径, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print(f"\n已保存 -> {输出路径}")
