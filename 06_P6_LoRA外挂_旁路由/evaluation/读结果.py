# -*- coding: utf-8 -*-
"""读取 LoRA 测试 JSON 汇总"""
import json
import sys

路径 = sys.argv[1] if len(sys.argv) > 1 else r"f:\lora外挂\evaluation\emotion_emotion_v1_test_3B.json"
d = json.load(open(路径, encoding="utf-8"))
for k, v in d.items():
    print(f"{k}: 熵={v['平均熵']:.3f} 重复={v['平均重复率']:.3f} 温柔命中={v['温柔命中率']:.2f}")
