# -*- coding: utf-8 -*-
"""诊断：7B 裁判手动加载 0xC0000005 崩溃点定位（用完即删）"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import gc
import glob as _glob
import torch

print("step0 start", flush=True)
裁判路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-7B-Instruct"
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors import safe_open

print("step1 tokenizer", flush=True)
分词器 = AutoTokenizer.from_pretrained(裁判路径, trust_remote_code=True)
print("step2 config", flush=True)
cfg = AutoConfig.from_pretrained(裁判路径, trust_remote_code=True)
print("step3 meta model", flush=True)
with torch.device("meta"):
    模型 = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
print("step4 to_empty cuda", flush=True)
模型 = 模型.to_empty(device="cuda")
torch.cuda.empty_cache()
print("gpu after to_empty GB:", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
print("step5 load tensors", flush=True)
for _分片 in sorted(_glob.glob(os.path.join(裁判路径, "model-*.safetensors"))):
    print("  shard", os.path.basename(_分片), flush=True)
    with safe_open(_分片, framework="pt", device="cpu") as f:
        for _k in f.keys():
            _t = f.get_tensor(_k)
            模型.load_state_dict({_k: _t}, strict=False)
            del _t
    gc.collect()
    torch.cuda.empty_cache()
print("step6 inv_freq fix", flush=True)
_base = getattr(cfg, "rope_theta", 1000000.0)
_头维 = cfg.hidden_size // cfg.num_attention_heads
_inv = 1.0 / (_base ** (torch.arange(0, _头维, 2, dtype=torch.int64).float() / _头维))
_inv = _inv.to(torch.float32)
for _模块 in 模型.modules():
    if hasattr(_模块, "inv_freq") and _模块.inv_freq is not None:
        _模块.inv_freq.copy_(_inv)
        if hasattr(_模块, "original_inv_freq") and _模块.original_inv_freq is not None:
            _模块.original_inv_freq.copy_(_inv)
print("step7 DONE ok", flush=True)
