# -*- coding: utf-8 -*-
"""诊断 v20：修复 rotary buffer 后 1.5B 对照 + 7B 推理验证"""
import gc
import glob
import os
import time
import torch

模型空间 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间"
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open


def 修复rotary(模型, cfg):
    base = getattr(cfg, "rope_theta", 1000000.0)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    inv = inv.to(torch.float32)
    修数量 = 0
    for module in 模型.modules():
        if hasattr(module, "inv_freq") and module.inv_freq is not None:
            module.inv_freq.copy_(inv)
            if hasattr(module, "original_inv_freq") and module.original_inv_freq is not None:
                module.original_inv_freq.copy_(inv)
            修数量 += 1
    print(f"  修复 rotary 模块数 = {修数量}", flush=True)


def 手动加载(路径):
    cfg = AutoConfig.from_pretrained(路径, trust_remote_code=True)
    with torch.device("meta"):
        模型 = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    模型 = 模型.to_empty(device="cuda")
    for sh in sorted(glob.glob(os.path.join(路径, "model-*.safetensors"))):
        with safe_open(sh, framework="pt", device="cpu") as f:
            _sd = {k: f.get_tensor(k) for k in f.keys()}
        模型.load_state_dict(_sd, strict=False)
        del _sd
        gc.collect()
    修复rotary(模型, cfg)
    模型.eval()
    return 模型


# ── 1.5B 对照 ──
小模型 = os.path.join(模型空间, "Qwen2.5-1.5B-Instruct")
分词器 = AutoTokenizer.from_pretrained(小模型, trust_remote_code=True)
标准 = AutoModelForCausalLM.from_pretrained(
    小模型, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True).to("cuda")
标准.eval()
手动 = 手动加载(小模型)

提示 = 分词器.apply_chat_template([{"role": "user", "content": "你好"}],
                                tokenize=False, add_generation_prompt=True)
inputs = 分词器(提示, return_tensors="pt").to("cuda")
with torch.no_grad():
    l1 = 标准(inputs.input_ids).logits[:, -1, :]
    l2 = 手动(inputs.input_ids).logits[:, -1, :]
top1 = torch.topk(l1, k=1, dim=-1).indices[0, 0].item()
top2 = torch.topk(l2, k=1, dim=-1).indices[0, 0].item()
print(f"1.5B 标准 top1={top1}{分词器.decode([top1])!r} 手动 top1={top2}{分词器.decode([top2])!r} 最大差异={(l1-l2).abs().max().item():.6f}", flush=True)

# ── 7B 推理验证 ──
del 标准, 手动
gc.collect()
torch.cuda.empty_cache()

裁判路径 = os.path.join(模型空间, "Qwen2.5-7B-Instruct")
分词器7 = AutoTokenizer.from_pretrained(裁判路径, trust_remote_code=True)
t0 = time.time()
裁判 = 手动加载(裁判路径)
print(f"7B 加载 {time.time()-t0:.1f}s 显存={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

提示 = 分词器7.apply_chat_template([{"role": "user", "content": "你好"}],
                                 tokenize=False, add_generation_prompt=True)
inputs = 分词器7(提示, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = 裁判.generate(inputs.input_ids, max_new_tokens=40, do_sample=False, pad_token_id=分词器7.eos_token_id)
新 = out[0, inputs.input_ids.shape[1]:]
print(f"7B 输出: {分词器7.decode(新, skip_special_tokens=True)[:60]!r}", flush=True)
