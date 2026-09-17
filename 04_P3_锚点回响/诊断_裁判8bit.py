# -*- coding: utf-8 -*-
"""诊断：7B 裁判 8bit 加载可行性（bitsandbytes）"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import gc
import torch

print("step0 start", flush=True)
裁判路径 = r"c:\Users\Administrator\Documents\论文+临时目录\模型空间\Qwen2.5-7B-Instruct"
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("step1 tokenizer", flush=True)
分词器 = AutoTokenizer.from_pretrained(裁判路径, trust_remote_code=True)

print("step2 8bit from_pretrained", flush=True)
量化配置 = BitsAndBytesConfig(load_in_8bit=True)
模型 = AutoModelForCausalLM.from_pretrained(
    裁判路径, quantization_config=量化配置, device_map="auto",
    low_cpu_mem_usage=True, trust_remote_code=True)
模型.eval()
print("gpu used GB:", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
print("step3 inference test", flush=True)
提示 = 分词器.apply_chat_template(
    [{"role": "user", "content": "请回复：你好"}], tokenize=False, add_generation_prompt=True)
inputs = 分词器(提示, return_tensors="pt").to(模型.device)
with torch.no_grad():
    out = 模型.generate(inputs.input_ids, max_new_tokens=20, temperature=0.2, do_sample=False,
                        pad_token_id=分词器.eos_token_id)
print("回复:", 分词器.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True), flush=True)
print("step4 DONE ok", flush=True)
