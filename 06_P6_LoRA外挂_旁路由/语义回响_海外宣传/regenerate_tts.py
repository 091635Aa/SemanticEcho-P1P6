# -*- coding: utf-8 -*-
import os, json, wave, requests, time, struct, subprocess
API_KEY = "sk-ws-H.EEDIDIL.OYKE.MEYCIQDKuZ5azx435v_mBTe-z_ydoIxWdOFouB9He5dyco7E5gIhAPRXUZ_Ow-DdSaaw7GWj1g87tmqSX29LCCpa4sMKWjCB"
URL = "https://ws-ee3sqayhxsekscra.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
FF = r"D:\FormatFactory\ffmpeg.exe"
OUT = r"f:\lora外挂\语义回响_海外宣传\audio\segs_v2"
os.makedirs(OUT, exist_ok=True)

def fix_wav(path):
    data = bytearray(open(path, "rb").read())
    i = 12; off = None
    while i < len(data) - 8:
        cid = bytes(data[i:i+4]); sz = struct.unpack("<I", bytes(data[i+4:i+8]))[0]
        if cid == b"data": off = i + 8; break
        i += 8 + sz + (sz % 2)
    if off is None: return
    struct.pack_into("<I", data, 4, len(data) - 8)
    struct.pack_into("<I", data, off - 4, len(data) - off)
    open(path, "wb").write(bytes(data))

def synth(text, out, retries=3):
    body = {"model": "qwen-audio-3.0-tts-plus",
            "input": {"text": text, "voice": "longanhuan_v3.6", "format": "wav", "sample_rate": 24000}}
    for a in range(retries):
        try:
            r = requests.post(URL, json=body, headers={"Authorization": "Bearer "+API_KEY, "Content-Type": "application/json"}, timeout=120)
            if r.status_code != 200: print("  HTTP", r.status_code, r.text[:100]); time.sleep(2); continue
            url = r.json()["output"]["audio"]["url"]
            ar = requests.get(url, timeout=120)
            if ar.status_code != 200: print("  dl", ar.status_code); time.sleep(2); continue
            open(out, "wb").write(ar.content); fix_wav(out); return True
        except Exception as e: print("  err", e); time.sleep(2)
    return False

# 目标时长 = 旧 segs_90 各段时长（保持时间轴不变）
TARGETS = {"s00":9.623,"s01":12.725,"s02":7.846,"s03":9.446,"s04":8.088,"s05":7.545,"s06":7.423,"s07":8.584,"s08":7.911,"s09":8.660}

# 英文拼写版本（P 编号/数字全部用英文单词）
SEGS = {
"s00": "What if a one point five billion parameter model could sound more human than models a thousand times bigger? No fine-tuning, no retraining. Just decoding-time magic.",
"s01": "Meet Semantic Echo. A family of decoding-time architectures that make AI emotionally expressive, without touching a single weight. P one captures hidden states the model would throw away, and feeds them back as emotion. Entropy up forty five percent.",
"s02": "P one point five auto-tunes injection strength for any model, any size, any quantization. Qwen, Phi, Gemma, DeepSeek. Plug in and go.",
"s03": "P two point five, Emotion Tidal Decoding, measures the emotional tide of the conversation and re-weights the sampling distribution. Blind judges: ninety one percent more human.",
"s04": "P three, Anchor Echo, scores every token against emotional anchors, read-only. TuringBench human likeness? Plus two hundred percent.",
"s05": "P four, KV Resonance, finds emotional tokens inside the attention cache and makes the model actually look at them. Near zero memory cost.",
"s06": "P five, the Ultra Fusion Decoder, fuses all four into one pipeline. Four architectures, one chain. No collapse.",
"s07": "The honest part: a small base model has a ceiling. Steering narrows the gap but can't break it. And these gains are about emotion, not general intelligence.",
"s08": "So build companions that feel, game characters that remember moods, private AI with a human touch on your own GPU. No cloud.",
"s09": "Everything is open source. Semantic Echo, ETD, Anchor Echo, KV Resonance, and the home hub, linked below. One small model, one big idea.",
}

results = {}
for name, text in SEGS.items():
    raw = os.path.join(OUT, name + "_raw.wav")
    if not synth(text, raw):
        print(name, "SYNTH FAIL"); continue
    w = wave.open(raw, "rb"); dur = round(w.getnframes()/w.getframerate(), 3); w.close()
    target = TARGETS[name]
    factor = dur / target if target else 1.0
    out = os.path.join(OUT, name + ".wav")
    p = subprocess.run([FF,"-y","-v","error","-i",raw,"-filter:a",f"atempo={factor:.4f}","-ar","24000","-ac","1",out])
    if p.returncode != 0: print(name, "ATEMPO FAIL"); continue
    os.remove(raw)
    w = wave.open(out,"rb"); ndur = round(w.getnframes()/w.getframerate(),3); w.close()
    results[name] = {"text": text, "dur": ndur, "file": out}
    print(f"{name} raw={dur}s target={target}s atempo={factor:.3f} -> {ndur}s")
with open(os.path.join(OUT, "timeline.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("total:", round(sum(v["dur"] for v in results.values()),2))
