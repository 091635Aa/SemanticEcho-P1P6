# -*- coding: utf-8 -*-
"""gen_series.py — 读 script.json → 合成英文旁白 TTS → 测时长 → 生成讲解视频 index.html
用法: python gen_series.py <ep_dir> [--tts-only]
"""
import os, sys, json, wave, time, struct, html

API_KEY = "sk-ws-H.EEDIDIL.OYKE.MEYCIQDKuZ5azx435v_mBTe-z_ydoIxWdOFouB9He5dyco7E5gIhAPRXUZ_Ow-DdSaaw7GWj1g87tmqSX29LCCpa4sMKWjCB"
TTS_URL = "https://ws-ee3sqayhxsekscra.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
VOICE = "longanhuan_v3.6"

def fix_wav(path):
    data = bytearray(open(path, "rb").read())
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return
    i = 12; off = None
    while i < len(data) - 8:
        cid = bytes(data[i:i+4]); sz = struct.unpack("<I", bytes(data[i+4:i+8]))[0]
        if cid == b"data": off = i + 8; break
        i += 8 + sz + (sz % 2)
    if off is None: return
    struct.pack_into("<I", data, 4, len(data) - 8)
    struct.pack_into("<I", data, off - 4, len(data) - off)
    open(path, "wb").write(bytes(data))

def synth(text, out, retries=4, speed=1.15):
    import requests
    body = {"model": "qwen-audio-3.0-tts-plus",
            "input": {"text": text, "voice": VOICE, "format": "wav", "sample_rate": 24000, "speed": speed}}
    for a in range(retries):
        try:
            r = requests.post(TTS_URL, json=body,
                headers={"Authorization": "Bearer "+API_KEY, "Content-Type": "application/json"}, timeout=120)
            if r.status_code != 200:
                print("  [tts] HTTP", r.status_code, r.text[:120]); time.sleep(2); continue
            url = r.json()["output"]["audio"]["url"]
            ar = requests.get(url, timeout=120)
            if ar.status_code != 200:
                print("  [tts] dl", ar.status_code); time.sleep(2); continue
            open(out, "wb").write(ar.content); fix_wav(out)
            return True
        except Exception as e:
            print("  [tts] err", str(e)[:100]); time.sleep(2)
    return False

def wav_dur(p):
    try:
        w = wave.open(p, "rb"); d = w.getnframes()/w.getframerate(); w.close(); return d
    except Exception:
        return None

def esc(s):
    return html.escape(s, quote=False)

def gen_html(ep, scenes, timeline, accent, accent2, title):
    """生成讲解视频 HTML（深色科技风 + 双语字幕条）"""
    # 计算每场景时间轴（audio 起点 + 0.4s 前导 + 0.6s 尾部缓冲）
    segs = []
    t = 0.6
    for i, sc in enumerate(scenes):
        dur = timeline.get(f"s{i:02d}", 8.0)
        segs.append({"idx": i, "start": t, "audio_start": t + 0.4, "dur": dur, "end": t + 0.4 + dur + 0.8, "sc": sc})
        t = segs[-1]["end"]
    total = t
    # 每场景显示时长 = 音频 + 1.2s 缓冲
    acts = []
    for s in segs:
        sc = s["sc"]
        kicker = esc(sc.get("kicker", ""))
        title_ = esc(sc.get("title", ""))
        sub = esc(sc.get("sub", ""))
        points = sc.get("points", [])
        pts = "".join(f'<div class="pt"><span class="dot" style="background:{accent};"></span><span>{esc(p)}</span></div>' for p in points)
        en = esc(sc.get("en", ""))
        zh = esc(sc.get("zh", ""))
        acts.append(f'''
  <div class="act clip" id="act{s['idx']}" data-start="{s['start']:.3f}" data-duration="{s['end']-s['start']:.3f}" data-track-index="{s['idx']}">
    <div class="glow" id="gl{s['idx']}" style="left:120px;top:140px;width:520px;height:520px;background:{accent};"></div>
    <div class="kicker" id="k{s['idx']}">{kicker}</div>
    <div class="big-title" id="t{s['idx']}">{title_}</div>
    {f'<div class="sub" id="su{s["idx"]}">{sub}</div>' if sub else ''}
    <div class="pts" id="p{s['idx']}">{pts}</div>
    <div class="scene-num" id="n{s['idx']}">{s['idx']+1:02d} / {len(scenes):02d}</div>
  </div>''')

    cap_items = "".join(
        f'tl.call(()=>{{const e=document.getElementById("cap-en");e.innerHTML={json.dumps(esc(s["sc"].get("en","")))};'
        f'document.getElementById("cap-zh").innerHTML={json.dumps(esc(s["sc"].get("zh","")))};}},[],{s["audio_start"]+0.1:.3f});'
        for s in segs)
    # 场景转场
    act_ids = "".join(f'  tl.set("#act{i}", {{clipPath:"circle(0% at 50% 50%)"}}, 0);\n' for i in range(1, len(scenes)))
    transitions = ""
    for s in segs[1:]:
        transitions += f'  tl.fromTo("#act{s["idx"]}", {{clipPath:"circle(0% at 50% 50%)"}}, {{clipPath:"circle(150% at 50% 50%)",duration:0.7,ease:"power3.inOut"}}, {s["start"]-0.4:.3f});\n'
    # 每场景元素入场
    entries = ""
    for s in segs:
        st = s["audio_start"]
        i = s["idx"]
        entries += f'''  tl.fromTo("#k{i}", {{y:-30,opacity:0}}, {{y:0,opacity:1,duration:0.5,ease:"power3.out"}}, {st:.3f});
  tl.fromTo("#t{i}", {{y:40,opacity:0}}, {{y:0,opacity:1,duration:0.6,ease:"power4.out"}}, {st+0.15:.3f});
  tl.fromTo("#p{i}", {{y:30,opacity:0}}, {{y:0,opacity:1,duration:0.5,ease:"power3.out"}}, {st+0.5:.3f});
'''
        if f"su{i}" in "".join(a for a in acts) if False else True:
            pass
        # sub 入场（如果存在）
    for s in segs:
        if s["sc"].get("sub"):
            entries += f'  tl.fromTo("#su{s["idx"]}", {{y:24,opacity:0}}, {{y:0,opacity:1,duration:0.5,ease:"power3.out"}}, {s["audio_start"]+0.35:.3f});\n'

    html_doc = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{esc(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html,body {{ width:100%; height:100%; background:#000; overflow:hidden;
    font-family:"Space Grotesk","Inter","Segoe UI",system-ui,sans-serif; }}
  .composition {{ position:relative; width:1920px; height:1080px; overflow:hidden; background:#04060d; }}
  #topbar {{ position:absolute; top:0; left:0; right:0; height:8px; background:rgba(255,255,255,0.06); z-index:90; }}
  #topbar-fill {{ height:100%; width:0%; background:linear-gradient(90deg,{accent},{accent2},#FF5D8F); }}
  #brand {{ position:absolute; top:40px; left:60px; z-index:90; font-weight:700; letter-spacing:4px; font-size:24px; color:rgba(255,255,255,0.85); }}
  #brand span {{ color:{accent}; }}
  #series-tag {{ position:absolute; top:44px; right:60px; z-index:90; font-size:20px; color:rgba(255,255,255,0.4); letter-spacing:2px; }}
  .act {{ position:absolute; inset:0; background:#04060d; }}
  .glow {{ position:absolute; border-radius:50%; filter:blur(120px); opacity:0.4; }}
  .kicker {{ position:absolute; left:140px; top:120px; font-size:28px; letter-spacing:8px; font-weight:700;
    text-transform:uppercase; color:{accent}; }}
  .big-title {{ position:absolute; left:140px; top:200px; width:1640px; font-size:84px; font-weight:800;
    line-height:1.08; color:#fff; }}
  .sub {{ position:absolute; left:140px; top:360px; width:1600px; font-size:38px; color:rgba(255,255,255,0.6); }}
  .pts {{ position:absolute; left:140px; top:430px; width:1600px; }}
  .pt {{ display:flex; align-items:center; gap:22px; margin-top:28px; font-size:38px; font-weight:600; color:rgba(255,255,255,0.88); }}
  .dot {{ width:16px; height:16px; border-radius:50%; flex-shrink:0; box-shadow:0 0 18px currentColor; }}
  .scene-num {{ position:absolute; left:140px; top:900px; font-size:22px; color:rgba(255,255,255,0.35); letter-spacing:3px; }}
  #cap {{ position:absolute; left:60px; right:60px; bottom:60px; z-index:80; min-height:120px;
    border-left:4px solid {accent}; padding-left:26px; background:linear-gradient(180deg,rgba(4,6,13,0.85),rgba(4,6,13,0.98));
    border-top:1px solid rgba(255,255,255,0.06); border-right:1px solid rgba(255,255,255,0.06);
    border-bottom:1px solid rgba(255,255,255,0.06); border-radius:0 16px 16px 0; }}
  #cap-zh {{ font-size:38px; font-weight:700; color:#fff; line-height:1.4; }}
  #cap-en {{ margin-top:6px; font-size:22px; color:rgba(255,255,255,0.55); line-height:1.4; }}
</style>
</head>
<body>
<div class="composition" data-composition-id="promo-video" data-duration="{total:.2f}">
  <div id="topbar"><div id="topbar-fill"></div></div>
  <div id="brand">SEMANTIC<span>&nbsp;ECHO</span></div>
  <div id="series-tag">{esc(title)}</div>
{''.join(acts)}
  <div id="cap"><div id="cap-zh"></div><div id="cap-en"></div></div>
</div>
<script>
  window.__timelines = window.__timelines || {{}};
  const tl = gsap.timeline({{paused:true}});
  const DUR = {total:.2f};
  tl.to("#topbar-fill", {{width:"100%",duration:DUR,ease:"none"}}, 0);
{cap_items}
{act_ids}
{transitions}
{entries}
  window.__timelines["promo-video"] = tl;
</script>
</body>
</html>'''
    return html_doc, total

def main():
    ep_dir = sys.argv[1]
    tts_only = "--tts-only" in sys.argv
    with open(os.path.join(ep_dir, "script.json"), "r", encoding="utf-8-sig") as f:
        script = json.load(f)
    audio_dir = os.path.join(ep_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    scenes = script["scenes"]
    print(f"[gen] {script['id']} {len(scenes)} scenes")

    # TTS
    timeline = {}
    for i, sc in enumerate(scenes):
        out = os.path.join(audio_dir, f"s{i:02d}.wav")
        if not os.path.exists(out):
            ok = synth(sc["en"], out)
            print(f"  s{i:02d} {'OK' if ok else 'FAIL'}")
        d = wav_dur(out)
        if d is None:
            d = 0.0
        timeline[f"s{i:02d}"] = d if d else 8.0
        print(f"    dur={d:.2f}s")
    with open(os.path.join(ep_dir, "timeline.json"), "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)
    print(f"[gen] total audio: {sum(timeline.values()):.1f}s")

    if tts_only:
        return

    # HTML
    doc, total = gen_html(ep_dir, scenes, timeline, script["accent"], script["accent2"], script["title"])
    with open(os.path.join(ep_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[gen] index.html written, video duration ~{total:.1f}s")

if __name__ == "__main__":
    main()
