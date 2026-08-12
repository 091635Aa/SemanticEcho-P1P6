# -*- coding: utf-8 -*-
import json, importlib.util
spec = importlib.util.spec_from_file_location("gen", r"f:\lora外挂\语义回响_系列\gen_series.py")
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
for ep in ['ep01_p1','ep02_etd','ep03_anchor','ep04_kv','ep05_ufd']:
    d = rf'f:\lora外挂\语义回响_系列\{ep}'
    script = json.load(open(d + r'\script.json', encoding='utf-8-sig'))
    tl = json.load(open(d + r'\timeline.json', encoding='utf-8'))
    doc, total = gen.gen_html(d, script['scenes'], tl, script['accent'], script['accent2'], script['title'])
    open(d + r'\index.html', 'w', encoding='utf-8').write(doc)
    print(ep, 'scenes', len(script['scenes']), 'total~', round(total,1), 's')
