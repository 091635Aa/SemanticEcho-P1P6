# -*- coding: utf-8 -*-
import wave, array, os
SR = 24000; DUR = 95.05
SEG_DIR = r"f:\lora外挂\语义回响_海外宣传\audio_new"
OUT = r"f:\lora外挂\语义回响_海外宣传\audio_new\narration_full.wav"
OFFSETS = {"s00":0.0,"s01":10.323,"s02":23.748,"s03":32.294,"s04":42.440,
           "s05":51.228,"s06":59.473,"s07":67.596,"s08":76.880,"s09":85.491}
total = int(DUR*SR)
buf = array.array("h",[0])*total
for name,off in OFFSETS.items():
    p=os.path.join(SEG_DIR,name+".wav")
    w=wave.open(p,"rb"); data=w.readframes(w.getnframes()); w.close()
    frames=array.array("h"); frames.frombytes(data)
    s=int(off*SR); e=min(s+len(frames),total)
    buf[s:e]=frames[:e-s]
out=wave.open(OUT,"wb"); out.setnchannels(1); out.setsampwidth(2); out.setframerate(SR)
out.writeframes(buf.tobytes()); out.close()
print("narration_full.wav", round(total/SR,2),"s")
