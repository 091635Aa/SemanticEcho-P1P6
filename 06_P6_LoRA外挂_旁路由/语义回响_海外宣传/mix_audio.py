# -*- coding: utf-8 -*-
import wave, array, os, struct
SR = 24000
DUR = 95.05
SEG_DIR = r"f:\lora外挂\语义回响_海外宣传\audio"
OUT = r"f:\lora外挂\语义回响_海外宣传\audio\narration_full.wav"
OFFSETS = {
  "s00": 0.0, "s01": 10.323, "s02": 23.748, "s03": 32.294, "s04": 42.440,
  "s05": 51.228, "s06": 59.473, "s07": 67.596, "s08": 76.880, "s09": 85.491,
}
total_frames = int(DUR * SR)
buf = array.array("h", [0]) * total_frames
for name, off in OFFSETS.items():
    p = os.path.join(SEG_DIR, name + ".wav")
    w = wave.open(p, "rb")
    assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, (name, w.getframerate(), w.getnchannels(), w.getsampwidth())
    data = w.readframes(w.getnframes())
    w.close()
    frames = array.array("h")
    frames.frombytes(data)
    start = int(off * SR)
    end = min(start + len(frames), total_frames)
    buf[start:end] = frames[:end - start]
    print(name, "placed at", off, "len", round(len(frames)/SR,2))
out = wave.open(OUT, "wb")
out.setnchannels(1); out.setsampwidth(2); out.setframerate(SR)
out.writeframes(buf.tobytes())
out.close()
print("narration_full.wav", round(total_frames/SR,2), "s")
