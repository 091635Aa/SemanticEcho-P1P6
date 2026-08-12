// build_final.mjs — 拼接所有 segments + 混音 → 最终视频
import { execFile } from "node:child_process";
import { readdirSync, writeFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";

const FFMPEG = "D:\\FormatFactory\\ffmpeg.exe";
const SEG_DIR = "F:\\lora外挂\\语义回响_海外宣传\\segments";
const AUDIO = "F:\\lora外挂\\语义回响_海外宣传\\audio_new\\narration_full.wav";
const OUT_SILENT = "F:\\lora外挂\\语义回响_海外宣传\\_full_silent.mp4";
const OUT_FINAL = "F:\\lora外挂\\语义回响_海外宣传\\semantic_echo_p1p6_final.mp4";

// 1) 按顺序列出 seg_*.mp4
const segs = readdirSync(SEG_DIR)
  .filter(f => /^seg_\d+_.+\.mp4$/.test(f))
  .sort((a, b) => parseInt(a.split("_")[1]) - parseInt(b.split("_")[1]));
if (!segs.length) { console.error("no segments found in", SEG_DIR); process.exit(1); }
console.log("segments:", segs.join(" | "));

// 2) concat list
const listFile = join(SEG_DIR, "concat_list.txt");
writeFileSync(listFile, segs.map(s => `file '${join(SEG_DIR, s).replace(/\\/g, "/")}'`).join("\n"));

await new Promise((res, rej) => {
  execFile(FFMPEG, ["-y","-f","concat","-safe","0","-i",listFile,"-c","copy",OUT_SILENT],
    { maxBuffer: 10*1024*1024 }, (err, so, se) => err ? rej(new Error(se || err.message)) : res());
});
console.log("concat done ->", OUT_SILENT);

// 3) 混音
await new Promise((res, rej) => {
  execFile(FFMPEG, ["-y","-i",OUT_SILENT,"-i",AUDIO,"-c:v","copy","-c:a","aac","-b:a","192k",
    "-shortest","-movflags","+faststart",OUT_FINAL],
    { maxBuffer: 10*1024*1024 }, (err, so, se) => err ? rej(new Error(se || err.message)) : res());
});
console.log("final ->", OUT_FINAL);
unlinkSync(listFile);
unlinkSync(OUT_SILENT);

