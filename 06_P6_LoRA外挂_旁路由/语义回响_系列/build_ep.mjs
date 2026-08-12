// build_ep.mjs — 拼接 + 混音单集：node build_ep.mjs <ep_dir>
import { execFile } from "node:child_process";
import { readdirSync, writeFileSync, unlinkSync, mkdirSync, readFileSync } from "node:fs";
import { join, basename } from "node:path";
const FFMPEG = "D:\\FormatFactory\\ffmpeg.exe";
const EP = process.argv[2];
const SEG_DIR = EP + "\\segments";
mkdirSync(SEG_DIR, { recursive: true });
const OUT_FINAL = EP + "\\" + basename(EP) + "_final.mp4";
// 构建完整音轨
const tl = JSON.parse(readFileSync(EP + "\\timeline.json", "utf-8"));
const script = JSON.parse(readFileSync(EP + "\\script.json", "utf-8"));
let t = 0.6;
const segs = script.scenes.map((sc, i) => {
  const dur = tl["s" + String(i).padStart(2, "0")] || 8;
  const s = { i, audio_start: t + 0.4, dur };
  t += 0.4 + dur + 0.8;
  return s;
});
const DUR = t;
// 用 ffmpeg 合成 narration（每段 adelay + amix）
const inputs = [];
const delays = [];
segs.forEach((s, i) => {
  const f = `${EP}\\audio\\s${String(s.i).padStart(2, "0")}.wav`;
  inputs.push("-i", f);
  delays.push(`[${i}]adelay=${Math.round(s.audio_start*1000)}|${Math.round(s.audio_start*1000)}[d${i}]`);
});
const mixInputs = segs.map((s,i) => `[d${i}]`).join("");
const totalFrames = Math.round(DUR * 24000);
const filter = `${delays.join(";")};${mixInputs}amix=inputs=${segs.length}:normalize=0,apad=pad_dur=${DUR}[a]`;
const audioOut = EP + "\\narration_full.wav";
await new Promise((res, rej) => execFile(FFMPEG, ["-y", ...inputs, "-filter_complex", filter, "-map", "[a]", "-ar", "24000", "-ac", "1", audioOut], {maxBuffer: 20*1024*1024}, (err, so, se) => err ? rej(new Error(se||err.message)) : res()));
console.log("audio ->", audioOut, "dur", DUR.toFixed(1));

const segsList = readdirSync(SEG_DIR).filter(f => /^seg_\d+\.mp4$/.test(f)).sort((a,b)=>parseInt(a.split("_")[1])-parseInt(b.split("_")[1]));
if (!segsList.length) { console.error("no segments"); process.exit(1); }
const listFile = join(SEG_DIR, "concat.txt");
writeFileSync(listFile, segsList.map(s => `file '${join(SEG_DIR, s).replace(/\\/g, "/")}'`).join("\n"));
const silent = EP + "\\_silent.mp4";
await new Promise((res, rej) => execFile(FFMPEG, ["-y","-f","concat","-safe","0","-i",listFile,"-c","copy",silent], {maxBuffer:10*1024*1024}, (err, so, se) => err ? rej(new Error(se||err.message)) : res()));
console.log("concat ->", silent);
await new Promise((res, rej) => execFile(FFMPEG, ["-y","-i",silent,"-i",audioOut,"-c:v","copy","-c:a","aac","-b:a","192k","-shortest","-movflags","+faststart",OUT_FINAL], {maxBuffer:10*1024*1024}, (err, so, se) => err ? rej(new Error(se||err.message)) : res()));
unlinkSync(listFile); unlinkSync(silent);
console.log("final ->", OUT_FINAL);
