// render_segment.mjs — 片段化渲染：node render_segment.mjs [索引] | all
// 每个片段自带头部转场（起点 = act_start - 0.45），终点 = 下一幕转场前。拼接后无缝。
import puppeteer from "puppeteer-core";
import { execFile } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const FFMPEG = "D:\\FormatFactory\\ffmpeg.exe";
const COMPOSITION = "file:///F:/lora外挂/语义回响_海外宣传/index.html";
const SEG_DIR = "F:\\lora外挂\\语义回响_海外宣传\\segments";
const WIDTH = 1920, HEIGHT = 1080, DUR = 95.05, FPS = 30;

// 片段定义：起点含转场(-0.45)，终点为下一幕转场前。与 index.html ACTS 一致。
const SEGMENTS = [
  { i: 0,  name: "s00_hook",   start: 0.0,     end: 9.873 },
  { i: 1,  name: "s01_p1",     start: 9.873,   end: 23.298 },
  { i: 2,  name: "s02_p15",    start: 23.298,  end: 31.844 },
  { i: 3,  name: "s03_p25_etd",start: 31.844,  end: 41.990 },
  { i: 4,  name: "s04_p3",     start: 41.990,  end: 50.778 },
  { i: 5,  name: "s05_p4",     start: 50.778,  end: 59.023 },
  { i: 6,  name: "s06_p5",     start: 59.023,  end: 67.146 },
  { i: 7,  name: "s07_honest", start: 67.146,  end: 76.430 },
  { i: 8,  name: "s08_use",    start: 76.430,  end: 85.041 },
  { i: 9,  name: "s09_github", start: 85.041,  end: 95.051 },
];

const arg = process.argv[2] || "all";
const targets = arg === "all" ? SEGMENTS : SEGMENTS.filter(s => String(s.i) === arg);
if (!targets.length) { console.error("no segment " + arg); process.exit(1); }
console.log("[render] targets:", targets.map(s => `${s.i}:${s.name}`).join(" | "));

const browser = await puppeteer.launch({
  executablePath: EDGE, headless: true,
  args: ["--disable-gpu","--disable-software-rasterizer","--disable-dev-shm-usage",
    "--force-device-scale-factor=1","--hide-scrollbars","--mute-audio",
    `--window-size=${WIDTH},${HEIGHT}`],
});
const page = await browser.newPage();
await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 1 });
page.on("pageerror", e => console.error("  [pageerror]", e.message));
await page.goto(COMPOSITION, { waitUntil: "networkidle2", timeout: 90000 });
await page.evaluate(() => new Promise((res, rej) => {
  const c = () => { if (window.__timelines && window.__timelines["promo-video"]) res(); else setTimeout(c, 50); };
  const t = setTimeout(() => rej(new Error("no tl")), 10000); c();
}));
console.log("[render] timeline ready");

for (const seg of targets) {
  const t0 = Date.now();
  const startFrame = Math.round(seg.start * FPS);
  const endFrame = Math.round(seg.end * FPS);
  const framesDir = mkdtempSync(join(tmpdir(), "hf-seg-"));
  console.log(`\n[seg ${seg.i}] ${seg.name}  ${seg.start}s -> ${seg.end}s  (${endFrame-startFrame} frames)`);
  for (let f = startFrame; f < endFrame; f++) {
    const tSec = f / FPS;
    await page.evaluate((time, dur) => {
      const tl = window.__timelines["promo-video"];
      if (tl) tl.progress(Math.min(time / dur, 0.999999));
    }, tSec, DUR);
    await page.screenshot({
      path: join(framesDir, `f_${String(f - startFrame).padStart(5, "0")}.png`),
      type: "png", clip: { x: 0, y: 0, width: WIDTH, height: HEIGHT },
      captureBeyondViewport: true, omitBackground: false,
    });
  }
  const outFile = join(SEG_DIR, `seg_${seg.i}_${seg.name}.mp4`);
  const pat = join(framesDir, "f_%05d.png").replace(/\\/g, "/");
  await new Promise((res, rej) => {
    execFile(FFMPEG, ["-y","-framerate",String(FPS),"-i",pat,"-c:v","libx264",
      "-preset","medium","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",outFile],
      { maxBuffer: 10*1024*1024 }, (err) => err ? rej(err) : res());
  });
  rmSync(framesDir, { recursive: true, force: true });
  console.log(`[seg ${seg.i}] done ${((Date.now()-t0)/1000).toFixed(1)}s -> ${outFile}`);
}
await browser.close();
console.log("[render] all done");
