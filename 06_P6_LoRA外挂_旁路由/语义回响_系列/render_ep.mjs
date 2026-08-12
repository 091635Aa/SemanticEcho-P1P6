// render_ep.mjs v2 — 每段渲染后重载页面防内存膨胀 + 截图重试
import puppeteer from "puppeteer-core";
import { execFile } from "node:child_process";
import { mkdtempSync, rmSync, readFileSync, existsSync, mkdirSync } from "node:fs";
import { join, basename } from "node:path";
import { tmpdir } from "node:os";

const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const FFMPEG = "D:\\FormatFactory\\ffmpeg.exe";
const EP = process.argv[2];
const W = 1920, H = 1080, FPS = 30;
const COMPOSITION = "file:///" + EP.replace(/\\/g, "/") + "/index.html";
const SEG_DIR = EP + "\\segments";
mkdirSync(SEG_DIR, { recursive: true });

const tl = JSON.parse(readFileSync(EP + "\\timeline.json", "utf-8"));
const script = JSON.parse(readFileSync(EP + "\\script.json", "utf-8"));
let t = 0.6;
const SEGMENTS = script.scenes.map((sc, i) => {
  const dur = tl["s" + String(i).padStart(2, "0")] || 8;
  const s = { i, start: t, audio_start: t + 0.4, end: t + 0.4 + dur + 0.8 };
  t = s.end; return s;
});
const DUR = t;
const arg = process.argv[3] || "all";
const targets = arg === "all" ? SEGMENTS : SEGMENTS.filter(s => String(s.i) === arg);
console.log("[render]", basename(EP), "total", DUR.toFixed(1), "s | targets:", targets.map(s=>s.i).join(","));

async function shot(page, time, outFile, tries=5) {
  for (let a = 0; a < tries; a++) {
    try {
      await page.screenshot({ path: outFile, type: "png", clip: { x:0,y:0,width:W,height:H }, captureBeyondViewport: true, omitBackground: false });
      return true;
    } catch (e) {
      console.log("    retry", a, e.message.slice(0,60));
      await new Promise(r => setTimeout(r, 1500));
    }
  }
  return false;
}

const browser = await puppeteer.launch({ executablePath: EDGE, headless: true,
  args: ["--disable-gpu","--disable-software-rasterizer","--disable-dev-shm-usage","--force-device-scale-factor=1","--hide-scrollbars","--mute-audio",`--window-size=${W},${H}`] });
let page = await browser.newPage();
await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
page.on("pageerror", e => console.error("  [pageerror]", e.message));
await page.goto(COMPOSITION, { waitUntil: "networkidle2", timeout: 90000 });
await page.evaluate(() => new Promise((res, rej) => {
  const c = () => { if (window.__timelines && window.__timelines["promo-video"]) res(); else setTimeout(c, 50); };
  const t2 = setTimeout(() => rej(new Error("no tl")), 10000); c();
}));
console.log("[render] timeline ready");

for (const seg of targets) {
  const outFile = join(SEG_DIR, `seg_${seg.i}.mp4`);
  if (existsSync(outFile)) { console.log(`[seg ${seg.i}] exists, skip`); continue; }
  const t0 = Date.now();
  const sF = Math.round(seg.start * FPS), eF = Math.round(seg.end * FPS);
  const framesDir = mkdtempSync(join(tmpdir(), "hf-ep-"));
  console.log(`[seg ${seg.i}] ${seg.start.toFixed(2)}->${seg.end.toFixed(2)}s (${eF-sF} frames)`);
  let ok = true;
  for (let f = sF; f < eF; f++) {
    const tSec = f / FPS;
    await page.evaluate((time, dur) => { const tl2 = window.__timelines["promo-video"]; if (tl2) tl2.progress(Math.min(time/dur,0.999999)); }, tSec, DUR);
    const okf = await shot(page, tSec, join(framesDir, `f_${String(f-sF).padStart(5,"0")}.png`));
    if (!okf) { console.error(`[seg ${seg.i}] shot failed at ${tSec}`); ok = false; break; }
  }
  if (ok) {
    const pat = join(framesDir, "f_%05d.png").replace(/\\/g, "/");
    await new Promise((res, rej) => execFile(FFMPEG, ["-y","-framerate",String(FPS),"-i",pat,"-c:v","libx264","-preset","medium","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",outFile], {maxBuffer:10*1024*1024}, (err) => err ? rej(err) : res()));
    console.log(`[seg ${seg.i}] done ${((Date.now()-t0)/1000).toFixed(1)}s`);
  }
  rmSync(framesDir, { recursive: true, force: true });
  // 每段后重载页面，释放内存
  try { await page.goto("about:blank"); await page.goto(COMPOSITION, { waitUntil: "networkidle2", timeout: 90000 }); } catch(e) { console.log("  reload err", e.message.slice(0,60)); }
}
await browser.close();
console.log("[render] done");
