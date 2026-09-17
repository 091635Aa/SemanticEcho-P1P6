// render.mjs — puppeteer-core + Edge 逐帧渲染 index.html → MP4
import puppeteer from "puppeteer-core";
import { execFile } from "node:child_process";
import { mkdtempSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const EDGE = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const FFMPEG = "D:\\FormatFactory\\ffmpeg.exe"; // 完整版 ffmpeg（支持音频）
const COMPOSITION = "file:///F:/lora外挂/语义回响_海外宣传/index.html";
const OUTPUT_VIDEO = "F:\\lora外挂\\语义回响_海外宣传\\semantic_echo_p1p6_95s_silent.mp4";
const WIDTH = 1920;
const HEIGHT = 1080;
const DURATION = 95.05;
const FPS = 30;
const TOTAL_FRAMES = Math.round(DURATION * FPS);

const framesDir = mkdtempSync(join(tmpdir(), "hf-frames-"));
console.log("[render] temp frames:", framesDir);
console.log(`[render] frames: ${TOTAL_FRAMES} (${FPS}fps x ${DURATION}s)`);

const browser = await puppeteer.launch({
  executablePath: EDGE,
  headless: true,
  args: [
    "--disable-gpu",
    "--disable-software-rasterizer",
    "--disable-dev-shm-usage",
    "--force-device-scale-factor=1",
    "--hide-scrollbars",
    "--mute-audio",
    `--window-size=${WIDTH},${HEIGHT}`,
  ],
});

const page = await browser.newPage();
await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 1 });
page.on("console", (m) => {
  const t = m.text();
  if (t.includes("[GSAP]") || t.includes("warning") || t.includes("error")) console.log("  [browser]", t);
});
page.on("pageerror", (e) => console.error("  [pageerror]", e.message));

await page.goto(COMPOSITION, { waitUntil: "networkidle2", timeout: 90000 });
console.log("[render] page loaded");

await page.evaluate(() => {
  return new Promise((resolve, reject) => {
    const check = () => {
      if (window.__timelines && window.__timelines["promo-video"]) resolve();
      else setTimeout(check, 50);
    };
    const tout = setTimeout(() => reject(new Error("GSAP timeline not registered in 10s")), 10000);
    check();
  });
});
console.log("[render] GSAP timeline ready");

const t0 = Date.now();
for (let i = 0; i < TOTAL_FRAMES; i++) {
  const tSec = i / FPS;
  await page.evaluate((time, dur) => {
    const tl = window.__timelines["promo-video"];
    if (!tl) return;
    tl.progress(Math.min(time / dur, 0.999999));
  }, tSec, DURATION);

  const outFile = join(framesDir, `frame_${String(i).padStart(5, "0")}.png`);
  await page.screenshot({
    path: outFile,
    type: "png",
    clip: { x: 0, y: 0, width: WIDTH, height: HEIGHT },
    captureBeyondViewport: true,
    omitBackground: false,
  });

  if (i % 90 === 0 || i === TOTAL_FRAMES - 1) {
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const pct = ((i + 1) / TOTAL_FRAMES * 100).toFixed(1);
    console.log(`  frame ${i + 1}/${TOTAL_FRAMES} (${pct}%) t=${tSec.toFixed(2)}s elapsed=${elapsed}s`);
  }
}

console.log("[render] frames done, closing browser");
await browser.close();

console.log("[render] ffmpeg encode...");
const inputPattern = join(framesDir, "frame_%05d.png").replace(/\\/g, "/");
await new Promise((resolve, reject) => {
  const args = [
    "-y", "-framerate", String(FPS),
    "-i", inputPattern,
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    OUTPUT_VIDEO,
  ];
  console.log("  ffmpeg", args.join(" "));
  const proc = execFile(FFMPEG, args, { maxBuffer: 10 * 1024 * 1024 }, (err, stdout, stderr) => {
    if (err) { console.error(stderr); reject(err); return; }
    resolve();
  });
});

console.log("[render] cleanup");
try { rmSync(framesDir, { recursive: true, force: true }); } catch {}

console.log(`[render] DONE -> ${OUTPUT_VIDEO}`);
