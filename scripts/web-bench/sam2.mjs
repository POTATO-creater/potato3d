#!/usr/bin/env node
// Runs the web version's SAM 2.1 (web/src/pipeline/sam2.ts) in headless Chrome
// over a folder of frames, measures time per frame and writes each frame's
// logits (256², float32), for comparison with Transformers' propagation.
// Requires `pnpm web:dev` running with `WEB_FS_ANY=1`.
//
//   node scripts/web-bench/sam2.mjs <frames> <output> --box x0,y0,x1,y1 [--at K] [--models <folder>] [--n N] [--masks]
//
// --at: prompt frame (default 0); propagates forward then backward, like the
// app's masking. --rmbg: writes RMBG-1.4 masks (frame by frame) instead of SAM,
// for comparison.
// --models: folder with the four .onnx files (copied into OPFS first, so the
// models aren't downloaded for every fresh Chrome profile). --variant fp16|fp32:
// which graphs to use (default: the GPU's, fp16 with shader-f16); --models must
// be the same variant. --masks: also writes frame-sized PNG masks. --url: dev
// server (default http://localhost:1440).

import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const [framesDir, outDir, ...rest] = process.argv.slice(2);
const opt = (name, def) => {
  const i = rest.indexOf(`--${name}`);
  return i >= 0 ? rest[i + 1] : def;
};
const box = opt("box")?.split(",").map(Number);
const rmbg = rest.includes("--rmbg");
if (!framesDir || !outDir || (box?.length !== 4 && !rmbg)) {
  console.error("usage: sam2.mjs <frames> <output> --box x0,y0,x1,y1 [--models <folder>] [--n N] [--masks]");
  process.exit(2);
}
const base = opt("url", "http://localhost:1440");
const n = Number(opt("n", 1e9));
const models = opt("models");
const frames = readdirSync(framesDir).filter((f) => /\.(jpe?g|png)$/i.test(f)).sort().slice(0, n);
mkdirSync(outDir, { recursive: true });

const port = Number(process.env.CDP_PORT || 9334);
const chromePath = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const profileDir = mkdtempSync(join(process.env.TMPDIR || tmpdir(), "splatforge-chrome-"));
const chrome = spawn(
  chromePath,
  [
    "--headless=new",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profileDir}`,
    "--enable-gpu",
    "--ignore-gpu-blocklist",
    "--enable-unsafe-webgpu",
    "--use-angle=metal",
    "--no-first-run",
    "about:blank",
  ],
  { stdio: "ignore" },
);
// Each run creates a profile (with copies of the models in OPFS): delete it on
// exit, otherwise they pile up gigabytes on disk.
process.on("exit", () => {
  chrome.kill("SIGKILL");
  try {
    rmSync(profileDir, { recursive: true, force: true });
  } catch {}
});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 100 && !target; i++) {
  await sleep(200);
  try {
    target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page");
  } catch {}
}
if (!target) throw new Error("Chrome did not open the debugging port");
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let nextId = 0;
const pending = new Map();
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
  if (m.method === "Runtime.exceptionThrown") console.log("EXC", m.params.exceptionDetails.exception?.description);
  if (m.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(m.params.type))
    console.log(m.params.type, m.params.args.map((a) => a.value ?? a.description).join(" ").split("\n")[0]);
};
const send = (method, params = {}) =>
  new Promise((r) => {
    const i = ++nextId;
    pending.set(i, r);
    ws.send(JSON.stringify({ id: i, method, params }));
  });
const evaluate = async (expression) => {
  const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description ?? "page error");
  return r.result?.result?.value;
};
await send("Runtime.enable");
await send("Page.enable");
// Any page from the same server, just to get the origin (OPFS, module imports).
await send("Page.navigate", { url: `${base}/share.html` });
await sleep(1500);

const fs = (p) => `/@fs${resolve(p)}`;
const variant = await evaluate(`(async () => ${JSON.stringify(opt("variant") ?? null)} ?? (await import('/src/pipeline/sam2.ts')).sam2Variant())()`);
if (models) {
  const names = ["vision_encoder", "mask_decoder", "memory_attention", "memory_encoder"];
  await evaluate(`(async () => {
    const { SAM2_VARIANTS } = await import('/src/pipeline/sam2.ts');
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('models', { create: true });
    for (const [i, src] of ${JSON.stringify(names)}.entries()) {
      const buf = await (await fetch(${JSON.stringify(fs(models))} + '/' + src + '.onnx')).arrayBuffer();
      const w = await (await dir.getFileHandle(SAM2_VARIANTS[${JSON.stringify(variant)}][i].file, { create: true })).createWritable();
      await w.write(buf); await w.close();
    }
  })()`);
}

const t0 = Date.now();
if (rmbg) {
  await evaluate(`(async () => {
    const { Segmenter } = await import('/src/pipeline/segment.ts');
    window.seg = await Segmenter.create('rmbg', () => {});
    window.png = await import('/src/pipeline/png.ts');
  })()`);
  for (const f of frames) {
    const r = await evaluate(`(async () => {
      const bmp = await createImageBitmap(await (await fetch(${JSON.stringify(fs(join(framesDir, f)))})).blob());
      const t = performance.now();
      const m = await seg.mask(bmp, bmp.width, bmp.height);
      const ms = performance.now() - t;
      const b = new Uint8Array(await (await png.encodePng(m, bmp.width, bmp.height, 1)).arrayBuffer());
      let s = ''; for (let k = 0; k < b.length; k += 0x8000) s += String.fromCharCode(...b.subarray(k, k + 0x8000));
      bmp.close();
      return { ms, png: btoa(s) };
    })()`);
    writeFileSync(join(outDir, f.replace(/\.[^.]+$/, ".png")), Buffer.from(r.png, "base64"));
    console.log(`${f}: ${r.ms.toFixed(0)} ms`);
  }
  process.exit(0);
}
await evaluate(`(async () => {
  const { Sam2Tracker } = await import('/src/pipeline/sam2.ts');
  window.tracker = await Sam2Tracker.create(${frames.length}, () => {}, ${JSON.stringify(variant)});
  window.sam2 = await import('/src/pipeline/sam2-math.ts');
  window.png = await import('/src/pipeline/png.ts');
})()`);
console.log(`${await evaluate("tracker.backend")}: sessions in ${((Date.now() - t0) / 1000).toFixed(1)} s`);

const at = Number(opt("at", 0));
const order = [...frames.keys()].slice(at).concat([...frames.keys()].slice(0, at).reverse());
const times = [];
for (const i of order) {
  const r = await evaluate(`(async () => {
    const bmp = await createImageBitmap(await (await fetch(${JSON.stringify(fs(join(framesDir, frames[i])))})).blob());
    const t = performance.now();
    const f = await tracker.frame(${i}, bmp, bmp.width, bmp.height, ${i === at ? JSON.stringify({ box }) : "undefined"}, ${i < at});
    const ms = performance.now() - t;
    let s = ''; const u = new Uint8Array(f.logits.buffer);
    for (let k = 0; k < u.length; k += 0x8000) s += String.fromCharCode(...u.subarray(k, k + 0x8000));
    let mask = null;
    if (${rest.includes("--masks")}) {
      const m = sam2.logitsToMask(f.logits, bmp.width, bmp.height);
      const b = new Uint8Array(await (await png.encodePng(m, bmp.width, bmp.height, 1)).arrayBuffer());
      mask = ''; for (let k = 0; k < b.length; k += 0x8000) mask += String.fromCharCode(...b.subarray(k, k + 0x8000));
      mask = btoa(mask);
    }
    bmp.close();
    return { ms, score: f.objectScore, logits: btoa(s), mask };
  })()`);
  times.push(r.ms);
  const stem = frames[i].replace(/\.[^.]+$/, "");
  writeFileSync(join(outDir, `${stem}.f32`), Buffer.from(r.logits, "base64"));
  if (r.mask) writeFileSync(join(outDir, `${stem}.png`), Buffer.from(r.mask, "base64"));
  console.log(`${stem}: ${r.ms.toFixed(0)} ms, object ${r.score.toFixed(2)}`);
}
const steady = times.slice(2);
console.log(
  `${frames.length} frames; mean ${(steady.reduce((a, b) => a + b, 0) / Math.max(steady.length, 1)).toFixed(0)} ms/frame (excluding the first 2)`,
);
process.exit(0);
