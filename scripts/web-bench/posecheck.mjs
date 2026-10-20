#!/usr/bin/env node
// Runs the pose check (web/src/posecheck) in headless Chrome on a COLMAP
// dataset with masks, and measures it. Requires the dev server with
// `WEB_FS_ANY=1` (to read local files via /@fs).
//
//   node scripts/web-bench/posecheck.mjs <origin> <dataset> [options]
//
// <dataset>: images/ (frames), sparse/0/ (poses) and masks (255 = subject,
// <frame stem>.png) in masks_gt/ (synth-orbit.py --green) or --masks <dir>;
// --chroma makes them with the app's solid-colour cutout instead.
// Options:
//   --mode auto|always     correct only when flagged (default) or always
//   --perturb a:b:deg[:k]  before the check, turns the cameras of frames a..b
//                          (indices in name order) by `deg` degrees and moves
//                          them by deg % of the orbit radius, all the same way
//                          (a block that disagrees with the rest of the orbit);
//                          with k, only every k-th frame from a
//   --jitter deg           also turns every frame by `deg` degrees and moves it by
//                          deg % of the radius, each its own random way (seeded)
//   --settings '<json>'    overrides POSE_CHECK fields (web/src/posecheck/job.ts),
//                          e.g. '{"rounds":2,"refine":{"steps":600,"resolution":480}}'
//   --out <dir>            writes <dir>/sparse/0 with the corrected poses and
//                          <dir>/trace.json (each correction step)
//   --dump a.png,b.png     with --out: also each step's render and frame of these
// Prints the score per frame block, the timing of each phase, and each
// frame's pose change against the dataset's own poses (with --perturb and a
// ground-truth dataset: how much of the error was recovered).
//
// Chrome comes from CHROME (default: the macOS one in /Applications).

import { spawn } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const [origin, datasetArg, ...rest] = process.argv.slice(2);
const opt = (name, def) => {
  const i = rest.indexOf(`--${name}`);
  return i >= 0 ? rest[i + 1] : def;
};
if (!origin || !datasetArg) {
  console.error("usage: posecheck.mjs <origin> <dataset> [options]  (see the file header)");
  process.exit(2);
}
const dataset = resolve(datasetArg);
const chroma = rest.includes("--chroma");
const masksDir = resolve(opt("masks", join(dataset, "masks_gt")));
const imageNames = readdirSync(join(dataset, "images")).filter((n) => /\.(jpe?g|png|webp)$/i.test(n)).sort();
const maskNames = chroma ? [] : readdirSync(masksDir).filter((n) => n.endsWith(".png")).sort();

const port = Number(process.env.CDP_PORT || 9334);
const chromePath = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const profileDir = mkdtempSync(join(process.env.TMPDIR || tmpdir(), "gausify-posecheck-"));
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
    "--no-default-browser-check",
    "about:blank",
  ],
  { stdio: "ignore" },
);
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
if (!target) {
  console.error("Chrome did not open the debugging port.");
  process.exit(1);
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let nextId = 0;
const pending = new Map();
const t0 = Date.now();
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
  if (m.method === "Runtime.exceptionThrown") console.log("EXC", m.params.exceptionDetails.exception?.description?.split("\n")[0]);
  if (m.method === "Runtime.consoleAPICalled") {
    const text = m.params.args.map((a) => a.value ?? a.description).join(" ");
    if (m.params.type === "error") console.log("ERR", text.split("\n")[0]);
    else if (text.startsWith("[pc]")) console.log(`[${((Date.now() - t0) / 1000).toFixed(0).padStart(4)}s] ${text.slice(5)}`);
  }
};
const send = (method, params = {}) =>
  new Promise((r) => {
    const i = ++nextId;
    pending.set(i, r);
    ws.send(JSON.stringify({ id: i, method, params }));
  });
const evaluate = async (expression) => {
  const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description ?? "evaluation failed");
  return r.result?.result?.value;
};

await send("Runtime.enable");
await send("Page.enable");
// A static file, not the app: no Vite client, so editing the sources during a
// run doesn't reload the page (modules are imported below, same origin).
await send("Page.navigate", { url: `${origin}/manifest.webmanifest` });
await sleep(1500);
await evaluate(`localStorage.setItem('gausify.poseCheck', ${JSON.stringify(opt("settings", "{}"))})`);

const job = {
  root: "/projects/posecheck-bench",
  fs: `${origin}/@fs`,
  dataset,
  masksDir,
  images: imageNames,
  masks: maskNames,
  mode: opt("mode", "auto"),
  chroma,
  perturb: opt("perturb", null),
  jitter: opt("jitter", null),
  dump: opt("dump", "").split(",").filter(Boolean),
};
const result = await evaluate(`(async () => {
  const job = ${JSON.stringify(job)};
  const files = await import('/src/host/files.ts');
  const { layout } = await import('/src/host/project.ts');
  const { checkPoses } = await import('/src/posecheck/job.ts');
  const { readViews } = await import('/src/mesh/colmap.ts');
  const { rewriteImages } = await import('/src/sfm/colmap.ts');
  const { expSO3, mul3, mulVec3, transpose3, quatFromMat } = await import('/src/sfm/linalg.ts');
  const l = layout(job.root);
  await files.remove(job.root);
  const get = async (p) => { const r = await fetch(job.fs + p); if (!r.ok) throw new Error('fetch ' + p + ': ' + r.status); return r.blob(); };
  for (const n of job.images) await files.writeFile(l.images + '/' + n, await get(job.dataset + '/images/' + n));
  for (const n of job.masks) await files.writeFile(l.masks + '/' + n, await get(job.masksDir + '/' + n));
  for (const n of ['cameras.bin', 'images.bin', 'points3D.bin']) await files.writeFile(l.model + '/' + n, await get(job.dataset + '/sparse/0/' + n));
  const cams = await (await files.readFile(l.model + '/cameras.bin')).arrayBuffer();
  const reference = readViews(cams, await (await files.readFile(l.model + '/images.bin')).arrayBuffer());
  const centre = (v) => mulVec3(transpose3(v.R), v.t).map((x) => -x);
  const cs = reference.map(centre);
  const mid = [0, 1, 2].map((k) => cs.reduce((a, c) => a + c[k], 0) / cs.length);
  const radius = cs.reduce((a, c) => a + Math.hypot(c[0] - mid[0], c[1] - mid[1], c[2] - mid[2]), 0) / cs.length;
  if (job.perturb) {
    const [a, b, deg, every = 1] = job.perturb.split(':').map(Number);
    const rad = deg * Math.PI / 180;
    const axis = [0.36, 0.8, -0.48];
    const dR = expSO3(axis.map((x) => x * rad));
    const shift = [0.6, -0.48, 0.64].map((x) => x * radius * deg / 100);
    const block = new Set(reference.slice(a, b + 1).filter((_, i) => i % every === 0).map((v) => v.name));
    const ib = await (await files.readFile(l.model + '/images.bin')).arrayBuffer();
    await files.writeFile(l.model + '/images.bin', rewriteImages(ib, (r) => {
      if (!block.has(r.name)) return r;
      const v = reference.find((x) => x.name === r.name);
      const t = mulVec3(dR, v.t);
      return { ...r, q: quatFromMat(mul3(dR, v.R)), t: [t[0] + shift[0], t[1] + shift[1], t[2] + shift[2]] };
    }));
    console.log('[pc] perturbed ' + block.size + ' frames in ' + a + '–' + b + ' by ' + deg + '° and ' + deg + '% of the radius');
  }
  if (job.jitter) {
    // every frame: its own random axis and direction (seeded), the same size
    const deg = Number(job.jitter);
    let seed = 12345;
    const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
    const unit = () => { const v = [rnd(), rnd(), rnd()]; const n = Math.hypot(...v); return v.map((x) => x / n); };
    const ib = await (await files.readFile(l.model + '/images.bin')).arrayBuffer();
    const now = readViews(cams, ib);
    await files.writeFile(l.model + '/images.bin', rewriteImages(ib, (r) => {
      const v = now.find((x) => x.name === r.name);
      const dR = expSO3(unit().map((x) => x * deg * Math.PI / 180));
      const shift = unit().map((x) => x * radius * deg / 100);
      const t = mulVec3(dR, v.t);
      return { ...r, q: quatFromMat(mul3(dR, v.R)), t: [t[0] + shift[0], t[1] + shift[1], t[2] + shift[2]] };
    }));
    console.log('[pc] jittered every frame by ' + deg + '° and ' + deg + '% of the radius');
  }
  const start = readViews(cams, await (await files.readFile(l.model + '/images.bin')).arrayBuffer());
  const chromaOpts = { autoColor: true, color: [0, 255, 0], tolerance: 12, softness: 6, contiguous: true };
  const masks = { enabled: true, forPoses: false, method: job.chroma ? 'chroma' : 'external', chroma: chromaOpts, despill: job.chroma, despillStrength: 100 };
  if (job.chroma) {
    const { makeMasks } = await import('/src/pipeline/masks.ts');
    await makeMasks(await files.dirAt(l.images), await files.dirAt(l.masks, true), masks, (s) => console.log('[pc] ' + s), () => {});
  }
  let last = '';
  const rep = {
    // one line per phase (the message without its counters)
    progress: (f, msg) => { const s = JSON.stringify(msg).replace(/"(iter|i)":\\d+,?/g, ''); if (s !== last) { last = s; console.log('[pc] ' + s); } },
    log: (line) => console.log('[pc] ' + line),
  };
  const t0 = performance.now();
  const traces = [], dumps = {};
  const b64 = (a) => { const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000)); return btoa(s); };
  const trace = {
    dump: job.dump,
    step: (t) => {
      const { data, ...rest } = t;
      traces.push(rest);
      if (data) dumps[t.round + '-' + t.name + '-' + t.step] = { w: data.w, h: data.h, K: data.K, rgba: b64(data.rgba), depth: b64(data.depth), frame: b64(data.frame) };
    },
  };
  const stats = await checkPoses(job.root, masks, job.mode, rep, new AbortController().signal, trace);
  const seconds = (performance.now() - t0) / 1000;
  const end = readViews(cams, await (await files.readFile(l.model + '/images.bin')).arrayBuffer());
  const angle = (A, B) => { const D = mul3(A, transpose3(B)); return Math.acos(Math.min(1, Math.max(-1, (D[0] + D[4] + D[8] - 1) / 2))) * 180 / Math.PI; };
  const err = (vs) => vs.map((v, i) => ({ deg: angle(v.R, reference[i].R), pos: 100 * Math.hypot(...centre(v).map((x, k) => x - cs[i][k])) / radius }));
  const bytes = new Uint8Array(await (await files.readFile(l.model + '/images.bin')).arrayBuffer());
  await files.remove(job.root);
  return { stats, seconds, names: reference.map((v) => v.name), before: err(start), after: err(end), imagesBin: b64(bytes), traces, dumps };
})()`);

const { stats, seconds, names, before, after } = result;
const pct = (x) => (x == null ? "—" : `${(100 * x).toFixed(2)}%`);
console.log(`\ncheck: ${seconds.toFixed(1)} s, mode ${job.mode}`);
if (!stats) process.exit(0);
console.log(`orbit score ${pct(stats.score)} (threshold ${pct(stats.threshold)})${stats.flagged ? " FLAGGED" : ""}; rounds ${stats.rounds}; after ${pct(stats.after)}`);
const block = 10;
for (let i = 0; i < names.length; i += block) {
  const run = (a) => (a ? a.slice(i, i + block) : null);
  const mean = (a) => (a && a.length ? a.reduce((s, x) => s + x, 0) / a.length : null);
  const e0 = before.slice(i, i + block), e1 = after.slice(i, i + block);
  const m = (a, k) => (a.reduce((s, x) => s + x[k], 0) / a.length).toFixed(3);
  console.log(
    `  ${String(i).padStart(3)}–${String(Math.min(i + block, names.length) - 1).padStart(3)}  score ${pct(mean(run(stats.frames))).padStart(6)} → ${pct(mean(run(stats.afterFrames))).padStart(6)}` +
      `   vs dataset poses: ${m(e0, "deg")}° / ${m(e0, "pos")}% → ${m(e1, "deg")}° / ${m(e1, "pos")}%`,
  );
}
const out = opt("out");
if (out) {
  mkdirSync(out, { recursive: true });
  writeFileSync(join(out, "trace.json"), JSON.stringify(result.traces, null, 0));
  for (const [k, d] of Object.entries(result.dumps)) {
    writeFileSync(join(out, `dump-${k}.json`), JSON.stringify({ w: d.w, h: d.h, K: d.K }));
    for (const f of ["rgba", "depth", "frame"]) writeFileSync(join(out, `dump-${k}.${f}`), Buffer.from(d[f], "base64"));
  }
  mkdirSync(join(out, "sparse", "0"), { recursive: true });
  for (const n of ["cameras.bin", "points3D.bin"]) copyFileSync(join(dataset, "sparse", "0", n), join(out, "sparse", "0", n));
  writeFileSync(join(out, "sparse", "0", "images.bin"), Buffer.from(result.imagesBin, "base64"));
  if (!existsSync(join(out, "images"))) spawn("ln", ["-s", join(dataset, "images"), join(out, "images")]);
  console.log(`poses written to ${join(out, "sparse", "0")}`);
}
process.exit(0);
