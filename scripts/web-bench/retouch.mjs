#!/usr/bin/env node
// End-to-end retouch in headless Chrome (visible tabs in the desktop app get
// throttled and WebGPU training stalls). Requires `pnpm web:dev` with
// WEB_FS_ANY=1 and a ComfyUI (or the mock, scripts/comfy-mock.mjs) at the
// given address.
//
//   node scripts/web-bench/retouch.mjs <app url> <project.gorbit> <workflow.json>
//        [--comfy http://127.0.0.1:8188] [--camera 27]
//        [--preset view|below|above|orbitBelow|orbitAbove|wave] [--seconds 3.2] [--tilt 30]
//        [--steps 3000] [--out splat.ply]
//
// Opens the package, aims at camera index --camera, creates the preset's clip,
// generates the reference video, restores it through ComfyUI, applies the
// retouch and writes the resulting splat to --out.
//
// The UI selectors below match pt-BR text, so the script forces the pt-BR locale.

import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const [url, gorbit, workflowPath, ...rest] = process.argv.slice(2);
const opt = (name, def) => {
  const i = rest.indexOf(`--${name}`);
  return i >= 0 ? rest[i + 1] : def;
};
if (!url || !gorbit || !workflowPath) {
  console.error("usage: retouch.mjs <url> <project.gorbit> <workflow.json> [options] (see the header)");
  process.exit(2);
}
const comfy = opt("comfy", "http://127.0.0.1:8188");
const camera = Number(opt("camera", 0));
const preset = opt("preset", "below");
const seconds = opt("seconds", null);
const tilt = Number(opt("tilt", 30));
const steps = Number(opt("steps", 3000));
const out = opt("out", "retouch_splat.ply");

const port = Number(process.env.CDP_PORT || 9334);
const chromePath = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const profileDir = mkdtempSync(join(process.env.TMPDIR || tmpdir(), "gorbit-retouch-"));
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
    "--window-size=1440,900",
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
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
  if (m.method === "Runtime.exceptionThrown") console.log("EXC", m.params.exceptionDetails.exception?.description?.split("\n")[0]);
  if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error")
    console.log("ERR", m.params.args.map((a) => a.value ?? a.description).join(" ").split("\n")[0]);
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
const fail = (msg) => {
  console.log("ERROR:", msg);
  process.exit(1);
};
const btn = (text) => `[...document.querySelectorAll('.retouch button')].find(b => b.textContent.trim().startsWith(${JSON.stringify(text)}))`;
const waitFor = async (expr, what, timeoutS = 600) => {
  for (let i = 0; i < timeoutS * 2; i++) {
    if (await evaluate(expr)) return;
    const err = await evaluate(`document.querySelector('.retouch-error')?.textContent ?? null`);
    if (err) fail(`${what}: ${err}`);
    await sleep(500);
  }
  fail(`timeout: ${what}`);
};

// ComfyUI and motion settings before opening the app.
const graph = JSON.parse(readFileSync(workflowPath, "utf8"));
const save = [...Object.entries(graph)].find(([, n]) => /^Save|Combine/.test(n.class_type))?.[0];
const load = [...Object.entries(graph)].find(([, n]) => /^(VHS_)?LoadVideo/.test(n.class_type));
const comfySettings = {
  url: comfy,
  workflows: [{ id: "wf", name: "workflow", kind: "restore", graph, io: { input: { node: load[0], field: load[1].class_type === "LoadVideo" ? "file" : "video" }, output: { node: save } }, importedAt: Date.now() }],
};
await send("Runtime.enable");
await send("Page.enable");
await send("Page.navigate", { url: new URL(url).origin });
await sleep(1000);
await evaluate(`localStorage.setItem('splatforge.locale.v1', 'pt-BR');
  localStorage.setItem('splatforge.comfy.v1', ${JSON.stringify(JSON.stringify(comfySettings))});
  localStorage.setItem('splatforge.retouchMotion.v1', ${JSON.stringify(JSON.stringify({ kind: preset, seconds: seconds && Number(seconds), tilt }))});`);
await send("Page.navigate", { url });
await sleep(2500);

const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(0);
console.log("opening the package…");
await evaluate(`(async () => {
  const file = new File([await (await fetch(${JSON.stringify(`/@fs${resolve(gorbit)}`)})).blob()], ${JSON.stringify(gorbit.split("/").pop())});
  const dt = new DataTransfer(); dt.items.add(file);
  for (const type of ["dragenter", "dragover", "drop"]) window.dispatchEvent(new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true }));
})()`);
await waitFor(`!!${btn("Traçar trajetória")} && !${btn("Traçar trajetória")}.disabled && !!window.__gorbitViewer?.hasSplat()`, "project open", 300);
const projectDir = await evaluate(`document.querySelector('.panel .path')?.textContent`);
console.log(`[${secs()}s] project ${projectDir}`);

// Aim at the requested real camera (the preset centers on the photo closest to the view).
await evaluate(`(async () => {
  // The same URL the app loaded the module from (otherwise it's another instance, without the host).
  const src = performance.getEntriesByType('resource').map((e) => e.name).find((n) => n.split('?')[0].endsWith('/src/lib/api.ts'));
  const { api } = await import(src);
  const cams = await api.retouch.cameras(${JSON.stringify(projectDir)});
  window.__gorbitViewer.goTo(cams.cameras[${camera}]);
})()`);
await sleep(500);
// The motion (--preset) is already chosen in localStorage.
await evaluate(`${btn("Traçar trajetória")}.click()`);
await sleep(300);
await evaluate(`${btn("Gerar vídeo de referência")}.click()`);
await waitFor(`!document.querySelector('.retouch-draft') && document.querySelectorAll('.retouch-clip').length > 0`, "reference video");
console.log(`[${secs()}s] reference generated`);

await evaluate(`[...document.querySelectorAll('.retouch-clip')].pop().querySelector('button').click()`);
let last = "";
await (async () => {
  for (;;) {
    const s = await evaluate(`[...document.querySelectorAll('.retouch-clip')].pop()?.querySelector('.retouch-progress .muted')?.textContent ?? null`);
    if (s && s !== last) console.log(`[${secs()}s] ${(last = s)}`);
    if (await evaluate(`[...document.querySelectorAll('.retouch-clip')].pop()?.querySelector('.retouch-chip.on') != null && !document.querySelector('.retouch-clip .retouch-progress')`)) return;
    const err = await evaluate(`document.querySelector('.retouch-error')?.textContent ?? null`);
    if (err) fail(`ComfyUI: ${err}`);
    await sleep(500);
  }
})();
console.log(`[${secs()}s] restored · ${await evaluate(`[...document.querySelectorAll('.retouch-clip')].pop()?.querySelector('.muted')?.textContent`)}`);

// Fine-tune steps, then apply.
await evaluate(`(() => {
  const input = document.querySelector('.retouch-apply input');
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, ${steps});
  input.dispatchEvent(new Event('input', { bubbles: true }));
})()`);
await evaluate(`${btn("Aplicar retoque")}.click()`);
last = "";
for (;;) {
  const s = await evaluate(`document.querySelector('.retouch-apply .retouch-progress .muted')?.textContent ?? null`);
  if (s && s !== last && !/^Passo \d+\/\d+/.test(last ?? "") ) console.log(`[${secs()}s] ${s}`);
  else if (s && /^Passo (\d+)00\//.test(s) && s !== last) console.log(`[${secs()}s] ${s}`);
  last = s ?? last;
  if (await evaluate(`!!document.querySelector('.retouch-applied') && !document.querySelector('.retouch-apply .retouch-progress')`)) break;
  const err = await evaluate(`document.querySelector('.retouch-error')?.textContent ?? null`);
  if (err) fail(`apply: ${err}`);
  await sleep(1000);
}
console.log(`[${secs()}s] ${await evaluate(`document.querySelector('.retouch-applied .good')?.textContent`)}`);

// Final splat from OPFS to disk.
const b64 = await evaluate(`(async () => {
  let d = await navigator.storage.getDirectory();
  for (const p of ${JSON.stringify(projectDir)}.split('/').filter(Boolean)) d = await d.getDirectoryHandle(p);
  const u = new Uint8Array(await (await (await d.getFileHandle('splat.ply')).getFile()).arrayBuffer());
  let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000));
  return btoa(s);
})()`);
writeFileSync(out, Buffer.from(b64, "base64"));
console.log(`retouched splat at ${out}`);
process.exit(0);
