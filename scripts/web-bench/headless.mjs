#!/usr/bin/env node
// Roda o protótipo web num Chrome headless e mede, sem depender de uma janela
// visível (abas ocultas sofrem throttling). Precisa do `pnpm web:dev`
// rodando; para abrir pastas locais por URL, suba-o com `WEB_FS_ANY=1`.
//
//   node scripts/web-bench/headless.mjs train <url> [--steps 3000] [--max-res 960] [--out splat.ply]
//   node scripts/web-bench/headless.mjs poses <url> [--out <pasta>]
//   node scripts/web-bench/headless.mjs profile <url> [--seconds 10]
//
// <url>: http://localhost:1440/?dataset=/@fs/<caminho absoluto do dataset>
// - train:   treina com os passos/resolução dados e a prévia manual; imprime o
//            progresso e as avaliações; --out salva o .ply final.
// - poses:   clica em "Calcular poses no navegador"; --out grava o modelo em
//            <pasta>/sparse/0 (e liga <pasta>/images às imagens do dataset,
//            para a pasta servir de dataset de treino).
// - profile: perfil de CPU de alguns segundos de treino (após o passo 100),
//            com as funções que mais gastam. Use `pnpm web:wasm --profiling`
//            para ter os nomes das funções do WASM.
//
// O Chrome vem de CHROME (padrão: o do macOS em /Applications).

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [mode, url, ...rest] = process.argv.slice(2);
const opt = (name, def) => {
  const i = rest.indexOf(`--${name}`);
  return i >= 0 ? rest[i + 1] : def;
};
if (!["train", "poses", "profile"].includes(mode) || !url) {
  console.error("uso: headless.mjs train|poses|profile <url> [opções]  (ver o cabeçalho do arquivo)");
  process.exit(2);
}

const port = Number(process.env.CDP_PORT || 9333);
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
    "--no-default-browser-check",
    "--window-size=1280,800",
    "about:blank",
  ],
  { stdio: "ignore" },
);
process.on("exit", () => chrome.kill());

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 100 && !target; i++) {
  await sleep(200);
  try {
    target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page");
  } catch {}
}
if (!target) {
  console.error("Chrome não abriu a porta de depuração.");
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
  if (m.method === "Runtime.exceptionThrown")
    console.log("EXC", m.params.exceptionDetails.exception?.description?.split("\n")[0]);
  if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error")
    console.log("ERR", m.params.args.map((a) => a.value ?? a.description).join(" ").split("\n")[0]);
};
const send = (method, params = {}) =>
  new Promise((r) => {
    const i = ++nextId;
    pending.set(i, r);
    ws.send(JSON.stringify({ id: i, method, params }));
  });
const evaluate = async (expression) =>
  (await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true })).result?.result?.value;
/** Bytes de um Uint8Array/Blob da página, via base64. */
const fetchBytes = async (expr) => {
  const b64 = await evaluate(`(async () => {
    let u = ${expr};
    if (u instanceof Blob) u = new Uint8Array(await u.arrayBuffer());
    let s = '';
    for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000));
    return btoa(s);
  })()`);
  return Buffer.from(b64, "base64");
};
const progressLine = async () =>
  (await evaluate(`document.querySelectorAll('.web-kv')[1]?.innerText.replace(/\\n/g, ' ')`)) ?? "";
const logText = async () => (await evaluate(`document.querySelector('.web-log').innerText`)) ?? "";
const errorBanner = () => evaluate(`document.querySelector('.banner.error')?.innerText`);
const waitEnabled = async (selector) => {
  for (let i = 0; i < 120; i++) {
    if (await evaluate(`(() => { const b = ${selector}; return !!b && !b.disabled; })()`)) return;
    await sleep(500);
  }
  throw new Error(`botão não habilitou: ${selector}`);
};

await send("Runtime.enable");
await send("Page.enable");
await send("Page.navigate", { url });
const trainButton = `[...document.querySelectorAll('button')].find(b => b.textContent.includes('Treinar no navegador'))`;
const posesButton = `document.querySelector('.web-sfm')`;
// dataset carregado e WebGPU pronto
for (let i = 0; ; i++) {
  await sleep(500);
  const kv = await evaluate(`[...document.querySelectorAll('.web-kv')].map(e => e.innerText).join(' ')`);
  if (kv?.includes("Imagens") && kv.includes("Adaptador")) break;
  if (i > 120) {
    console.error("Timeout carregando a página:", await evaluate("document.body.innerText.slice(0, 400)"));
    process.exit(1);
  }
}

if (mode === "poses") {
  await waitEnabled(posesButton);
  await evaluate(`${posesButton}.click()`);
  let shown = 0;
  for (;;) {
    await sleep(2000);
    const log = await logText();
    if (log.length > shown) {
      console.log(log.slice(shown).trim());
      shown = log.length;
    }
    const err = await errorBanner();
    if (err) {
      console.log("ERRO:", err);
      process.exit(1);
    }
    if (log.includes("Poses prontas")) break;
  }
  const out = opt("out");
  if (out) {
    mkdirSync(join(out, "sparse", "0"), { recursive: true });
    for (const name of ["cameras.bin", "images.bin", "points3D.bin"])
      writeFileSync(join(out, "sparse", "0", name), await fetchBytes(`window.__splatforgeSfm.files['${name}']`));
    const local = new URL(url).searchParams.get("dataset")?.replace(/^\/@fs/, "");
    if (local && !existsSync(join(out, "images"))) symlinkSync(join(local, "images"), join(out, "images"));
    console.log("modelo salvo em", join(out, "sparse", "0"));
  }
  process.exit(0);
}

// train / profile: configura os campos e começa
await evaluate(`(() => {
  const set = (input, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, v);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  };
  const nums = document.querySelectorAll('input.input.num');
  set(nums[0], '${opt("steps", "3000")}');
  set(nums[1], '${opt("max-res", "960")}');
  [...document.querySelectorAll('.segmented button')].find(b => b.textContent === 'Manual').click();
})()`);
await sleep(300);
await waitEnabled(trainButton);
await evaluate(`${trainButton}.click()`);
const t0 = Date.now();

if (mode === "profile") {
  for (;;) {
    await sleep(1000);
    const m = /Passo ([\d.]+)/.exec(await progressLine());
    if (m && Number(m[1].replace(/\./g, "")) >= 100) break;
  }
  await send("Profiler.enable");
  await send("Profiler.setSamplingInterval", { interval: 200 });
  await send("Profiler.start");
  await sleep(Number(opt("seconds", "10")) * 1000);
  const { result } = await send("Profiler.stop");
  const { nodes, samples } = result.profile;
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const self = new Map();
  for (const s of samples) {
    const n = byId.get(s);
    const k = n.callFrame.functionName || n.callFrame.url.split("/").pop() || "(anônima)";
    self.set(k, (self.get(k) ?? 0) + 1);
  }
  console.log(`${samples.length} amostras; tempo próprio por função:`);
  for (const [k, v] of [...self].sort((a, b) => b[1] - a[1]).slice(0, 30))
    console.log(`${((100 * v) / samples.length).toFixed(1).padStart(5)}%  ${k.slice(0, 160)}`);
  process.exit(0);
}

let last = "";
for (;;) {
  await sleep(5000);
  const line = await progressLine();
  if (line !== last) console.log(`${((Date.now() - t0) / 1000).toFixed(0)}s  ${line}`);
  last = line;
  const err = await errorBanner();
  if (err) {
    console.log("ERRO:", err);
    process.exit(1);
  }
  if (await evaluate(`[...document.querySelectorAll('button')].some(b => b.textContent.startsWith('Baixar .ply'))`)) break;
}
console.log((await logText()).split("\n").filter((l) => /Avaliação|concluído/.test(l)).join("\n"));
const out = opt("out");
if (out) {
  writeFileSync(out, await fetchBytes("window.__splatforgePly"));
  console.log("salvo", out);
}
process.exit(0);
