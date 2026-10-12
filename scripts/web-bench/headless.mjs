#!/usr/bin/env node
// Roda a versão web num Chrome headless e mede, sem depender de uma janela
// visível (abas ocultas sofrem throttling). Precisa do `pnpm web:dev`
// rodando com `WEB_FS_ANY=1`, para abrir arquivos locais por /@fs.
//
//   node scripts/web-bench/headless.mjs video <url> [--mask chroma|rmbg|sam2] [--for-poses] [--no-train] [--runs N] [--out <pasta>]
//   node scripts/web-bench/headless.mjs train <url> [--out splat.ply]
//   node scripts/web-bench/headless.mjs poses <url> [--out <pasta>]
//   node scripts/web-bench/headless.mjs profile <url> [--seconds 10]
//
// <url>: http://localhost:1440/?video=/@fs/<vídeo>, mais &dataset=/@fs/<dataset
// COLMAP> para train/poses/profile (o projeto do vídeo começa com esse
// dataset: imagens e poses).
// - video:   "Gerar splat" desde os frames, com o recorte de --mask (rmbg e sam2
//            baixa o modelo se preciso); --no-train cancela ao fim das poses;
//            --runs N repete na mesma página; --out copia o projeto do OPFS
//            para <pasta>, mais um frames.json com o instante de cada frame;
//            --export '{"format":"sog","decimate":0.5}' exporta pelo painel
//            no fim e abre o resultado no 3D.
// - train:   treina a partir da etapa Treino (poses do dataset); --out salva
//            o .ply final.
// - poses:   recalcula as poses a partir da etapa Poses e para; --out grava
//            <pasta>/sparse/0 (e liga <pasta>/images às imagens do dataset).
// - profile: perfil de CPU de alguns segundos de treino (após o passo 100).
//            Use `pnpm web:wasm --profiling` para ter os nomes do WASM.
//
// Ajustes (partem do preset Prévia): --preset preview|balanced|quality,
// --frames N, --steps N, --max-res N, --flat (órbita plana), --fill (usar
// todos os frames; com &sfmDrop=0,20,… na URL, simula câmeras perdidas). O Chrome vem de CHROME (padrão: o do
// macOS em /Applications).

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DEFAULT_CHROMA } from "../../src/lib/api.ts";
import { presetById } from "../../src/lib/presets.ts";

const [mode, url, ...rest] = process.argv.slice(2);
const opt = (name, def) => {
  const i = rest.indexOf(`--${name}`);
  return i >= 0 ? rest[i + 1] : def;
};
const flag = (name) => rest.includes(`--${name}`);
if (!["train", "poses", "profile", "video"].includes(mode) || !url) {
  console.error("uso: headless.mjs video|train|poses|profile <url> [opções]  (ver o cabeçalho do arquivo)");
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
// Cada execução cria um perfil (com cópias dos modelos no OPFS): apaga ao sair,
// senão eles acumulam gigabytes no disco.
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
/** Copia uma pasta do OPFS da página para o disco, recursivamente. */
async function copyOut(handleExpr, dest) {
  const list = await evaluate(`(async () => {
    const out = [];
    const walk = async (d, p) => { for await (const [n, h] of d.entries()) h.kind === 'file' ? out.push(p + n) : await walk(h, p + n + '/'); };
    await walk(${handleExpr}, '');
    return out;
  })()`);
  for (const rel of list) {
    const parts = rel.split("/");
    const expr = `(async () => { let d = ${handleExpr}; for (const p of ${JSON.stringify(parts.slice(0, -1))}) d = await d.getDirectoryHandle(p); return await (await d.getFileHandle(${JSON.stringify(parts.at(-1))})).getFile(); })()`;
    mkdirSync(join(dest, ...parts.slice(0, -1)), { recursive: true });
    writeFileSync(join(dest, ...parts), await fetchBytes(`await ${expr}`));
  }
  return list.length;
}

const ui = {
  run: `[...document.querySelectorAll('.run .btn-primary')].find(b => b.textContent.includes('Gerar splat'))`,
  // "Etapa: mensagem" de cada etapa da barra inferior
  stages: `[...document.querySelectorAll('.stage')].map(e => e.querySelector('.stage-title').textContent + ': ' + e.querySelector('.stage-msg').textContent.trim()).join(' | ')`,
  metrics: `[...document.querySelectorAll('.stage')].map(e => e.querySelector('.stage-title').textContent + ' ' + (e.querySelector('.stage-time')?.textContent ?? '—') + ' ' + [...e.querySelectorAll('.metric')].map(m => m.innerText.replace('\\n', ' ')).join(', ')).join(' | ')`,
  log: `document.querySelector('pre.log')?.innerText ?? ''`,
  error: `document.querySelector('.banner.error')?.innerText`,
  project: `document.querySelector('.panel .path')?.textContent`,
};
const click = (expr) => evaluate(`(${expr})?.click()`);
const fail = (msg) => {
  console.log("ERRO:", msg);
  process.exit(1);
};

// Ajustes guardados pela interface (localStorage da origem) antes de abrir.
const settings = structuredClone(presetById(opt("preset", "preview")).settings);
if (opt("frames")) settings.frames.targetFrames = Number(opt("frames"));
if (opt("steps")) settings.train.totalSteps = Number(opt("steps"));
if (opt("max-res")) settings.train.maxResolution = Number(opt("max-res"));
if (flag("flat")) settings.poses.flatOrbit = true;
if (flag("fill")) settings.poses.fillMissing = true;
const mask = opt("mask");
const masks = {
  enabled: !!mask,
  forPoses: flag("for-poses"),
  method: mask ?? "rmbg",
  chroma: DEFAULT_CHROMA,
  despill: true,
  despillStrength: 100,
};
await send("Runtime.enable");
await send("Page.enable");
await send("Page.navigate", { url: new URL(url).origin });
await sleep(1000);
// Os seletores abaixo usam os textos em pt-BR.
await evaluate(`localStorage.setItem('splatforge.locale.v1', 'pt-BR');
  localStorage.setItem('splatforge.settings.v3', ${JSON.stringify(JSON.stringify({ preset: "custom", settings }))});
  localStorage.setItem('splatforge.masks.v1', ${JSON.stringify(JSON.stringify(masks))});
  localStorage.setItem('splatforge.export.v1', ${JSON.stringify(opt("export", "{}"))})`);
await send("Page.navigate", { url });

// Vídeo aberto, projeto carregado e Brush/WebGPU de pé.
for (let i = 0; ; i++) {
  await sleep(500);
  const ready = await evaluate(`!!document.querySelector('.meta-row') && !!document.querySelector('.stage-tool') && !document.body.innerText.includes('Verificando o que precisa')`);
  if (ready && (await evaluate(`!!document.querySelector('.setup') || !(${ui.run})?.disabled`))) break;
  if (i > 120) fail(`timeout carregando a página: ${await evaluate("document.body.innerText.slice(0, 400)")}`);
}
// Primeiro uso do RMBG: o banner de ferramentas baixa o modelo.
if (await evaluate(`!!document.querySelector('.setup')`)) {
  if (!(await evaluate(`!!document.querySelector('.setup-btn')`))) fail(await evaluate(`document.querySelector('.setup').innerText`));
  console.log("baixando o modelo de recorte…");
  await click(`document.querySelector('.setup-btn')`);
  for (let i = 0; await evaluate(`!!document.querySelector('.setup')`); i++) {
    await sleep(1000);
    const err = await evaluate(`document.querySelector('.setup-error')?.innerText`);
    if (err) fail(err);
    if (i > 600) fail("timeout baixando o modelo");
  }
}

const from = { poses: "Poses", train: "Treino", profile: "Treino" }[mode];
// O dataset de ?dataset= é copiado depois que o vídeo abre.
const segment = `[...document.querySelectorAll('.run .segmented button')].find(b => b.textContent === '${from}')`;
for (let i = 0; from && (await evaluate(`${segment}.disabled`)) && i < 240; i++) await sleep(500);
if (from) await click(segment);
await sleep(300);
if (from && !(await evaluate(`[...document.querySelectorAll('.run .segmented button')].find(b => b.textContent === '${from}').classList.contains('active')`)))
  fail(`a etapa ${from} não está disponível: o dataset tem poses?`);

const stopAfterPoses = mode === "poses" || flag("no-train");
const runs = mode === "video" ? Number(opt("runs", "1")) : 1;
let shown = 0;
const flushLog = async () => {
  const log = (await evaluate(ui.log)) ?? "";
  if (log.length < shown) shown = 0; // nova execução limpa o log
  if (log.length > shown) console.log(log.slice(shown).trim());
  shown = log.length;
};

for (let run = 1; run <= runs; run++) {
  for (let i = 0; await evaluate(`(${ui.run})?.disabled ?? true`); i++) {
    if (i > 120) fail("o botão Gerar splat não habilitou");
    await sleep(500);
  }
  await click(ui.run);
  await sleep(300);
  await click(`[...document.querySelectorAll('.tabs button')].find(b => b.textContent === 'Log')`);
  const t0 = Date.now();
  let lastStages = "", lastPrint = 0;

  if (mode === "profile") {
    for (;;) {
      await sleep(1000);
      const m = /Passo (\d+)\//.exec((await evaluate(ui.stages)) ?? "");
      if (m && Number(m[1]) >= 100) break;
      const err = await evaluate(ui.error);
      if (err) fail(err);
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

  for (;;) {
    await sleep(2000);
    await flushLog();
    const stages = (await evaluate(ui.stages)) ?? "";
    const active = stages.split(" | ").find((x) => !/: (|Na fila|Resultado salvo no projeto)$/.test(x) && !/concluído|selecionados|recortes ·|Registradas|desativado|reaproveitadas/.test(x)) ?? "";
    if (active && active !== lastStages && Date.now() - lastPrint > 10_000) {
      console.log(`  [${((Date.now() - t0) / 1000).toFixed(0)}s] ${active}`);
      lastStages = active;
      lastPrint = Date.now();
    }
    const err = await evaluate(ui.error);
    if (err) fail(err);
    if (stopAfterPoses && (await evaluate(`document.querySelectorAll('.stage')[2].classList.contains('stage-done')`))) {
      await click(`document.querySelector('.run .btn-primary.danger')`);
      await sleep(1000);
      break;
    }
    if (await evaluate(`!!document.querySelector('.done-note')`)) break;
  }
  await flushLog();
  console.log(`execução ${run}: ${((Date.now() - t0) / 1000).toFixed(1)} s\n  ${(await evaluate(ui.metrics)).replaceAll(" | ", "\n  ")}`);
}

if (opt("export") && !stopAfterPoses) {
  const t0 = Date.now();
  await click(`[...document.querySelectorAll('.export-run')].find(b => b.textContent.startsWith('Exportar'))`);
  for (let i = 0; !(await evaluate(`!!document.querySelector('.export-result') && !document.querySelector('.export-progress')`)); i++) {
    const err = await evaluate(`document.querySelector('.export-error')?.textContent`);
    if (err) fail(`exportação: ${err}`);
    if (i > 600) fail("timeout na exportação");
    await sleep(500);
  }
  console.log(`exportação em ${((Date.now() - t0) / 1000).toFixed(1)} s: ${(await evaluate(`document.querySelector('.export-result').innerText`)).replace(/\n+/g, " · ")}`);
  // A exportação já abre no 3D: espera o Spark carregar e confere o rótulo.
  await click(`[...document.querySelectorAll('.tabs button')].find(b => b.textContent === '3D')`);
  await sleep(4000);
  const viewer = await evaluate(`document.querySelector('.content').innerText`);
  if (/erro|falh/i.test(viewer)) fail(`visualizador: ${viewer}`);
  console.log(`visualizador: ${viewer.replace(/\n+/g, " · ").slice(0, 120)}`);
}

const project = await evaluate(ui.project);
const opfs = (path) => `(async () => { let d = await navigator.storage.getDirectory(); for (const p of ${JSON.stringify(path)}.split('/').filter(Boolean)) d = await d.getDirectoryHandle(p); return d; })()`;
const opfsFile = (path) => {
  const parts = path.split("/").filter(Boolean);
  return `(async () => { let d = await navigator.storage.getDirectory(); for (const p of ${JSON.stringify(parts.slice(0, -1))}) d = await d.getDirectoryHandle(p); return await (await d.getFileHandle(${JSON.stringify(parts.at(-1))})).getFile(); })()`;
};
const out = opt("out");
if (out && mode === "video") {
  console.log(`${await copyOut(`await ${opfs(project)}`, out)} arquivos copiados para ${out}`);
  // nome de cada frame e o instante do vídeo de onde saiu (para comparar com a verdade)
  const state = JSON.parse((await fetchBytes(`await ${opfsFile(`${project}/project.json`)}`)).toString());
  const frames = state.frames.map((f) => ({ name: f.path.split("/").pop(), time: f.time, sharpness: f.sharpness }));
  writeFileSync(join(out, "frames.json"), JSON.stringify(frames, null, 1));
}
if (out && mode === "poses") {
  mkdirSync(join(out, "sparse", "0"), { recursive: true });
  for (const name of ["cameras.bin", "images.bin", "points3D.bin"])
    writeFileSync(join(out, "sparse", "0", name), await fetchBytes(`await ${opfsFile(`${project}/dataset/sparse/0/${name}`)}`));
  const local = new URL(url).searchParams.get("dataset")?.replace(/^\/@fs/, "");
  if (local && !existsSync(join(out, "images"))) symlinkSync(join(local, "images"), join(out, "images"));
  console.log("modelo salvo em", join(out, "sparse", "0"));
}
if (out && mode === "train") {
  writeFileSync(out, await fetchBytes(`await ${opfsFile(`${project}/splat.ply`)}`));
  console.log("salvo", out);
}
process.exit(0);
