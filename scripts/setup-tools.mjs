#!/usr/bin/env node
// Baixa e verifica (SHA-256) as ferramentas externas do Gausify Orbit.
//
//   pnpm setup:tools            instala o que falta
//   pnpm setup:tools --force    baixa tudo de novo
//   pnpm setup:tools --cpu      (Windows) COLMAP sem CUDA mesmo com GPU NVIDIA
//   pnpm setup:tools --only bundle   só o que vai dentro do instalador (CI)
//
// - Brush (todas as plataformas): sidecar do Tauri em src-tauri/binaries/.
// - Windows: DirectML.dll 1.15.4 (NuGet da Microsoft), usado pelo ONNX
//   Runtime no recorte; vai junto do .exe no instalador.
// - Windows: COLMAP (com CUDA se houver GPU NVIDIA) e ffmpeg na pasta de
//   ferramentas do app, onde o backend procura (ver src-tauri/src/tools.rs).
// - macOS/Linux: COLMAP e ffmpeg vêm do gerenciador de pacotes.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createWriteStream, existsSync } from "node:fs";
import { chmod, copyFile, mkdir, mkdtemp, readdir, rename, rm, stat } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// URLs e SHA-256 fixos: não depende da API do GitHub, cujo limite anônimo
// (60/h por IP) estoura nos runners compartilhados do CI.
const BRUSH = {
  tag: "v0.3.0",
  assets: {
    "aarch64-apple-darwin": ["brush-app-aarch64-apple-darwin.tar.xz", 41447012, "65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c"],
    "x86_64-pc-windows-msvc": ["brush-app-x86_64-pc-windows-msvc.zip", 158791348, "b68e3e9cf052d51bf3ee30776fa5a364de7f2ba13b58443128ff797bb7bcfcd6"],
    "x86_64-unknown-linux-gnu": ["brush-app-x86_64-unknown-linux-gnu.tar.xz", 44242224, "4f0f9a8785d1951c62df26aae247c02c5bba32b00f40b06df4e1c9b867399e20"],
  },
};
const COLMAP_ASSETS = {
  cuda: ["colmap-x64-windows-cuda.zip", 414666626, "e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881"],
  cpu: ["colmap-x64-windows-nocuda.zip", 127731595, "c493d88cb4a43f21afba42cb55bf786e5434a0420e73db0ba1454c5d8b3686ae"],
};

const pinned = (repo, tag, [name, size, sha256]) => ({
  name,
  size,
  sha256,
  url: `https://github.com/${repo}/releases/download/${tag}/${name}`,
});
// Mesma versão que o ONNX Runtime 1.28 (crate `ort`) espera; hashes conferidos.
const DIRECTML = {
  url: "https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg",
  name: "microsoft.ai.directml.1.15.4.nupkg",
  size: 202292617,
  sha256: "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9",
  dllSha256: "9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1",
};
const COLMAP = { repo: "colmap/colmap", tag: "4.2.1" };
// Só o ffmpeg ("latest", refeito todo dia) ainda consulta a API; no CI o
// GITHUB_TOKEN eleva o limite.
const ghHeaders = {
  Accept: "application/vnd.github+json",
  "User-Agent": "gausify-orbit-setup",
  ...((process.env.GITHUB_TOKEN || process.env.GH_TOKEN) && {
    Authorization: `Bearer ${process.env.GITHUB_TOKEN || process.env.GH_TOKEN}`,
  }),
};
// BtbN publica builds estáveis do ffmpeg na tag "latest"; o nome inclui a versão.
const FFMPEG = { repo: "BtbN/FFmpeg-Builds", tag: "latest", pattern: /^ffmpeg-n([\d.]+)-latest-win64-lgpl-[\d.]+\.zip$/ };

const args = new Set(process.argv.slice(2));
const only = process.argv.includes("--only") ? process.argv[process.argv.indexOf("--only") + 1] : null;
const force = args.has("--force");
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const isWin = process.platform === "win32";
const isMac = process.platform === "darwin";

function toolsRoot() {
  if (isWin) return join(process.env.LOCALAPPDATA ?? join(homedir(), "AppData", "Local"), "SplatForge", "tools");
  if (isMac) return join(homedir(), "Library", "Application Support", "SplatForge", "tools");
  return join(process.env.XDG_DATA_HOME ?? join(homedir(), ".local", "share"), "SplatForge", "tools");
}

function hostTriple() {
  const r = spawnSync("rustc", ["-vV"], { encoding: "utf8" });
  const m = r.stdout?.match(/^host: (.+)$/m);
  if (!m) throw new Error("rustc não encontrado; instale o Rust (https://rustup.rs).");
  return m[1].trim();
}

function has(cmd, argv = ["-version"]) {
  return spawnSync(cmd, argv, { stdio: "ignore", shell: false }).status === 0;
}

async function releaseAsset(repo, tag, match) {
  const url = `https://api.github.com/repos/${repo}/releases/tags/${tag}`;
  const res = await fetch(url, { headers: ghHeaders });
  if (!res.ok) throw new Error(`GitHub respondeu ${res.status} para ${url}`);
  const release = await res.json();
  const assets = release.assets.filter((a) => (typeof match === "string" ? a.name === match : match.test(a.name)));
  if (!assets.length) throw new Error(`nenhum arquivo de ${repo}@${tag} corresponde a ${match}`);
  // Para padrões (ffmpeg), pega a maior versão.
  assets.sort((a, b) => b.name.localeCompare(a.name, undefined, { numeric: true }));
  const asset = assets[0];
  let sha256 = asset.digest?.startsWith("sha256:") ? asset.digest.slice(7) : null;
  if (!sha256) {
    const side = release.assets.find((a) => a.name === `${asset.name}.sha256`);
    if (side) sha256 = (await (await fetch(side.browser_download_url)).text()).trim().split(/\s+/)[0];
  }
  if (!sha256) throw new Error(`sem checksum publicado para ${asset.name}; abortando por segurança`);
  return { name: asset.name, url: asset.browser_download_url, size: asset.size, sha256 };
}

async function download(asset, dir) {
  const dest = join(dir, asset.name);
  const res = await fetch(asset.url);
  if (!res.ok) throw new Error(`download falhou (${res.status}): ${asset.url}`);
  const hash = createHash("sha256");
  const out = createWriteStream(dest);
  let done = 0;
  let lastPct = -1;
  for await (const chunk of res.body) {
    hash.update(chunk);
    done += chunk.length;
    if (!out.write(chunk)) await new Promise((r) => out.once("drain", r));
    const pct = Math.floor((done / asset.size) * 100);
    if (pct !== lastPct && pct % 5 === 0) {
      lastPct = pct;
      process.stdout.write(`\r  ${asset.name}: ${pct}% de ${(asset.size / 1e6).toFixed(0)} MB`);
    }
  }
  await new Promise((r, j) => out.end((e) => (e ? j(e) : r())));
  process.stdout.write("\n");
  const got = hash.digest("hex");
  if (got !== asset.sha256) throw new Error(`SHA-256 não confere para ${asset.name}\n  esperado ${asset.sha256}\n  obtido   ${got}`);
  return dest;
}

function extract(archive, into) {
  // bsdtar (macOS e Windows 10+) abre .zip e .tar.xz.
  const r = spawnSync("tar", ["-xf", archive, "-C", into], { stdio: "inherit" });
  if (r.status !== 0) throw new Error(`não foi possível extrair ${archive}`);
}

async function findFile(dir, name) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) {
      const hit = await findFile(p, name);
      if (hit) return hit;
    } else if (entry.name.toLowerCase() === name.toLowerCase()) {
      return p;
    }
  }
  return null;
}

async function withTemp(fn) {
  const tmp = await mkdtemp(join(tmpdir(), "splatforge-"));
  try {
    return await fn(tmp);
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
}

async function setupBrush() {
  const triple = hostTriple();
  const ext = isWin ? ".exe" : "";
  const dest = join(root, "src-tauri", "binaries", `brush_app-${triple}${ext}`);
  if (existsSync(dest) && !force) return console.log(`✓ Brush já instalado (${dest})`);
  const asset = BRUSH.assets[triple];
  if (!asset) throw new Error(`sem binário pronto do Brush para ${triple}; compile a partir do código-fonte`);
  console.log(`→ Brush ${BRUSH.tag}`);
  await withTemp(async (tmp) => {
    const archive = await download(pinned("ArthurBrussee/brush", BRUSH.tag, asset), tmp);
    extract(archive, tmp);
    const bin = await findFile(tmp, `brush_app${ext}`);
    if (!bin) throw new Error("brush_app não encontrado no pacote");
    await mkdir(dirname(dest), { recursive: true });
    await copyFile(bin, dest);
    if (!isWin) await chmod(dest, 0o755);
    if (isMac) spawnSync("xattr", ["-d", "com.apple.quarantine", dest], { stdio: "ignore" });
  });
  console.log(`✓ Brush em ${dest}`);
}

async function sha256File(path) {
  const { createReadStream } = await import("node:fs");
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

async function setupDirectML() {
  const dest = join(root, "src-tauri", "binaries", "DirectML.dll");
  if (existsSync(dest) && !force && (await sha256File(dest)) === DIRECTML.dllSha256) {
    return console.log(`✓ DirectML.dll já presente (${dest})`);
  }
  console.log("→ DirectML 1.15.4 (NuGet)");
  await withTemp(async (tmp) => {
    const pkg = await download(DIRECTML, tmp);
    const out = join(tmp, "x");
    await mkdir(out);
    extract(pkg, out);
    const dll = join(out, "bin", "x64-win", "DirectML.dll");
    if ((await sha256File(dll)) !== DIRECTML.dllSha256) throw new Error("SHA-256 do DirectML.dll não confere");
    await mkdir(dirname(dest), { recursive: true });
    await copyFile(dll, dest);
  });
  console.log(`✓ DirectML.dll em ${dest}`);
}

/** Instala um pacote cujo conteúdo tem `bin/<exe>` em `<toolsRoot>/<name>`. */
async function setupWinPackage(name, label, asset, exe) {
  const target = join(toolsRoot(), name);
  if (existsSync(join(target, "bin", exe)) && !force) return console.log(`✓ ${label} já instalado (${target})`);
  console.log(`→ ${label} (${asset.name})`);
  await withTemp(async (tmp) => {
    const archive = await download(asset, tmp);
    const out = join(tmp, "x");
    await mkdir(out);
    extract(archive, out);
    const bin = await findFile(out, exe);
    if (!bin) throw new Error(`${exe} não encontrado no pacote`);
    const pkgRoot = dirname(dirname(bin)); // .../<pacote>/bin/<exe>
    await rm(target, { recursive: true, force: true });
    await mkdir(dirname(target), { recursive: true });
    await rename(pkgRoot, target).catch(async () => {
      // rename entre volumes falha; copia recursivamente.
      spawnSync("robocopy", [pkgRoot, target, "/E", "/NFL", "/NDL", "/NJH", "/NJS"], { stdio: "inherit" });
      await stat(join(target, "bin", exe));
    });
  });
  console.log(`✓ ${label} em ${target}`);
}

async function setupWindowsTools() {
  const cuda = !args.has("--cpu") && has("nvidia-smi", ["-L"]);
  const colmapAsset = pinned(COLMAP.repo, COLMAP.tag, COLMAP_ASSETS[cuda ? "cuda" : "cpu"]);
  console.log(cuda ? "GPU NVIDIA detectada: COLMAP com CUDA." : "Sem GPU NVIDIA (ou --cpu): COLMAP sem CUDA.");
  await setupWinPackage("colmap", `COLMAP ${COLMAP.tag}`, colmapAsset, "colmap.exe");
  await setupWinPackage("ffmpeg", "ffmpeg", await releaseAsset(FFMPEG.repo, FFMPEG.tag, FFMPEG.pattern), "ffmpeg.exe");
}

function checkPackageManagerTools() {
  const missing = [["ffmpeg", ["-version"]], ["colmap", ["help"]]].filter(([c, a]) => !has(c, a)).map(([c]) => c);
  if (!missing.length) return console.log("✓ ffmpeg e COLMAP encontrados");
  const cmd = isMac ? `brew install ${missing.join(" ")}` : `instale pelo gerenciador de pacotes: ${missing.join(", ")}`;
  console.log(`! Faltam: ${missing.join(", ")} → ${cmd}`);
}

try {
  if (!only || only === "brush" || only === "bundle") await setupBrush();
  if (isWin && (!only || only === "bundle")) await setupDirectML();
  if (!only || only === "deps") {
    if (isWin) await setupWindowsTools();
    else checkPackageManagerTools();
  }
} catch (e) {
  console.error(`\n✗ ${e.message}`);
  process.exit(1);
}
