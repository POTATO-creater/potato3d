#!/usr/bin/env node
// Downloads and verifies (SHA-256) Gausify Orbit's external tools.
//
//   pnpm setup:tools            installs whatever is missing
//   pnpm setup:tools --force    downloads everything again
//   pnpm setup:tools --cpu      (Windows) COLMAP without CUDA even with an NVIDIA GPU
//   pnpm setup:tools --only bundle   only what ships inside the installer (CI)
//
// - Brush (all platforms): Tauri sidecar in src-tauri/binaries/, built from
//   source by scripts/build-brush-web.mjs --native.
// - Windows: DirectML.dll 1.15.4 (Microsoft's NuGet), used by ONNX Runtime
//   for masking; ships next to the .exe in the installer.
// - Windows: COLMAP (with CUDA if there's an NVIDIA GPU) and ffmpeg in the
//   app's tools folder, where the backend looks (see src-tauri/src/tools.rs).
// - macOS/Linux: COLMAP and ffmpeg come from the package manager.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createWriteStream, existsSync } from "node:fs";
import { copyFile, mkdir, mkdtemp, readdir, rename, rm, stat } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// Pinned URLs and SHA-256: no dependency on the GitHub API, whose anonymous
// limit (60/h per IP) runs out on shared CI runners.
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
// Same version ONNX Runtime 1.28 (`ort` crate) expects; hashes verified.
const DIRECTML = {
  url: "https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg",
  name: "microsoft.ai.directml.1.15.4.nupkg",
  size: 202292617,
  sha256: "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9",
  dllSha256: "9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1",
};
const COLMAP = { repo: "colmap/colmap", tag: "4.2.1" };
// Only ffmpeg ("latest", rebuilt daily) still queries the API; in CI the
// GITHUB_TOKEN raises the limit.
const ghHeaders = {
  Accept: "application/vnd.github+json",
  "User-Agent": "gausify-orbit-setup",
  ...((process.env.GITHUB_TOKEN || process.env.GH_TOKEN) && {
    Authorization: `Bearer ${process.env.GITHUB_TOKEN || process.env.GH_TOKEN}`,
  }),
};
// BtbN publishes stable ffmpeg builds under the "latest" tag; the name includes the version.
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

function has(cmd, argv = ["-version"]) {
  return spawnSync(cmd, argv, { stdio: "ignore", shell: false }).status === 0;
}

async function releaseAsset(repo, tag, match) {
  const url = `https://api.github.com/repos/${repo}/releases/tags/${tag}`;
  const res = await fetch(url, { headers: ghHeaders });
  if (!res.ok) throw new Error(`GitHub answered ${res.status} for ${url}`);
  const release = await res.json();
  const assets = release.assets.filter((a) => (typeof match === "string" ? a.name === match : match.test(a.name)));
  if (!assets.length) throw new Error(`no file in ${repo}@${tag} matches ${match}`);
  // For patterns (ffmpeg), take the highest version.
  assets.sort((a, b) => b.name.localeCompare(a.name, undefined, { numeric: true }));
  const asset = assets[0];
  let sha256 = asset.digest?.startsWith("sha256:") ? asset.digest.slice(7) : null;
  if (!sha256) {
    const side = release.assets.find((a) => a.name === `${asset.name}.sha256`);
    if (side) sha256 = (await (await fetch(side.browser_download_url)).text()).trim().split(/\s+/)[0];
  }
  if (!sha256) throw new Error(`no published checksum for ${asset.name}; aborting for safety`);
  return { name: asset.name, url: asset.browser_download_url, size: asset.size, sha256 };
}

async function download(asset, dir) {
  const dest = join(dir, asset.name);
  const res = await fetch(asset.url);
  if (!res.ok) throw new Error(`download failed (${res.status}): ${asset.url}`);
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
      process.stdout.write(`\r  ${asset.name}: ${pct}% of ${(asset.size / 1e6).toFixed(0)} MB`);
    }
  }
  await new Promise((r, j) => out.end((e) => (e ? j(e) : r())));
  process.stdout.write("\n");
  const got = hash.digest("hex");
  if (got !== asset.sha256) throw new Error(`SHA-256 mismatch for ${asset.name}\n  expected ${asset.sha256}\n  got      ${got}`);
  return dest;
}

function extract(archive, into) {
  // bsdtar (macOS and Windows 10+) opens .zip and .tar.xz.
  const r = spawnSync("tar", ["-xf", archive, "-C", into], { stdio: "inherit" });
  if (r.status !== 0) throw new Error(`could not extract ${archive}`);
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

// Brush is built from the same commit and patches as the web version (the
// releases lack the training options the app uses); rebuilds only when they change.
function setupBrush() {
  const script = join(root, "scripts", "build-brush-web.mjs");
  const r = spawnSync(process.execPath, [script, "--native", ...(force ? [] : ["--if-stale"])], { stdio: "inherit" });
  if (r.status !== 0) throw new Error("failed to build Brush");
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
    return console.log(`✓ DirectML.dll already present (${dest})`);
  }
  console.log("→ DirectML 1.15.4 (NuGet)");
  await withTemp(async (tmp) => {
    const pkg = await download(DIRECTML, tmp);
    const out = join(tmp, "x");
    await mkdir(out);
    extract(pkg, out);
    const dll = join(out, "bin", "x64-win", "DirectML.dll");
    if ((await sha256File(dll)) !== DIRECTML.dllSha256) throw new Error("DirectML.dll SHA-256 mismatch");
    await mkdir(dirname(dest), { recursive: true });
    await copyFile(dll, dest);
  });
  console.log(`✓ DirectML.dll at ${dest}`);
}

/** Installs a package containing `bin/<exe>` into `<toolsRoot>/<name>`. */
async function setupWinPackage(name, label, asset, exe) {
  const target = join(toolsRoot(), name);
  if (existsSync(join(target, "bin", exe)) && !force) return console.log(`✓ ${label} already installed (${target})`);
  console.log(`→ ${label} (${asset.name})`);
  await withTemp(async (tmp) => {
    const archive = await download(asset, tmp);
    const out = join(tmp, "x");
    await mkdir(out);
    extract(archive, out);
    const bin = await findFile(out, exe);
    if (!bin) throw new Error(`${exe} not found in the package`);
    const pkgRoot = dirname(dirname(bin)); // .../<package>/bin/<exe>
    await rm(target, { recursive: true, force: true });
    await mkdir(dirname(target), { recursive: true });
    await rename(pkgRoot, target).catch(async () => {
      // rename across volumes fails; copy recursively.
      spawnSync("robocopy", [pkgRoot, target, "/E", "/NFL", "/NDL", "/NJH", "/NJS"], { stdio: "inherit" });
      await stat(join(target, "bin", exe));
    });
  });
  console.log(`✓ ${label} at ${target}`);
}

async function setupWindowsTools() {
  const cuda = !args.has("--cpu") && has("nvidia-smi", ["-L"]);
  const colmapAsset = pinned(COLMAP.repo, COLMAP.tag, COLMAP_ASSETS[cuda ? "cuda" : "cpu"]);
  console.log(cuda ? "NVIDIA GPU detected: COLMAP with CUDA." : "No NVIDIA GPU (or --cpu): COLMAP without CUDA.");
  await setupWinPackage("colmap", `COLMAP ${COLMAP.tag}`, colmapAsset, "colmap.exe");
  await setupWinPackage("ffmpeg", "ffmpeg", await releaseAsset(FFMPEG.repo, FFMPEG.tag, FFMPEG.pattern), "ffmpeg.exe");
}

function checkPackageManagerTools() {
  const missing = [["ffmpeg", ["-version"]], ["colmap", ["help"]]].filter(([c, a]) => !has(c, a)).map(([c]) => c);
  if (!missing.length) return console.log("✓ ffmpeg and COLMAP found");
  const cmd = isMac ? `brew install ${missing.join(" ")}` : `install via your package manager: ${missing.join(", ")}`;
  console.log(`! Missing: ${missing.join(", ")} → ${cmd}`);
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
