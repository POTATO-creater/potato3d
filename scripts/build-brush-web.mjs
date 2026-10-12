#!/usr/bin/env node
// Compila o Brush do commit fixo, com os patches de web/brush/: em
// WebAssembly (bindings `brush-js`) para a versão web, ou nativo (`brush-cli`)
// como sidecar do desktop. Assim os dois treinam com o mesmo Brush.
//
//   pnpm web:wasm             clona/atualiza o Brush, aplica o patch e compila
//   pnpm web:wasm --dev       build de debug (mais rápido, wasm bem maior)
//   pnpm web:wasm --profiling otimizado e com nomes de função, para perfilar
//   pnpm brush:native         sidecar do desktop em src-tauri/binaries/
//   pnpm brush:native --if-stale   só compila se o commit ou os patches mudaram
//
// - Fonte: github.com/ArthurBrussee/brush num commit fixo, em web/.brush/.
// - Patch: web/brush/*.patch, consertos do brush-js para rodar de verdade no
//   navegador (ver README, "Versão web") e as opções de treino do app.
// - Saída: web/pkg/ (módulo ES + .wasm, alvo `web` do wasm-pack), ou
//   src-tauri/binaries/brush_app-<triple> mais um `.rev` com o carimbo do
//   commit e dos patches.
// - Requer rustup (e wasm-pack, no web) e Rust >= 1.95 (o Burn do Brush
//   exige). Usa o toolchain de BRUSH_TOOLCHAIN (padrão 1.99) sem mexer no
//   `stable`.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { chmod, copyFile, mkdir, readdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const BRUSH = {
  repo: "https://github.com/ArthurBrussee/brush.git",
  // main de 2026-10-03, #545 ("Train in metres"). O seguinte (#546, fusão do
  // passo de treino) leva o kernel project_forward a 12 storage buffers, acima
  // dos 10 por estágio que o Chrome expõe no Metal: o treino não roda no Mac.
  rev: "38185dedaf8143988a31a64c1f27c2323b1febdf",
};
const TOOLCHAIN = process.env.BRUSH_TOOLCHAIN || "1.99";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "web", ".brush");
const patches = join(root, "web", "brush");
const out = join(root, "web", "pkg");
const profile = process.argv.includes("--dev") ? "--dev" : process.argv.includes("--profiling") ? "--profiling" : "--release";
const native = process.argv.includes("--native");
const env = { ...process.env, RUSTUP_TOOLCHAIN: TOOLCHAIN };

function run(cmd, args, opts = {}) {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  const r = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  if (r.error) throw r.error;
  if (r.status !== 0) process.exit(r.status ?? 1);
}

const patchFiles = (await readdir(patches)).filter((f) => f.endsWith(".patch")).sort();

// Sidecar do desktop: o nome `brush_app-<triple>` é o que o Tauri empacota
// (externalBin) e o que src-tauri/src/tools.rs procura.
let sidecar = null;
if (native) {
  const host = spawnSync("rustc", ["-vV"], { encoding: "utf8" }).stdout?.match(/^host: (.+)$/m)?.[1].trim();
  if (!host) throw new Error("rustc não encontrado; instale o Rust (https://rustup.rs).");
  const ext = process.platform === "win32" ? ".exe" : "";
  const hash = createHash("sha256").update(BRUSH.rev);
  for (const p of patchFiles) hash.update(p).update(readFileSync(join(patches, p)));
  sidecar = { dest: join(root, "src-tauri", "binaries", `brush_app-${host}${ext}`), ext, stamp: hash.digest("hex") };
  const current = existsSync(`${sidecar.dest}.rev`) ? readFileSync(`${sidecar.dest}.rev`, "utf8").trim() : null;
  if (process.argv.includes("--if-stale") && existsSync(sidecar.dest) && current === sidecar.stamp) {
    console.log(`✓ Brush já compilado deste commit e patches (${sidecar.dest})`);
    process.exit(0);
  }
}

if (!existsSync(join(src, ".git"))) {
  run("git", ["init", "-q", src]);
  run("git", ["-C", src, "remote", "add", "origin", BRUSH.repo]);
}
const head = spawnSync("git", ["-C", src, "rev-parse", "HEAD"], { encoding: "utf8" }).stdout.trim();
if (head !== BRUSH.rev) run("git", ["-C", src, "fetch", "-q", "--depth", "1", "origin", BRUSH.rev]);
// O clone é só cache do build: volta sempre ao commit fixo e reaplica os patches.
run("git", ["-C", src, "checkout", "-q", "--force", BRUSH.rev]);
run("git", ["-C", src, "clean", "-q", "-fd", "apps", "crates"]);
for (const p of patchFiles) run("git", ["-C", src, "apply", join(patches, p)]);

if (sidecar) {
  run("rustup", ["toolchain", "install", TOOLCHAIN, "--profile", "minimal"]);
  // Sem símbolos de depuração: vai dentro do instalador (143 → 105 MB no Mac).
  run("cargo", ["build", "--release", "-p", "brush-cli"], { cwd: src, env: { ...env, CARGO_PROFILE_RELEASE_STRIP: "symbols" } });
  await mkdir(dirname(sidecar.dest), { recursive: true });
  await copyFile(join(src, "target", "release", `brush-cli${sidecar.ext}`), sidecar.dest);
  if (process.platform !== "win32") await chmod(sidecar.dest, 0o755);
  await writeFile(`${sidecar.dest}.rev`, `${sidecar.stamp}\n`);
  console.log(`✓ Brush em ${sidecar.dest}`);
  process.exit(0);
}

run("rustup", ["toolchain", "install", TOOLCHAIN, "--profile", "minimal", "--target", "wasm32-unknown-unknown"]);
run(
  "wasm-pack",
  // --profiling sem wasm-opt: o -O dele apaga a seção `name` e o perfil sai sem nomes.
  ["build", "apps/brush-js", profile, ...(profile === "--profiling" ? ["--no-opt"] : []), "--target", "web", "--out-dir", out],
  { cwd: src, env },
);
