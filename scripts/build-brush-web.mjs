#!/usr/bin/env node
// Compila o Brush para WebAssembly (bindings `brush-js`) para a versão web.
//
//   pnpm web:wasm             clona/atualiza o Brush, aplica o patch e compila
//   pnpm web:wasm --dev       build de debug (mais rápido, wasm bem maior)
//   pnpm web:wasm --profiling otimizado e com nomes de função, para perfilar
//
// - Fonte: github.com/ArthurBrussee/brush num commit fixo, em web/.brush/.
// - Patch: web/brush/*.patch, consertos do brush-js para rodar de verdade no
//   navegador (ver README, "Versão web").
// - Saída: web/pkg/ (módulo ES + .wasm, alvo `web` do wasm-pack).
// - Requer rustup, wasm-pack e Rust >= 1.95 (o Burn do Brush exige). Usa o
//   toolchain de BRUSH_TOOLCHAIN (padrão 1.99) sem mexer no `stable`.

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readdir } from "node:fs/promises";
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

function run(cmd, args, opts = {}) {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  const r = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  if (r.error) throw r.error;
  if (r.status !== 0) process.exit(r.status ?? 1);
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
for (const p of (await readdir(patches)).filter((f) => f.endsWith(".patch")).sort()) {
  run("git", ["-C", src, "apply", join(patches, p)]);
}

run("rustup", ["toolchain", "install", TOOLCHAIN, "--profile", "minimal", "--target", "wasm32-unknown-unknown"]);
run(
  "wasm-pack",
  // --profiling sem wasm-opt: o -O dele apaga a seção `name` e o perfil sai sem nomes.
  ["build", "apps/brush-js", profile, ...(profile === "--profiling" ? ["--no-opt"] : []), "--target", "web", "--out-dir", out],
  { cwd: src, env: { ...process.env, RUSTUP_TOOLCHAIN: TOOLCHAIN } },
);
