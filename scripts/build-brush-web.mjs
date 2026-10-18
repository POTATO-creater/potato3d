#!/usr/bin/env node
// Builds Brush from the pinned commit with the patches in web/brush/ as
// WebAssembly (`brush-js` bindings) for the app.
//
//   pnpm web:wasm             clones/updates Brush, applies the patches and builds
//   pnpm web:wasm --dev       debug build (faster, much larger wasm)
//   pnpm web:wasm --profiling optimized with function names, for profiling
//
// - Source: github.com/ArthurBrussee/brush at a pinned commit, in web/.brush/.
// - Patches: web/brush/*.patch, brush-js fixes needed to actually run in the
//   browser (see README, "Web version") and the app's training options.
// - Output: web/pkg/ (ES module + .wasm, wasm-pack `web` target).
// - Requires rustup, wasm-pack and Rust >= 1.95 (Brush's Burn
//   needs it). Uses the BRUSH_TOOLCHAIN toolchain (default 1.99) without
//   touching `stable`.

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const BRUSH = {
  repo: "https://github.com/ArthurBrussee/brush.git",
  // main as of 2026-10-03, #545 ("Train in metres"). The next one (#546, fused
  // training step) takes the project_forward kernel to 12 storage buffers, over
  // the 10 per stage Chrome exposes on Metal: training doesn't run on a Mac.
  rev: "38185dedaf8143988a31a64c1f27c2323b1febdf",
};
const TOOLCHAIN = process.env.BRUSH_TOOLCHAIN || "1.99";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "web", ".brush");
const patches = join(root, "web", "brush");
const out = join(root, "web", "pkg");
const profile = process.argv.includes("--dev") ? "--dev" : process.argv.includes("--profiling") ? "--profiling" : "--release";
const env = { ...process.env, RUSTUP_TOOLCHAIN: TOOLCHAIN };

function run(cmd, args, opts = {}) {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  const r = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  if (r.error) throw r.error;
  if (r.status !== 0) process.exit(r.status ?? 1);
}

const patchFiles = (await readdir(patches)).filter((f) => f.endsWith(".patch")).sort();

if (!existsSync(join(src, ".git"))) {
  run("git", ["init", "-q", src]);
  run("git", ["-C", src, "remote", "add", "origin", BRUSH.repo]);
}
const head = spawnSync("git", ["-C", src, "rev-parse", "HEAD"], { encoding: "utf8" }).stdout.trim();
if (head !== BRUSH.rev) run("git", ["-C", src, "fetch", "-q", "--depth", "1", "origin", BRUSH.rev]);
// The clone is just a build cache: always reset to the pinned commit and reapply the patches.
run("git", ["-C", src, "checkout", "-q", "--force", BRUSH.rev]);
run("git", ["-C", src, "clean", "-q", "-fd", "apps", "crates"]);
for (const p of patchFiles) run("git", ["-C", src, "apply", join(patches, p)]);

run("rustup", ["toolchain", "install", TOOLCHAIN, "--profile", "minimal", "--target", "wasm32-unknown-unknown"]);
run(
  "wasm-pack",
  // --profiling without wasm-opt: its -O strips the `name` section and profiles lose names.
  ["build", "apps/brush-js", profile, ...(profile === "--profiling" ? ["--no-opt"] : []), "--target", "web", "--out-dir", out],
  { cwd: src, env },
);
