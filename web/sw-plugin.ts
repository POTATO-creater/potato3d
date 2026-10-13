// Generates the build's sw.js: web/src/sw.template.js prefixed with the list of
// output files and a version (content hash). Walks the output folder after it is
// written, so it also picks up the worker files and public/.

import { createHash } from "node:crypto";
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative, resolve, sep } from "node:path";
import type { Plugin, ResolvedConfig } from "vite";

const SW = "sw.js";
// Cloudflare rules (web/public/_headers): read at deploy time, never served.
const NOT_SERVED = new Set([SW, "_headers", "_redirects"]);

function walk(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)],
  );
}

export function serviceWorker(template: string): Plugin {
  let config: ResolvedConfig;
  return {
    name: "gausify-service-worker",
    apply: "build",
    configResolved(c) {
      config = c;
    },
    closeBundle() {
      const out = resolve(config.root, config.build.outDir);
      const files = walk(out)
        .map((f) => relative(out, f).split(sep).join("/"))
        .filter((f) => !NOT_SERVED.has(f) && !f.endsWith(".map") && !f.startsWith("."))
        .sort();
      // The service worker's own code is hashed too: changing only it is a new version.
      const hash = createHash("sha256").update(readFileSync(template));
      for (const f of files) hash.update(f).update(readFileSync(join(out, f)));
      const build = { version: hash.digest("hex").slice(0, 16), files };
      const source = `const SW_BUILD = ${JSON.stringify(build)};\n\n${readFileSync(template, "utf8")}`;
      writeFileSync(join(out, SW), source);
      config.logger.info(`sw.js: ${files.length} files in the offline cache (version ${build.version})`);
    },
  };
}
