// Gera o sw.js do build: web/src/sw.template.js prefixado com a lista de
// arquivos da saída e uma versão (hash do conteúdo). Varre a pasta de saída
// depois de gravada, para pegar também os arquivos dos workers e de public/.

import { createHash } from "node:crypto";
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative, resolve, sep } from "node:path";
import type { Plugin, ResolvedConfig } from "vite";

const SW = "sw.js";
// Regras do Cloudflare (web/public/_headers): lidas no deploy, nunca servidas.
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
      // O código do próprio service worker também entra: mudar só ele já é versão nova.
      const hash = createHash("sha256").update(readFileSync(template));
      for (const f of files) hash.update(f).update(readFileSync(join(out, f)));
      const build = { version: hash.digest("hex").slice(0, 16), files };
      const source = `const SW_BUILD = ${JSON.stringify(build)};\n\n${readFileSync(template, "utf8")}`;
      writeFileSync(join(out, SW), source);
      config.logger.info(`sw.js: ${files.length} arquivos no cache offline (versão ${build.version})`);
    },
  };
}
