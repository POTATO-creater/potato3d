// O .wasm do ONNX Runtime (~27 MB) passa do limite de 25 MiB por arquivo do
// Cloudflare Workers. No build ele sai gzipado (~7 MB) e o original é
// descartado; o app descompacta no navegador e entrega o binário ao ORT
// (`env.wasm.wasmBinary`, ver web/src/pipeline/segment.ts). Continua tudo no
// mesmo domínio, então o cache offline do service worker segue valendo.
//
// `import url from "virtual:ort-wasm-gz"` dá a URL do .gz no build e `null`
// no dev, onde o ORT carrega o próprio .wasm como sempre.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { gzipSync } from "node:zlib";
import type { Plugin } from "vite";

const ID = "virtual:ort-wasm-gz";
// O que o bundle `onnxruntime-web/webgpu` carrega (JSEP saiu na 1.2x).
const WASM = "ort-wasm-simd-threaded.asyncify.wasm";

export function ortWasmGzip(): Plugin {
  let build = false;
  return {
    name: "ort-wasm-gzip",
    configResolved(c) {
      build = c.command === "build";
    },
    resolveId: (id) => (id === ID ? `\0${ID}` : null),
    load(id) {
      if (id !== `\0${ID}`) return null;
      if (!build) return "export default null;";
      const dist = dirname(createRequire(import.meta.url).resolve("onnxruntime-web"));
      const ref = this.emitFile({ type: "asset", name: `${WASM}.gz`, source: gzipSync(readFileSync(join(dist, WASM)), { level: 9 }) });
      return `export default import.meta.ROLLUP_FILE_URL_${ref};`;
    },
    generateBundle(_, bundle) {
      for (const file of Object.keys(bundle)) if (/ort-wasm[^/]*\.wasm$/.test(file)) delete bundle[file];
    },
  };
}
