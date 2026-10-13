// The ONNX Runtime .wasm (~27 MB) exceeds Cloudflare Workers' 25 MiB per-file
// limit. The build emits it gzipped (~7 MB) and drops the original; the app
// decompresses it in the browser and hands the binary to ORT
// (`env.wasm.wasmBinary`, see web/src/pipeline/segment.ts). Everything stays on
// the same origin, so the service worker's offline cache still covers it.
//
// `import url from "virtual:ort-wasm-gz"` yields the .gz URL in the build and
// `null` in dev, where ORT loads its own .wasm as usual.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { gzipSync } from "node:zlib";
import type { Plugin } from "vite";

const ID = "virtual:ort-wasm-gz";
// What the `onnxruntime-web/webgpu` bundle loads (JSEP was removed in 1.2x).
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
