import { readFileSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { ortWasmGzip } from "./ort-wasm-plugin";
import { serviceWorker } from "./sw-plugin";

// Web version: the src/ UI with the browser host (web/src/host/); training
// runs on Brush compiled to WASM + WebGPU (`pnpm web:wasm` generates web/pkg).
export default defineConfig({
  root: import.meta.dirname,
  // Written to the project package's manifest.json (.gorbit).
  define: { __APP_VERSION__: JSON.stringify(JSON.parse(readFileSync(`${import.meta.dirname}/../package.json`, "utf8")).version) },
  // Build-only: gzips the ONNX Runtime .wasm (Cloudflare's 25 MiB limit) and emits
  // the offline-cache service worker, which would get in the way of HMR in dev.
  plugins: [react(), ortWasmGzip(), serviceWorker(`${import.meta.dirname}/src/sw.template.js`)],
  // ONNX Runtime locates its own .wasm via import.meta.url, so it can't be pre-bundled.
  // The mesh worker's dependencies are only met when an export starts; found
  // then, their pre-bundling reloads the page and the worker fails to load.
  // colmap-wasm also resolves its worker and .wasm via import.meta.url.
  optimizeDeps: { exclude: ["onnxruntime-web", "colmap-wasm"], include: ["meshoptimizer", "xatlas-wasm", "three/examples/jsm/loaders/GLTFLoader.js"] },
  build: {
    outDir: "../dist-web",
    emptyOutDir: true,
    target: "es2022",
    // Two pages: the app and the shared-link viewer, which the Worker serves
    // at /s/<id> (worker/index.ts).
    rolldownOptions: {
      input: {
        main: `${import.meta.dirname}/index.html`,
        share: `${import.meta.dirname}/share.html`,
      },
    },
  },
  server: {
    port: 1440,
    strictPort: true,
    // WEB_FS_ANY=1 lets the dev server serve any folder via /@fs, for testing
    // with `?dataset=/@fs/<dataset path>`.
    fs: { allow: [".."], strict: !process.env.WEB_FS_ANY },
  },
});
