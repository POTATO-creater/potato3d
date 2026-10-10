import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { ortWasmGzip } from "./ort-wasm-plugin";
import { serviceWorker } from "./sw-plugin";

// Versão web: a interface de src/ com o host do navegador (web/src/host/);
// o treino roda no Brush compilado para WASM + WebGPU (`pnpm web:wasm` gera
// web/pkg).
export default defineConfig({
  root: import.meta.dirname,
  // Só agem no build: o .wasm do ONNX Runtime sai gzipado (limite de 25 MiB
  // do Cloudflare) e o service worker do cache offline, que no dev atrapalharia o HMR.
  plugins: [react(), ortWasmGzip(), serviceWorker(`${import.meta.dirname}/src/sw.template.js`)],
  // O ONNX Runtime acha os próprios .wasm por import.meta.url: não pode ser pré-empacotado.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: {
    outDir: "../dist-web",
    emptyOutDir: true,
    target: "es2022",
  },
  server: {
    port: 1440,
    strictPort: true,
    // WEB_FS_ANY=1 deixa o dev server servir qualquer pasta via /@fs, para
    // testar com `?dataset=/@fs/<caminho do dataset>`.
    fs: { allow: [".."], strict: !process.env.WEB_FS_ANY },
  },
});
