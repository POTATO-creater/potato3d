import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Protótipo 100% web: o treino roda no Brush compilado para WASM + WebGPU
// (`pnpm web:wasm` gera web/pkg). Reaproveita componentes de src/.
export default defineConfig({
  root: import.meta.dirname,
  plugins: [react()],
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
