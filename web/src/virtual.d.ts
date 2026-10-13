/** URL of the gzipped ONNX Runtime .wasm in the build; `null` in dev (web/ort-wasm-plugin.ts). */
declare module "virtual:ort-wasm-gz" {
  const url: string | null;
  export default url;
}

/** App version (package.json), defined in web/vite.config.ts. */
declare const __APP_VERSION__: string;
