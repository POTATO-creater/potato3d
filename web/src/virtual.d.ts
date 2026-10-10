/** URL do .wasm gzipado do ONNX Runtime no build; `null` no dev (web/ort-wasm-plugin.ts). */
declare module "virtual:ort-wasm-gz" {
  const url: string | null;
  export default url;
}
