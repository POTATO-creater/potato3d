// Loads the vendored libwebp (see README.md) and exposes the lossless encoder.

import factory from "./webp.mjs";
import wasmUrl from "./webp.wasm?url";

interface WebpModule {
  HEAPU8: Uint8Array;
  HEAPU32: Uint32Array;
  _malloc(size: number): number;
  _free(ptr: number): void;
  _webp_free(ptr: number): void;
  _webp_encode_lossless_rgba(input: number, width: number, height: number, stride: number, outPtr: number, outSize: number): number;
}

let module: Promise<WebpModule> | null = null;

export async function webpEncoder() {
  module ??= (factory as (o: object) => Promise<WebpModule>)({ locateFile: () => wasmUrl });
  const m = await module;
  return (rgba: Uint8Array, width: number, height: number): Uint8Array => {
    const input = m._malloc(rgba.length), outPtr = m._malloc(4), outSize = m._malloc(4);
    try {
      m.HEAPU8.set(rgba, input);
      if (!m._webp_encode_lossless_rgba(input, width, height, width * 4, outPtr, outSize))
        throw new Error("WebP encoding failed");
      const ptr = m.HEAPU32[outPtr >> 2], size = m.HEAPU32[outSize >> 2];
      const bytes = m.HEAPU8.slice(ptr, ptr + size);
      m._webp_free(ptr);
      return bytes;
    } finally {
      m._free(input);
      m._free(outPtr);
      m._free(outSize);
    }
  };
}
