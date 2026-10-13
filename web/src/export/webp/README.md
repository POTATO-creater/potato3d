# libwebp (WASM)

`webp.mjs` + `webp.wasm` are artifacts copied unmodified from
[playcanvas/splat-transform](https://github.com/playcanvas/splat-transform)
(`lib/webp.{mjs,wasm}`, MIT license), via WebGaussian: a minimal libwebp
wrapper with `_webp_encode_lossless_rgba`. SOG needs lossless WebP with
straight alpha, which canvas can't produce (it premultiplies alpha).
