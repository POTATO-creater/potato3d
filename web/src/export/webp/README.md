# libwebp (WASM)

`webp.mjs` + `webp.wasm` são artefatos copiados sem alteração de
[playcanvas/splat-transform](https://github.com/playcanvas/splat-transform)
(`lib/webp.{mjs,wasm}`, licença MIT), via o WebGaussian: um wrapper mínimo do
libwebp com `_webp_encode_lossless_rgba`. O SOG precisa de WebP sem perdas com
alfa direto, que o canvas não faz (pré-multiplica o alfa).
