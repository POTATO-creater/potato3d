// Minimal PNG encoder (8-bit grayscale or RGBA with straight alpha). Canvas stores
// premultiplied alpha and would lose color on semi-transparent edges.

const CRC = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(bytes: Uint8Array, crc = 0xffffffff) {
  for (let i = 0; i < bytes.length; i++) crc = CRC[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  return crc;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const v = new DataView(out.buffer);
  v.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  v.setUint32(8 + data.length, (crc32(out.subarray(4, 8 + data.length)) ^ 0xffffffff) >>> 0);
  return out;
}

async function zlib(data: Uint8Array): Promise<Uint8Array> {
  // "deflate" in CompressionStream is the zlib format, which is what IDAT expects
  const stream = new Blob([data as BlobPart]).stream().pipeThrough(new CompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** `channels`: 1 (grayscale) or 4 (RGBA). "Sub" filter on every row. */
export async function encodePng(pixels: Uint8Array | Uint8ClampedArray, width: number, height: number, channels: 1 | 4) {
  const stride = width * channels;
  const raw = new Uint8Array((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    const o = y * (stride + 1);
    raw[o] = 1; // Sub
    const row = y * stride;
    for (let x = 0; x < stride; x++) raw[o + 1 + x] = (pixels[row + x] - (x >= channels ? pixels[row + x - channels] : 0)) & 0xff;
  }
  const ihdr = new Uint8Array(13);
  const v = new DataView(ihdr.buffer);
  v.setUint32(0, width);
  v.setUint32(4, height);
  ihdr[8] = 8;
  ihdr[9] = channels === 1 ? 0 : 6;
  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  return new Blob([sig, chunk("IHDR", ihdr), chunk("IDAT", await zlib(raw)), chunk("IEND", new Uint8Array(0))] as BlobPart[], {
    type: "image/png",
  });
}
