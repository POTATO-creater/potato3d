// ZIP sem compressão (STORE), como o SOG pede: o WebP já vem comprimido.

const CRC = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c;
});

function crc32(bytes: Uint8Array) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

export function zipStore(files: [name: string, data: Uint8Array][]): Uint8Array {
  const enc = new TextEncoder();
  const entries = files.map(([name, data]) => ({ name: enc.encode(name), data, crc: crc32(data), offset: 0 }));
  const size = entries.reduce((s, e) => s + 30 + e.name.length + e.data.length + 46 + e.name.length, 22);
  const out = new Uint8Array(size);
  const v = new DataView(out.buffer);
  let o = 0;
  // cabeçalho comum ao local e ao central: versão 2.0, sem flags, STORE, sem data
  const common = (at: number, e: (typeof entries)[number]) => {
    v.setUint16(at, 20, true);
    v.setUint16(at + 2, 0, true);
    v.setUint16(at + 4, 0, true);
    v.setUint32(at + 6, 0x00210000, true); // 1980-01-01 00:00
    v.setUint32(at + 10, e.crc, true);
    v.setUint32(at + 14, e.data.length, true);
    v.setUint32(at + 18, e.data.length, true);
    v.setUint16(at + 22, e.name.length, true);
    v.setUint16(at + 24, 0, true);
  };
  for (const e of entries) {
    e.offset = o;
    v.setUint32(o, 0x04034b50, true);
    common(o + 4, e);
    out.set(e.name, o + 30);
    out.set(e.data, o + 30 + e.name.length);
    o += 30 + e.name.length + e.data.length;
  }
  const central = o;
  for (const e of entries) {
    v.setUint32(o, 0x02014b50, true);
    v.setUint16(o + 4, 20, true); // feito por
    common(o + 6, e);
    // comentário, disco, atributos internos/externos ficam zerados
    v.setUint32(o + 42, e.offset, true);
    out.set(e.name, o + 46);
    o += 46 + e.name.length;
  }
  v.setUint32(o, 0x06054b50, true);
  v.setUint16(o + 8, entries.length, true);
  v.setUint16(o + 10, entries.length, true);
  v.setUint32(o + 12, o - central, true);
  v.setUint32(o + 16, central, true);
  return out;
}
