// Minimal dependency-free ZIP for the project package (.gorbit): writes as a
// stream (STORE, or DEFLATE via the browser's CompressionStream, with a data
// descriptor) and reads via the central directory, with ZIP64 (packages from
// the former desktop app's `zip` crate switch to it above 4 GB). Stored
// (uncompressed) entries become slices of the file when read, with no copy.

/** Sink for the ZIP bytes, in order. */
export type ZipSink = (part: Blob | Uint8Array<ArrayBuffer>) => Promise<void> | void;

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

/** ZIP CRC-32; `crc` continues a previous computation. */
export function crc32(data: Uint8Array, crc = 0): number {
  let c = ~crc >>> 0;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return ~c >>> 0;
}

const LOCAL = 0x04034b50;
const CENTRAL = 0x02014b50;
const DESCRIPTOR = 0x08074b50;
const END = 0x06054b50;
const END64 = 0x06064b50;
const LOCATOR64 = 0x07064b50;
const MAX32 = 0xffffffff;
// Bit 3: sizes and CRC in the data descriptor; bit 11: UTF-8 name.
const FLAGS = 0x0808;
// 1980-01-01 00:00 in DOS format.
const DOS_DATE = (0 << 9) | (1 << 5) | 1;

interface Written {
  name: Uint8Array;
  method: number;
  crc: number;
  csize: number;
  size: number;
  offset: number;
}

function header(size: number, fill: (v: DataView) => void): Uint8Array<ArrayBuffer> {
  const b = new Uint8Array(size);
  fill(new DataView(b.buffer));
  return b;
}

export class ZipWriter {
  private offset = 0;
  private entries: Written[] = [];
  private sink: ZipSink;

  constructor(sink: ZipSink) {
    this.sink = sink;
  }

  private async out(part: Blob | Uint8Array<ArrayBuffer>) {
    await this.sink(part);
    this.offset += part instanceof Blob ? part.size : part.length;
  }

  /** `onBytes` receives how many bytes of the entry have been read so far. */
  async add(name: string, data: Blob, deflate: boolean, onBytes?: (n: number) => void) {
    const nameBytes = new TextEncoder().encode(name);
    const method = deflate ? 8 : 0;
    const offset = this.offset;
    if (offset > MAX32) throw new Error("the package exceeds 4 GB, the web version's limit");
    await this.out(
      header(30 + nameBytes.length, (v) => {
        v.setUint32(0, LOCAL, true);
        v.setUint16(4, 20, true);
        v.setUint16(6, FLAGS, true);
        v.setUint16(8, method, true);
        v.setUint16(12, DOS_DATE, true);
        v.setUint16(26, nameBytes.length, true);
        new Uint8Array(v.buffer).set(nameBytes, 30);
      }),
    );

    let crc = 0;
    let read = 0;
    const reader = data.stream().getReader();
    const pump = new ReadableStream<Uint8Array<ArrayBuffer>>({
      async pull(ctl) {
        const { done, value } = await reader.read();
        if (done) return ctl.close();
        crc = crc32(value, crc);
        read += value.length;
        onBytes?.(read);
        ctl.enqueue(value as Uint8Array<ArrayBuffer>);
      },
      cancel: (r) => reader.cancel(r),
    });
    let csize = 0;
    if (deflate) {
      const chunks: Uint8Array<ArrayBuffer>[] = [];
      for await (const c of pump.pipeThrough(new CompressionStream("deflate-raw")) as unknown as AsyncIterable<Uint8Array<ArrayBuffer>>) {
        chunks.push(c);
        csize += c.length;
      }
      // One Blob per entry: the browser may keep it out of memory.
      await this.out(new Blob(chunks));
    } else {
      // Only the CRC needs to read the bytes; the file itself goes in whole.
      for await (const _ of pump as unknown as AsyncIterable<Uint8Array>) void _;
      csize = data.size;
      await this.out(data);
    }
    if (read > MAX32 || csize > MAX32) throw new Error(`"${name}" exceeds 4 GB, the web version's limit`);

    await this.out(
      header(16, (v) => {
        v.setUint32(0, DESCRIPTOR, true);
        v.setUint32(4, crc, true);
        v.setUint32(8, csize, true);
        v.setUint32(12, read, true);
      }),
    );
    this.entries.push({ name: nameBytes, method, crc, csize, size: read, offset });
  }

  /** Writes the central directory; returns the total ZIP size. */
  async finish(): Promise<number> {
    const start = this.offset;
    for (const e of this.entries)
      await this.out(
        header(46 + e.name.length, (v) => {
          v.setUint32(0, CENTRAL, true);
          v.setUint16(4, 20, true);
          v.setUint16(6, 20, true);
          v.setUint16(8, FLAGS, true);
          v.setUint16(10, e.method, true);
          v.setUint16(14, DOS_DATE, true);
          v.setUint32(16, e.crc, true);
          v.setUint32(20, e.csize, true);
          v.setUint32(24, e.size, true);
          v.setUint16(28, e.name.length, true);
          v.setUint32(42, e.offset, true);
          new Uint8Array(v.buffer).set(e.name, 46);
        }),
      );
    const size = this.offset - start;
    if (this.offset > MAX32 || this.entries.length > 0xffff) throw new Error("the package exceeds the web version's limit (4 GB)");
    await this.out(
      header(22, (v) => {
        v.setUint32(0, END, true);
        v.setUint16(8, this.entries.length, true);
        v.setUint16(10, this.entries.length, true);
        v.setUint32(12, size, true);
        v.setUint32(16, start, true);
      }),
    );
    return this.offset;
  }
}

export interface ZipEntry {
  name: string;
  method: number;
  /** Uncompressed size. */
  size: number;
  csize: number;
  offset: number;
}

const view = async (b: Blob) => new DataView(await b.arrayBuffer());
const u64 = (v: DataView, at: number) => Number(v.getBigUint64(at, true));

/** ZIP entries from the central directory. */
export async function readZip(file: Blob): Promise<Map<string, ZipEntry>> {
  const notZip = () => new Error("the file is not a valid ZIP");
  const tailStart = Math.max(0, file.size - 22 - 0xffff);
  const tail = await view(file.slice(tailStart));
  let end = -1;
  for (let i = tail.byteLength - 22; i >= 0; i--)
    if (tail.getUint32(i, true) === END) {
      end = i;
      break;
    }
  if (end < 0) throw notZip();
  let count = tail.getUint16(end + 10, true);
  let cdSize = tail.getUint32(end + 12, true);
  let cdStart = tail.getUint32(end + 16, true);
  if ((count === 0xffff || cdSize === MAX32 || cdStart === MAX32) && end >= 20 && tail.getUint32(end - 20, true) === LOCATOR64) {
    const at = u64(tail, end - 12);
    const e64 = await view(file.slice(at, at + 56));
    if (e64.getUint32(0, true) !== END64) throw notZip();
    count = u64(e64, 32);
    cdSize = u64(e64, 40);
    cdStart = u64(e64, 48);
  }
  const cd = await view(file.slice(cdStart, cdStart + cdSize));
  const entries = new Map<string, ZipEntry>();
  const decoder = new TextDecoder();
  let o = 0;
  for (let i = 0; i < count; i++) {
    if (cd.getUint32(o, true) !== CENTRAL) throw notZip();
    const nameLen = cd.getUint16(o + 28, true);
    const extraLen = cd.getUint16(o + 30, true);
    const commentLen = cd.getUint16(o + 32, true);
    const name = decoder.decode(new Uint8Array(cd.buffer, cd.byteOffset + o + 46, nameLen));
    let size = cd.getUint32(o + 24, true);
    let csize = cd.getUint32(o + 20, true);
    let offset = cd.getUint32(o + 42, true);
    // ZIP64: only the overflowed fields appear in extra field 0x0001, in this order.
    for (let x = o + 46 + nameLen; x + 4 <= o + 46 + nameLen + extraLen; ) {
      const id = cd.getUint16(x, true);
      const len = cd.getUint16(x + 2, true);
      if (id === 1) {
        let f = x + 4;
        if (size === MAX32) (size = u64(cd, f)), (f += 8);
        if (csize === MAX32) (csize = u64(cd, f)), (f += 8);
        if (offset === MAX32) offset = u64(cd, f);
      }
      x += 4 + len;
    }
    entries.set(name, { name, method: cd.getUint16(o + 10, true), size, csize, offset });
    o += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

/** An entry's content: a slice of the file if it wasn't compressed. */
export async function entryBlob(file: Blob, e: ZipEntry): Promise<Blob> {
  const local = await view(file.slice(e.offset, e.offset + 30));
  if (local.getUint32(0, true) !== LOCAL) throw new Error(`corrupt package at ${e.name}`);
  const start = e.offset + 30 + local.getUint16(26, true) + local.getUint16(28, true);
  const raw = file.slice(start, start + e.csize);
  if (e.method === 0) return raw;
  if (e.method !== 8) throw new Error(`unsupported compression in ${e.name}`);
  return new Response(raw.stream().pipeThrough(new DecompressionStream("deflate-raw"))).blob();
}
