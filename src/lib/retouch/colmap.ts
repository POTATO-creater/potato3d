// Appends cameras and images to a binary COLMAP model (cameras.bin and
// images.bin), copying the existing entries byte for byte. Used for the retouch
// dataset: restored frames get their own camera (different size) and no 2D
// points. Port of write_cameras/image_entry from
// scripts/tests/restore-test/restore_test.py.

export interface NewCamera {
  width: number;
  height: number;
  /** PINHOLE: fx, fy, cx, cy. */
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

export interface NewImage {
  name: string;
  qvec: [number, number, number, number];
  tvec: [number, number, number];
  /** Index into the `cameras` passed to appendCameras. */
  camera: number;
}

const PINHOLE = 1;
/** Parameter count per COLMAP camera model. */
const PARAMS: Record<number, number> = { 0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12 };

/** cameras.bin with the new cameras appended; returns the ids they were given. */
export function appendCameras(bin: ArrayBuffer, cams: NewCamera[]): { bin: ArrayBuffer; ids: number[] } {
  const v = new DataView(bin);
  const n = Number(v.getBigUint64(0, true));
  let o = 8, maxId = 0;
  for (let i = 0; i < n; i++) {
    maxId = Math.max(maxId, v.getInt32(o, true));
    const model = v.getInt32(o + 4, true);
    const params = PARAMS[model];
    if (params == null) throw new Error(`unknown COLMAP camera model: ${model}`);
    o += 24 + 8 * params;
  }
  const out = new ArrayBuffer(o + cams.length * (24 + 8 * 4));
  new Uint8Array(out).set(new Uint8Array(bin, 0, o));
  const w = new DataView(out);
  w.setBigUint64(0, BigInt(n + cams.length), true);
  const ids = cams.map((c, k) => {
    const id = maxId + 1 + k;
    w.setInt32(o, id, true);
    w.setInt32(o + 4, PINHOLE, true);
    w.setBigUint64(o + 8, BigInt(c.width), true);
    w.setBigUint64(o + 16, BigInt(c.height), true);
    [c.fx, c.fy, c.cx, c.cy].forEach((p, j) => w.setFloat64(o + 24 + 8 * j, p, true));
    o += 24 + 32;
    return id;
  });
  return { bin: out, ids };
}

/** images.bin with the new images (no 2D points) appended. */
export function appendImages(bin: ArrayBuffer, images: NewImage[], cameraIds: number[]): ArrayBuffer {
  const v = new DataView(bin);
  const bytes = new Uint8Array(bin);
  const n = Number(v.getBigUint64(0, true));
  let o = 8, maxId = 0;
  for (let i = 0; i < n; i++) {
    maxId = Math.max(maxId, v.getInt32(o, true));
    o += 4 + 7 * 8 + 4;
    o = bytes.indexOf(0, o) + 1;
    const pts = Number(v.getBigUint64(o, true));
    o += 8 + pts * 24;
  }
  const enc = new TextEncoder();
  const names = images.map((im) => enc.encode(im.name));
  const size = names.reduce((s, nm) => s + 4 + 56 + 4 + nm.length + 1 + 8, 0);
  const out = new ArrayBuffer(o + size);
  new Uint8Array(out).set(bytes.subarray(0, o));
  const w = new DataView(out);
  w.setBigUint64(0, BigInt(n + images.length), true);
  images.forEach((im, k) => {
    w.setInt32(o, maxId + 1 + k, true);
    [...im.qvec, ...im.tvec].forEach((x, j) => w.setFloat64(o + 4 + 8 * j, x, true));
    w.setInt32(o + 60, cameraIds[im.camera], true);
    o += 64;
    new Uint8Array(out, o, names[k].length).set(names[k]);
    o += names[k].length;
    w.setUint8(o, 0);
    w.setBigUint64(o + 1, 0n, true);
    o += 9;
  });
  return out;
}
