import assert from "node:assert/strict";
import { test } from "node:test";
import { appendCameras, appendImages } from "./colmap.ts";

/** cameras.bin with one SIMPLE_PINHOLE (id 1, 960×1920, f 1600). */
function camerasBin() {
  const b = new ArrayBuffer(8 + 24 + 24);
  const v = new DataView(b);
  v.setBigUint64(0, 1n, true);
  v.setInt32(8, 1, true);
  v.setInt32(12, 0, true);
  v.setBigUint64(16, 960n, true);
  v.setBigUint64(24, 1920n, true);
  [1600, 480, 960].forEach((p, i) => v.setFloat64(32 + 8 * i, p, true));
  return b;
}

/** images.bin with two images; the second has one 2D point. */
function imagesBin() {
  const enc = new TextEncoder();
  const parts: number[] = [];
  const push = (n: number, bytes: number, fn: (v: DataView) => void) => {
    const b = new ArrayBuffer(bytes);
    fn(new DataView(b));
    parts.push(...new Uint8Array(b));
    void n;
  };
  push(0, 8, (v) => v.setBigUint64(0, 2n, true));
  for (const [id, name, pts] of [[3, "00000.jpg", 0], [7, "00001.jpg", 1]] as const) {
    push(0, 64, (v) => {
      v.setInt32(0, id, true);
      [1, 0, 0, 0, 0.1, 0.2, 0.3].forEach((x, j) => v.setFloat64(4 + 8 * j, x, true));
      v.setInt32(60, 1, true);
    });
    parts.push(...enc.encode(name), 0);
    push(0, 8, (v) => v.setBigUint64(0, BigInt(pts), true));
    for (let p = 0; p < pts; p++) push(0, 24, (v) => (v.setFloat64(0, 1.5, true), v.setFloat64(8, 2.5, true), v.setBigInt64(16, 9n, true)));
  }
  return new Uint8Array(parts).buffer;
}

/** Full read-back for checking (same format as images.bin). */
function readImages(b: ArrayBuffer) {
  const v = new DataView(b), bytes = new Uint8Array(b), dec = new TextDecoder();
  const n = Number(v.getBigUint64(0, true));
  let o = 8;
  const out = [];
  for (let i = 0; i < n; i++) {
    const id = v.getInt32(o, true);
    const q = [0, 1, 2, 3].map((j) => v.getFloat64(o + 4 + 8 * j, true));
    const cam = v.getInt32(o + 60, true);
    o += 64;
    const end = bytes.indexOf(0, o);
    const name = dec.decode(bytes.subarray(o, end));
    o = end + 1;
    const pts = Number(v.getBigUint64(o, true));
    o += 8 + pts * 24;
    out.push({ id, q, cam, name, pts });
  }
  assert.equal(o, b.byteLength);
  return out;
}

test("new cameras appended, with ids after the largest", () => {
  const { bin, ids } = appendCameras(camerasBin(), [{ width: 480, height: 960, fx: 800, fy: 800, cx: 240, cy: 480 }]);
  assert.deepEqual(ids, [2]);
  const v = new DataView(bin);
  assert.equal(Number(v.getBigUint64(0, true)), 2);
  assert.equal(v.getInt32(56, true), 2);
  assert.equal(v.getInt32(60, true), 1); // PINHOLE
  assert.equal(Number(v.getBigUint64(64, true)), 480);
  assert.deepEqual([0, 1, 2, 3].map((j) => v.getFloat64(80 + 8 * j, true)), [800, 800, 240, 480]);
  assert.equal(bin.byteLength, 56 + 24 + 32);
});

test("new images have no 2D points; existing ones stay intact", () => {
  const src = imagesBin();
  const out = appendImages(
    src,
    [
      { name: "retouch_a_00000.png", qvec: [0, 1, 0, 0], tvec: [1, 2, 3], camera: 0 },
      { name: "retouch_a_00001.png", qvec: [0, 0, 1, 0], tvec: [4, 5, 6], camera: 0 },
    ],
    [2],
  );
  const imgs = readImages(out);
  assert.deepEqual(imgs.map((i) => [i.id, i.name, i.cam, i.pts]), [
    [3, "00000.jpg", 1, 0],
    [7, "00001.jpg", 1, 1],
    [8, "retouch_a_00000.png", 2, 0],
    [9, "retouch_a_00001.png", 2, 0],
  ]);
  assert.deepEqual(imgs[2].q, [0, 1, 0, 0]);
  assert.deepEqual(new Uint8Array(out, 0, src.byteLength).subarray(8), new Uint8Array(src).subarray(8));
});
