// Same cases as the tests in src-tauri/src/export (ply.rs, sog.rs, decimate.rs).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { decimate, ellipsoidArea, quatFromMatrix, quatToMatrix, sigmoid } from "./decimate.ts";
import { runExport } from "./run.ts";
import { kmeansTwoLevel, packQuat, quantize1d } from "./sog.ts";
import { plyInfo, readPly, sample, truncateSh, writePly, type Splats } from "./splats.ts";
import factory from "./webp/webp.mjs";

test("PLY: round trip and SH band truncation", () => {
  const s = sample(100, 2);
  const bytes = writePly(s);
  const back = readPly(bytes);
  assert.equal(back.count, 100);
  assert.equal(back.shDegree, 2);
  assert.deepEqual(back.pos, s.pos);
  assert.deepEqual(back.rest, s.rest);
  assert.deepEqual(back.rot, s.rot);
  assert.deepEqual(plyInfo(bytes), { count: 100, shDegree: 2 });

  const t = truncateSh(back, 1);
  assert.equal(t.rest.length, 100 * 9);
  // Coefficient 0 of the G channel lands at index 3 (it was at 8).
  assert.equal(t.rest[3], back.rest[8]);
  assert.equal(t.rest[9 + 6], back.rest[24 + 16]);
});

test("SOG: quantizer is monotonic and close", () => {
  const col = Float32Array.from({ length: 10_000 }, (_, i) => Math.sin(i * 0.37) * 3);
  const q = quantize1d([col], 256);
  for (let i = 1; i < q.centroids.length; i++) assert.ok(q.centroids[i - 1] <= q.centroids[i]);
  let err = 0;
  col.forEach((v, i) => (err = Math.max(err, Math.abs(v - q.centroids[q.labels[0][i]]))));
  assert.ok(err < 0.1, `max error ${err}`);
});

test("SOG: quaternion in 4 bytes", () => {
  const q = [0.3, -0.5, 0.6, 0.2];
  const p = new Uint8Array(4);
  packQuat(q, 0, p, 0);
  const l = Math.hypot(...q);
  assert.equal(p[3] - 252, 2);
  // Decode like the reader: (b/255 − 0.5)·2/√2.
  const dec = (b: number) => ((b / 255 - 0.5) * 2) / Math.SQRT2;
  [0, 1, 3].forEach((i, k) => assert.ok(Math.abs(dec(p[k]) - (q[i] / l) * Math.sign(q[2])) < 0.01));
});

test("SOG: k-means with labels inside the palette", () => {
  const s = sample(5_000, 1);
  const { centroids, labels } = kmeansTwoLevel(s.rest, s.count, 9, 512);
  const count = centroids.length / 9;
  assert.ok(count <= 512 && count > 256, String(count));
  assert.ok(labels.every((l) => l < count));
});

test("decimation: reaches the target and conserves ink", () => {
  const s = sample(40_000, 1);
  const out = decimate(s, 0.5, () => {})!;
  assert.ok(Math.abs(out.count - 20_000) < 400, `count ${out.count}`);
  assert.equal(out.rest.length, out.count * 9);
  for (const a of [out.pos, out.scale, out.opacity]) assert.ok(a.every(Number.isFinite));
  // Total ink (opacity × area) should not shrink much from merging.
  const ink = (x: Splats) => {
    let t = 0;
    for (let i = 0; i < x.count; i++) {
      const [a, b, c] = [0, 1, 2].map((k) => Math.exp(x.scale[i * 3 + k]));
      t += sigmoid(x.opacity[i]) * ellipsoidArea(a, b, c);
    }
    return t;
  };
  assert.ok(ink(out) > ink(s) * 0.8);
  assert.equal(decimate(sample(500, 0), 0.5, () => {}), null);
});

test("decimation: rotation round trip", () => {
  const n = Math.hypot(0.8, 0.2, -0.4, 0.4);
  const q = [0.8, 0.2, -0.4, 0.4].map((v) => v / n);
  const back = quatFromMatrix(quatToMatrix(q[0], q[1], q[2], q[3]));
  assert.ok(Math.abs(Math.abs(q.reduce((s, v, i) => s + v * back[i], 0)) - 1) < 1e-9);
});

test("SOG: package with meta.json, SH bands and WebP that libwebp can read", async () => {
  const wasm = new URL("./webp/webp.wasm", import.meta.url);
  const m = (await factory({ wasmBinary: readFileSync(wasm) })) as {
    HEAPU8: Uint8Array;
    HEAPU32: Uint32Array;
    _malloc(n: number): number;
    _free(p: number): void;
    _webp_free(p: number): void;
    _webp_encode_lossless_rgba(i: number, w: number, h: number, s: number, o: number, os: number): number;
  };
  const webp = (rgba: Uint8Array, w: number, h: number) => {
    const i = m._malloc(rgba.length), o = m._malloc(4), os = m._malloc(4);
    m.HEAPU8.set(rgba, i);
    assert.ok(m._webp_encode_lossless_rgba(i, w, h, w * 4, o, os));
    const out = m.HEAPU8.slice(m.HEAPU32[o >> 2], m.HEAPU32[o >> 2] + m.HEAPU32[os >> 2]);
    m._webp_free(m.HEAPU32[o >> 2]);
    [i, o, os].forEach((p) => m._free(p));
    return out;
  };
  const r = runExport(writePly(sample(3_000, 1)), { format: "sog", decimate: 0, minOpacity: 0, shDegree: null }, webp, () => {});
  assert.equal(r.name, "splat.sog");
  assert.equal(r.stats.outputCount, 3000);

  // Read the ZIP through the central directory.
  const z = r.bytes, v = new DataView(z.buffer, z.byteOffset);
  const eocd = z.length - 22;
  assert.equal(v.getUint32(eocd, true), 0x06054b50);
  const files = new Map<string, Uint8Array>();
  for (let k = 0, c = v.getUint32(eocd + 16, true); k < v.getUint16(eocd + 10, true); k++) {
    const nameLen = v.getUint16(c + 28, true), size = v.getUint32(c + 24, true), local = v.getUint32(c + 42, true);
    const name = new TextDecoder().decode(z.subarray(c + 46, c + 46 + nameLen));
    const start = local + 30 + v.getUint16(local + 26, true);
    files.set(name, z.subarray(start, start + size));
    c += 46 + nameLen;
  }
  const meta = JSON.parse(new TextDecoder().decode(files.get("meta.json")));
  assert.equal(meta.count, 3000);
  assert.equal(meta.shN.bands, 1);
  assert.equal(meta.scales.codebook.length, 256);
  for (const f of ["means_l", "means_u", "quats", "scales", "sh0", "shN_centroids", "shN_labels"]) {
    const webpFile = files.get(`${f}.webp`)!;
    assert.equal(new TextDecoder().decode(webpFile.subarray(8, 12)), "WEBP", f);
  }
});
