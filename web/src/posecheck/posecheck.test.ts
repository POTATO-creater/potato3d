// Pose check: distance transform, XOR score and the 6-DOF alignment on a
// ray-traced scene.  pnpm test:web
import assert from "node:assert/strict";
import { test } from "node:test";
import { expSO3, mul3, mulVec3, transpose3, type Mat3 } from "../sfm/linalg.ts";
import { alignCost, alignStep, applyStep, choosePose, DEFAULT_ALIGN as defaults, prepareTarget, type Intrinsics } from "./align.ts";
import { distanceTo, fillHoles, openDisk, signedDistance } from "./image.ts";
import { frameValue, orbitScore, segments, xorScore } from "./score.ts";
import { readViews } from "../mesh/colmap.ts";
import { rewriteImages } from "../sfm/colmap.ts";

function rand(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

test("distance transform: matches brute force", () => {
  const w = 37, h = 23, r = rand(3);
  const m = new Uint8Array(w * h);
  for (let i = 0; i < m.length; i++) m[i] = r() < 0.04 ? 1 : 0;
  const d = distanceTo(m, w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let best = Infinity;
      for (let j = 0; j < m.length; j++) if (m[j]) best = Math.min(best, Math.hypot((j % w) - x, Math.floor(j / w) - y));
      assert.ok(Math.abs(d[y * w + x] - best) < 1e-4, `(${x}, ${y}): ${d[y * w + x]} vs ${best}`);
    }
});

test("signed distance: ±1 next to the edge, grows away from it", () => {
  const w = 20, h = 20;
  const m = new Uint8Array(w * h);
  for (let y = 5; y < 15; y++) for (let x = 5; x < 15; x++) m[y * w + x] = 1;
  const s = signedDistance(m, w, h);
  const at = (x: number, y: number) => s[y * w + x];
  assert.deepEqual([at(4, 10), at(5, 10), at(9, 10), at(1, 10)], [1, -1, -5, 4]);
});

test("opening drops strands thinner than the disk", () => {
  const w = 30, h = 30;
  const m = new Uint8Array(w * h);
  for (let y = 2; y < 28; y++) m[y * w + 3] = 1; // 1 px strand
  for (let y = 10; y < 20; y++) for (let x = 10; x < 20; x++) m[y * w + x] = 1; // 10×10 block
  const o = openDisk(m, w, h, 1);
  let strand = 0, block = 0;
  for (let y = 0; y < h; y++) {
    strand += o[y * w + 3];
    for (let x = 10; x < 20; x++) block += o[y * w + x];
  }
  assert.equal(strand, 0);
  assert.equal(block, 100 - 4); // the corners go
});

test("hole filling: closes holes, keeps gaps open to the border", () => {
  const w = 12, h = 10;
  const m = new Uint8Array(w * h);
  for (let y = 1; y < 9; y++) for (let x = 1; x < 11; x++) m[y * w + x] = 1;
  m[4 * w + 4] = m[4 * w + 5] = m[5 * w + 4] = 0; // a hole
  for (let y = 1; y < 6; y++) m[y * w + 8] = 0; // a notch reaching the top border via row 0
  m[0 * w + 8] = 0;
  const f = fillHoles(m, w, h);
  assert.deepEqual([f[4 * w + 4], f[4 * w + 5], f[5 * w + 4]], [1, 1, 1]);
  assert.equal(f[3 * w + 8], 0);
  assert.equal(f[0], 0);
  assert.equal(f.reduce((a, b) => a + b, 0), m.reduce((a, b) => a + b, 0) + 3);
});

test("XOR score: a shifted silhouette sticks out; thin slivers don't count", () => {
  const w = 64, h = 64;
  const square = (x0: number, y0: number, s: number) => {
    const a = new Float32Array(w * h);
    for (let y = y0; y < y0 + s; y++) for (let x = x0; x < x0 + s; x++) a[y * w + x] = 1;
    return a;
  };
  const mask = Uint8Array.from(square(16, 16, 32), (v) => (v > 0.5 ? 1 : 0));
  // 1 px off: only slivers, removed by the opening
  const s1 = xorScore(square(17, 16, 32), mask, w, h, 1);
  assert.deepEqual([s1.out, s1.in, s1.subject], [0, 0, 1024]);
  // 4 px off: a 4×32 band out on one side and in on the other
  const s4 = xorScore(square(20, 16, 32), mask, w, h, 1);
  assert.ok(s4.out > 100 && s4.out <= 128, `out ${s4.out}`);
  assert.ok(s4.in > 100 && s4.in <= 128, `in ${s4.in}`);
  assert.ok(Math.abs(frameValue(s4) - s4.out / 1024) < 1e-12);
  assert.equal(frameValue({ out: 5, in: 0, subject: 0 }), 0);
  assert.equal(orbitScore([0.01, 0.03]), 0.02);
  assert.deepEqual(segments([1, 1, 3, 3, 5], 2), [
    { start: 0, end: 1, value: 1 },
    { start: 2, end: 3, value: 3 },
    { start: 4, end: 4, value: 5 },
  ]);
});

// --- alignment on a ray-traced scene ---------------------------------------

/** Two textured spheres (a body and a smaller "head" off to one side). */
const SPHERES = [
  { c: [0, 0.25, 0], r: 0.8 },
  { c: [0.35, -0.75, 0.1], r: 0.38 },
];
function shade(p: number[]) {
  const v =
    Math.sin(9 * p[0] + 2) * Math.sin(7 * p[1] - 1) +
    0.6 * Math.sin(17 * p[2] + 5 * p[0]) +
    0.4 * Math.sin(23 * p[1] + 11 * p[2]);
  return 0.5 + 0.22 * v;
}

/** World → camera pose looking at the origin from `eye`. */
function lookAt(eye: number[]): { R: Mat3; t: [number, number, number] } {
  const f = eye.map((v) => -v);
  const n = Math.hypot(...f);
  const z = f.map((v) => v / n);
  const up = [0, -1, 0];
  let x = [z[1] * up[2] - z[2] * up[1], z[2] * up[0] - z[0] * up[2], z[0] * up[1] - z[1] * up[0]];
  const nx = Math.hypot(...x);
  x = x.map((v) => v / nx);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const R = Float64Array.from([...x, ...y, ...z]);
  const t = mulVec3(R, eye).map((v) => -v) as [number, number, number];
  return { R, t };
}

/** Colour (premultiplied, alpha from 2×2 supersampling) and depth, like Brush's renders. */
function raytrace(R: Mat3, t: ArrayLike<number>, K: Intrinsics, w: number, h: number) {
  const Rt = transpose3(R);
  const c = mulVec3(Rt, t).map((v) => -v);
  const rgba = new Float32Array(4 * w * h), depth = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let a = 0, col = 0, zsum = 0;
      for (const [sx, sy] of [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]) {
        const dc = [(x + sx - K.cx) / K.fx, (y + sy - K.cy) / K.fy, 1];
        const d = mulVec3(Rt, dc);
        let best = Infinity;
        for (const s of SPHERES) {
          const o = [c[0] - s.c[0], c[1] - s.c[1], c[2] - s.c[2]];
          const A = d[0] ** 2 + d[1] ** 2 + d[2] ** 2, B = 2 * (o[0] * d[0] + o[1] * d[1] + o[2] * d[2]);
          const C = o[0] ** 2 + o[1] ** 2 + o[2] ** 2 - s.r * s.r;
          const disc = B * B - 4 * A * C;
          if (disc < 0) continue;
          const l = (-B - Math.sqrt(disc)) / (2 * A);
          if (l > 0 && l < best) best = l;
        }
        if (best === Infinity) continue;
        a += 0.25;
        col += 0.25 * shade([c[0] + best * d[0], c[1] + best * d[1], c[2] + best * d[2]]);
        zsum += 0.25 * best; // dc has z = 1: the ray parameter is the camera depth
      }
      const i = y * w + x;
      rgba.set([col, col, col, a], 4 * i);
      depth[i] = a > 0 ? zsum / a : 0;
    }
  return { rgba, depth };
}

function poseError(a: { R: Mat3; t: ArrayLike<number> }, b: { R: Mat3; t: ArrayLike<number> }) {
  const D = mul3(a.R, transpose3(b.R));
  const deg = (Math.acos(Math.min(1, Math.max(-1, (D[0] + D[4] + D[8] - 1) / 2))) * 180) / Math.PI;
  const ca = mulVec3(transpose3(a.R), a.t), cb = mulVec3(transpose3(b.R), b.t);
  return { deg, move: Math.hypot(ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]) };
}

function recover(o: Partial<typeof defaults>, start?: "truth") {
  const w = 160, h = 200;
  const K: Intrinsics = { fx: 260, fy: 260, cx: w / 2, cy: h / 2 };
  const truth = lookAt([0.4, -0.2, -4.2]);
  const frame = raytrace(truth.R, truth.t, K, w, h);
  const rgba8 = Uint8ClampedArray.from(frame.rgba, (v, i) => (i % 4 === 3 ? v * 255 : (v / Math.max(frame.rgba[i - (i % 4) + 3], 1e-6)) * 255));
  const target = prepareTarget(rgba8, w, h);
  const xor = (rgba: Float32Array) => {
    const s = xorScore(Float32Array.from({ length: w * h }, (_, i) => rgba[4 * i + 3]), target.mask, w, h, 1);
    return s.out + s.in;
  };
  // ~1° and ~1% of the distance off, like the poses that make doubled edges
  const dR = expSO3([0.006, -0.012, 0.008]);
  let pose =
    start === "truth"
      ? truth
      : { R: mul3(dR, truth.R), t: [truth.t[0] + 0.03, truth.t[1] - 0.02, truth.t[2] + 0.04] as [number, number, number] };
  const before = { ...poseError(pose, truth), xor: xor(raytrace(pose.R, pose.t, K, w, h).rgba) };
  for (let i = 0; i < 3; i++) {
    const r = raytrace(pose.R, pose.t, K, w, h);
    const s = alignStep({ w, h, ...r }, target, K, { ...defaults, ...o });
    pose = applyStep(pose.R, pose.t, s.dx);
  }
  return { before, after: { ...poseError(pose, truth), xor: xor(raytrace(pose.R, pose.t, K, w, h).rgba) } };
}

test("alignment: undamped, recovers a ~1° pose error from contour + texture", () => {
  const { before, after } = recover({ damping: 0 });
  assert.ok(before.deg > 0.8, `before ${before.deg}°`);
  assert.ok(after.deg < before.deg / 5, `after ${after.deg.toFixed(3)}° (before ${before.deg.toFixed(3)}°)`);
  assert.ok(after.move < before.move / 5, `centre moved ${after.move.toFixed(4)} (before ${before.move.toFixed(4)})`);
  assert.ok(before.xor > 200 && after.xor < before.xor / 10, `XOR ${before.xor} → ${after.xor} px`);
});

// Damped (the default), most of the error left is motion the images barely
// show (orbiting the spheres): the silhouettes still line up.
test("alignment: damped, aligns the silhouettes", () => {
  const { before, after } = recover({});
  assert.ok(after.xor < before.xor / 4, `XOR ${before.xor} → ${after.xor} px`);
});

test("acceptance: keeps the cheapest pose only if it clearly beats the start", () => {
  assert.equal(choosePose([1, 0.9, 0.95], 0.02), 1);
  assert.equal(choosePose([1, 0.99, 0.985], 0.02), 0);
  assert.equal(choosePose([1, 1.2, 1.1], 0.02), 0);
  assert.equal(choosePose([1], 0.02), 0);
});

test("alignment cost: lower at the true pose than 1° off", () => {
  const w = 160, h = 200;
  const K: Intrinsics = { fx: 260, fy: 260, cx: w / 2, cy: h / 2 };
  const truth = lookAt([0.4, -0.2, -4.2]);
  const frame = raytrace(truth.R, truth.t, K, w, h);
  const rgba8 = Uint8ClampedArray.from(frame.rgba, (v, i) => (i % 4 === 3 ? v * 255 : (v / Math.max(frame.rgba[i - (i % 4) + 3], 1e-6)) * 255));
  const target = prepareTarget(rgba8, w, h);
  const off = { R: mul3(expSO3([0.006, -0.012, 0.008]), truth.R), t: truth.t };
  const at = (p: typeof truth) => alignCost({ w, h, ...raytrace(p.R, p.t, K, w, h) }, target, K);
  assert.ok(at(truth) < at(off) / 2, `${at(truth)} vs ${at(off)}`);
});

test("alignment: a right pose stays put", () => {
  const { after } = recover({}, "truth");
  // (0.2% of the distance: edge pixels at exactly half coverage)
  assert.ok(after.deg < 0.05 && after.move < 0.01, `moved ${after.deg.toFixed(4)}° / ${after.move.toFixed(5)}`);
});

// Spheres' outlines don't pin all six degrees of freedom: on its own the
// contour aligns the silhouettes but the pose may slide.
test("alignment: contour alone aligns the silhouettes", () => {
  const { before, after } = recover({ interiorWeight: 0, damping: 0 });
  assert.ok(after.xor < before.xor / 5, `XOR ${before.xor} → ${after.xor} px`);
});

test("images.bin rewrite: drops, renames and moves images, keeps their points", async () => {
  const cams = new DataView(new ArrayBuffer(8 + 24 + 32));
  cams.setBigUint64(0, 1n, true);
  cams.setUint32(8, 1, true);
  cams.setInt32(12, 1, true); // PINHOLE
  cams.setBigUint64(16, 64n, true);
  cams.setBigUint64(24, 48n, true);
  [50, 50, 32, 24].forEach((v, k) => cams.setFloat64(32 + 8 * k, v, true));
  const parts: number[] = [];
  const u8 = (b: ArrayBuffer) => parts.push(...new Uint8Array(b));
  const head = new DataView(new ArrayBuffer(8));
  head.setBigUint64(0, 2n, true);
  u8(head.buffer);
  for (const [id, name, pts] of [[1, "a.jpg", 2], [2, "b.jpg", 1]] as const) {
    const r = new DataView(new ArrayBuffer(64));
    r.setUint32(0, id, true);
    [1, 0, 0, 0, id, 0, 0].forEach((v, k) => r.setFloat64(4 + 8 * k, v, true));
    r.setUint32(60, 1, true);
    u8(r.buffer);
    parts.push(...new TextEncoder().encode(name), 0);
    const p = new DataView(new ArrayBuffer(8 + 24 * pts));
    p.setBigUint64(0, BigInt(pts), true);
    for (let i = 0; i < pts; i++) p.setFloat64(8 + 24 * i, 10 + i, true);
    u8(p.buffer);
  }
  const bin = new Uint8Array(parts).buffer;
  const out = await rewriteImages(bin, (r) => (r.name === "a.jpg" ? null : { name: "b.png", q: [0, 1, 0, 0], t: [1, 2, 3] })).arrayBuffer();
  const v = readViews(cams.buffer, out);
  assert.deepEqual(v.map((x) => [x.name, ...x.q, ...x.t]), [["b.png", 0, 1, 0, 0, 1, 2, 3]]);
  assert.equal(out.byteLength, 8 + 64 + "b.png".length + 1 + 8 + 24);
  assert.equal(new DataView(out).getFloat64(8 + 64 + 6 + 8, true), 10);
  const same = await rewriteImages(bin, (r) => r).arrayBuffer();
  assert.deepEqual(new Uint8Array(same), new Uint8Array(bin));
});
