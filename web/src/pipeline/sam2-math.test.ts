// Pure parts of SAM 2.1; reference values come from Transformers 4.57
// (facebook/sam2.1-hiera-tiny).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  fixVisionShape,
  logitsToMask,
  parseParams,
  pointerPosEnc,
  promptFrame,
  promptPixels,
  roundBf16,
  toNchw,
  toSeq,
} from "./sam2-math.ts";

const close = (a: ArrayLike<number>, b: number[], tol: number) =>
  b.forEach((v, i) => assert.ok(Math.abs(a[i] - v) < tol, `[${i}] ${a[i]} ≠ ${v}`));

test("parameters and temporal encoding of object pointers, as in Transformers", () => {
  const p = parseParams(readFileSync(new URL("./sam2-params.bin", import.meta.url)).buffer as ArrayBuffer);
  // projection(get_1d_sine_pe(t, 256))[:4] for t = 0, 1/15 and 1
  close(pointerPosEnc(0, p), [-0.57184, 0.46389, 1.8605, -1.30934], 1e-4);
  close(pointerPosEnc(1 / 15, p), [-0.6333, 0.43865, 1.86967, -1.27273], 1e-4);
  close(pointerPosEnc(1, p), [-1.40792, 0.01282, 1.85801, -0.65518], 1e-4);
});

test("bf16 rounds to nearest even", () => {
  const a = roundBf16(new Float32Array([1, 1 + 2 ** -8, 1 + 3 * 2 ** -8, -2.5, 1 + 2 ** -9]));
  assert.deepEqual([...a], [1, 1, 1 + 2 ** -6, -2.5, 1]);
});

test("NCHW ↔ sequence", () => {
  const nchw = Float32Array.from({ length: 4096 * 2 }, (_, i) => i);
  const seq = toSeq(nchw, 2);
  assert.deepEqual([seq[0], seq[1], seq[2], seq[3]], [0, 4096, 1, 4097]);
  assert.deepEqual(toNchw(seq, 2), nchw);
});

test("mask: sigmoid of logits upscaled to the frame", () => {
  const logits = new Float32Array(256 * 256).fill(-20);
  for (let y = 0; y < 256; y++) for (let x = 128; x < 256; x++) logits[y * 256 + x] = 20;
  const m = logitsToMask(logits, 64, 32);
  assert.equal(m[0], 0);
  assert.equal(m[63], 255);
  assert.equal(m[16 * 64 + 31], 0);
  assert.equal(m[16 * 64 + 32], 255);
});

test("prompt: marked frame (timestamp or image) and pixel coordinates", () => {
  const names = ["a.jpg", "b.jpg", "c.jpg"];
  const times = new Map([["a.jpg", { time: 0 }], ["b.jpg", { time: 1 }], ["c.jpg", { time: 2 }]]);
  const pr = { time: 1.4, box: [0.25, 0.5, 0.75, 1] as [number, number, number, number], points: [{ x: 0.5, y: 0.5, label: 0 as const }] };
  assert.equal(promptFrame(names, pr, times), 1);
  assert.equal(promptFrame(names, { ...pr, time: null }, times), 0);
  assert.deepEqual(promptPixels(pr, 200, 100), { box: [50, 50, 150, 100], points: [{ x: 100, y: 50, label: 0 }] });
  // images: by source image name; -1 if it did not become a frame
  const origin = new Map([["a.jpg", { source: "x.png" }], ["b.jpg", { source: "y.png" }]]);
  assert.equal(promptFrame(names, { ...pr, time: null, image: "y.png" }, origin), 1);
  assert.equal(promptFrame(names, { ...pr, time: null, image: "z.png" }, origin), -1);
  // no marking: the center
  assert.deepEqual(promptPixels(null, 200, 100), { points: [{ x: 100, y: 50, label: 1 }] });
});

test("encoder shape fix: only on the expected layout", () => {
  const buf = new ArrayBuffer(10200);
  assert.throws(() => fixVisionShape(buf));
  new Uint8Array(buf).set([0x0a, 0x0a, 0x08, 0x08, 0x07, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x05], 10137);
  fixVisionShape(buf);
  assert.equal(new Uint8Array(buf)[10147], 0x04);
  fixVisionShape(buf); // idempotent
});
