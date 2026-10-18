// Chroma key and despill cases.  pnpm test:web
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyDespill, borderKey, chromaMask, DEFAULT_CHROMA, despillFor, keyDistance, type RGB } from "./chroma.ts";

function image(w: number, h: number, fill: RGB) {
  const px = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) px.set([...fill, 255], 4 * i);
  return px;
}
const put = (px: Uint8ClampedArray, w: number, x: number, y: number, c: RGB) => px.set(c, 4 * (y * w + x));

test("chroma key: contiguous keeps the hole, soft edges", () => {
  const w = 40, h = 40;
  const img = image(w, h, [20, 200, 30]);
  for (let y = 10; y < 30; y++) for (let x = 10; x < 30; x++) put(img, w, x, y, [200, 40, 40]);
  put(img, w, 20, 20, [20, 200, 30]);
  const key = borderKey(img, w, h);
  assert.deepEqual(key, [20, 200, 30]);
  let m = chromaMask(img, w, h, key, DEFAULT_CHROMA);
  assert.equal(m[0], 0, "background removed");
  assert.equal(m[15 * w + 15], 255, "subject kept");
  assert.equal(m[20 * w + 20], 255, "inner hole preserved (contiguous)");
  m = chromaMask(img, w, h, key, { ...DEFAULT_CHROMA, contiguous: false });
  assert.equal(m[20 * w + 20], 0, "without contiguous, the hole is removed");
  assert.ok(keyDistance(30, 250, 45, [20, 200, 30]) < 15);
  assert.ok(keyDistance(30, 80, 40, [30, 200, 40]) > 100);
});

const LUMA = [0.2126, 0.7152, 0.0722];
const apply = (d: NonNullable<ReturnType<typeof despillFor>>, p: RGB): RGB => {
  const px = new Uint8ClampedArray([...p, 255]);
  applyDespill(px, new Uint8Array([255]), d);
  return [px[0], px[1], px[2]];
};

test("despill removes green spill without darkening", () => {
  const d = despillFor([30, 200, 40], 100)!;
  assert.ok(d);
  const ball = apply(d, [30, 80, 40]);
  assert.ok(ball[1] <= Math.max(ball[0], ball[2]), `no dominant green: ${ball}`);
  assert.ok(ball.reduce((s, v, i) => s + v * LUMA[i], 0) > 50, `luminance preserved: ${ball}`);
  for (const keep of [[200, 150, 120], [220, 220, 30], [60, 60, 60]] as RGB[]) assert.deepEqual(apply(d, keep), keep);
  const half = apply(despillFor([30, 200, 40], 50)!, [40, 140, 40]);
  assert.ok(half[1] > half[0] && half[1] < 140, `${half}`);
  assert.equal(despillFor([20, 60, 210], 100)?.channel, 2);
  assert.equal(despillFor([150, 190, 150], 100), null, "light-green wall");
  assert.equal(despillFor([210, 120, 60], 100), null, "orange background");
  assert.equal(despillFor([30, 200, 40], 0), null);
});

test("despill leaves pixels where the mask is 0 untouched", () => {
  const px = new Uint8ClampedArray([30, 200, 40, 255]);
  applyDespill(px, new Uint8Array([0]), despillFor([30, 200, 40], 100)!);
  assert.deepEqual([...px], [30, 200, 40, 255]);
});
