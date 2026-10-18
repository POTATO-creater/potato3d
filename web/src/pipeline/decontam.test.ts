// Edge decontamination cases.  pnpm test:web
import assert from "node:assert/strict";
import { test } from "node:test";
import { decontaminate } from "./decontam.ts";
import { erode } from "./erode.ts";

// The matte leaks 1 px past the real outline: its erosion is the true mixing ratio.
test("decontamination: soft edge of a red subject on a white wall becomes red again", () => {
  const w = 96, h = 64;
  const F = [200, 30, 40], B = [240, 240, 235];
  const mask = new Uint8Array(w * h);
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let p = 0; p < w * h; p++) mask[p] = Math.min(Math.max((((p % w) - 40) / 8) * 255, 0), 255);
  const alpha = erode(mask, w, h, 1);
  for (let p = 0; p < w * h; p++) {
    const m = alpha[p] / 255;
    for (let c = 0; c < 3; c++) rgba[4 * p + c] = Math.round(m * F[c] + (1 - m) * B[c]);
    rgba[4 * p + 3] = 255;
  }
  const before = rgba.slice();
  decontaminate(rgba, mask, alpha, w, h);
  const dist = (px: Uint8ClampedArray, p: number) => Math.hypot(...[0, 1, 2].map((c) => px[4 * p + c] - F[c]));
  for (let x = 41; x < 48; x++) {
    const p = 30 * w + x;
    if (alpha[p] > 0 && alpha[p] < 255) {
      assert.ok(dist(rgba, p) < 12, `x=${x}: edge still mixed`);
      assert.ok(dist(rgba, p) < dist(before, p), `x=${x}: no improvement`);
    }
  }
  const px = (a: Uint8ClampedArray, x: number) => Array.from(a.slice(4 * (30 * w + x), 4 * (30 * w + x) + 4));
  assert.deepEqual(px(rgba, 80), px(before, 80), "solid interior untouched");
  assert.deepEqual(px(rgba, 5), px(before, 5), "background untouched");
});

test("decontamination: no background anywhere keeps colors", () => {
  const w = 32, h = 32;
  const mask = new Uint8Array(w * h).fill(200);
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let p = 0; p < w * h; p++) rgba.set([10, 120, 230, 255], 4 * p);
  const before = rgba.slice();
  decontaminate(rgba, mask, mask.slice(), w, h);
  assert.deepEqual(rgba, before);
});
