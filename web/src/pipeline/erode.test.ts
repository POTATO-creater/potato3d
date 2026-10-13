// Same case as the erode_shrinks_one_pixel test in masks.rs.  pnpm test:web
import assert from "node:assert/strict";
import { test } from "node:test";
import { erode } from "./erode.ts";

test("erosion: shrinks 1 px, but not at the image border", () => {
  const w = 10, h = 8;
  const m = new Uint8Array(w * h);
  for (let y = 2; y < 6; y++) for (let x = 2; x < 6; x++) m[y * w + x] = 255;
  for (let y = 0; y < h; y++) {
    m[y * w + 9] = 255;
    m[y * w + 8] = 128;
  }
  const e = erode(m, w, h, 1);
  const at = (x: number, y: number) => e[y * w + x];
  assert.deepEqual([at(2, 2), at(3, 3), at(4, 4), at(5, 5)], [0, 255, 255, 0]);
  assert.deepEqual([at(9, 0), at(9, 7), at(8, 3)], [128, 128, 0]);
  assert.deepEqual(erode(m, w, h, 0), m);
});
