// RMBG pre/post-processing cases.
import assert from "node:assert/strict";
import { test } from "node:test";
import { inputSize, MODELS, postprocess, preprocess } from "./segment.ts";

test("RMBG: 1024², x − 0.5 and min–max stretch", () => {
  assert.deepEqual(inputSize("rmbg", 64, 48), [1024, 1024]);
  const t = preprocess(new Uint8ClampedArray([255, 0, 128, 255]), 1, 1, MODELS.rmbg);
  assert.ok(Math.abs(t[0] - 0.5) < 1e-3 && Math.abs(t[1] + 0.5) < 1e-3 && Math.abs(t[2]) < 0.01);
  const out = new Float32Array(8).fill(0.1);
  out.fill(0.9, 0, 4);
  const m = postprocess("rmbg", out);
  assert.deepEqual([m[0], m[7]], [255, 0]);
  assert.ok(postprocess("rmbg", new Float32Array(8).fill(0.02)).every((v) => v === 5));
});
