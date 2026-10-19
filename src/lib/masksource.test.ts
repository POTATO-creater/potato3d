import { test } from "node:test";
import assert from "node:assert/strict";
import { fpsStep, maskFrameFor, maskLevel, paddedFrames, rgbaToMask, workflowSize } from "./masksource.ts";

test("attached video: frame k is at k / fps", () => {
  const src = { count: 97, positions: null, fps: 30 };
  assert.deepEqual(maskFrameFor(src, 0), { index: 0, gap: 0 });
  assert.deepEqual(maskFrameFor(src, 1), { index: 30, gap: 0 });
  // LTX returned 97 of 100 frames: the tail clamps to the last one.
  assert.deepEqual(maskFrameFor(src, 99 / 30), { index: 96, gap: 3 });
});

test("attached images: frame k is image k", () => {
  const src = { count: 10, positions: null, fps: null };
  assert.deepEqual(maskFrameFor(src, 4), { index: 4, gap: 0 });
  assert.deepEqual(maskFrameFor(src, 12), { index: 9, gap: 3 });
});

test("generated: nearest position, gap in source frames", () => {
  // Trimmed from 2 s at 60 fps, every other frame.
  const positions = Array.from({ length: 50 }, (_, i) => 2 + i / 30);
  const src = { count: 57, positions, fps: 60 };
  assert.deepEqual(maskFrameFor(src, 2), { index: 0, gap: 0 });
  const mid = maskFrameFor(src, 2 + 1 / 60);
  assert.ok(mid.index === 0 || mid.index === 1);
  assert.ok(Math.abs(mid.gap - 1) < 1e-9);
  // Before the trim: far away.
  assert.ok(maskFrameFor(src, 1).gap > 50);
  // Padding frames (count > positions) are never chosen.
  assert.equal(maskFrameFor(src, 10).index, 49);
});

test("padding to 8n + 1", () => {
  assert.equal(paddedFrames(1), 1);
  assert.equal(paddedFrames(9), 9);
  assert.equal(paddedFrames(10), 17);
  assert.equal(paddedFrames(100), 105);
});

test("fps step and size", () => {
  assert.equal(fpsStep(30), 1);
  assert.equal(fpsStep(24), 1);
  assert.equal(fpsStep(60), 2);
  assert.equal(fpsStep(59.94), 2);
  assert.deepEqual(workflowSize(3840, 2160), [1920, 1080]);
  assert.deepEqual(workflowSize(1081, 1921), [1080, 1920]);
  assert.deepEqual(workflowSize(721, 405), [722, 406]);
});

test("mask levels", () => {
  assert.equal(maskLevel(0), 0);
  assert.equal(maskLevel(6), 0);
  assert.equal(maskLevel(250), 255);
  assert.ok(maskLevel(128) > 120 && maskLevel(128) < 136);
  assert.deepEqual([...rgbaToMask(new Uint8Array([255, 255, 255, 255, 3, 3, 3, 255]))], [255, 0]);
});
