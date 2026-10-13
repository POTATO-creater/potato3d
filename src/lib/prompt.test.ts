import assert from "node:assert/strict";
import { test } from "node:test";
import { eraseAt, promptInTrim, withMarks } from "./prompt.ts";

const pt = (x: number, y: number, label: 0 | 1 = 1) => ({ x, y, label });

test("Shift+click erases the nearest point, else the box", () => {
  const m = { box: [0.2, 0.2, 0.8, 0.8] as [number, number, number, number], points: [pt(0.5, 0.5), pt(0.52, 0.5, 0)] };
  assert.deepEqual(eraseAt(m, 0.515, 0.5, 1).points, [pt(0.5, 0.5)]);
  // inside the box, away from points: the box goes
  assert.equal(eraseAt(m, 0.3, 0.3, 1).box, null);
  // outside everything: nothing changes
  assert.equal(eraseAt(m, 0.95, 0.95, 1), m);
  // wide frame (2:1): 0.04 of height equals 0.02 of width
  assert.equal(eraseAt({ box: null, points: [pt(0.5, 0.5)] }, 0.5, 0.57, 2).points.length, 0);
});

test("marking only applies inside the trim range; no marks falls back to automatic", () => {
  const p = { time: 2, box: null, points: [pt(0.5, 0.5)] };
  assert.ok(promptInTrim(p, { start: 1, end: null }));
  assert.ok(!promptInTrim(p, { start: 2.5, end: null }));
  assert.ok(!promptInTrim(p, { start: 0, end: 1.5 }));
  assert.ok(promptInTrim({ ...p, time: null, image: "a.png" }, { start: 5, end: 6 }));
  assert.equal(withMarks(p, { box: null, points: [] }), null);
  assert.deepEqual(withMarks(p, { box: [0, 0, 1, 1], points: [] }), { ...p, box: [0, 0, 1, 1], points: [] });
});
