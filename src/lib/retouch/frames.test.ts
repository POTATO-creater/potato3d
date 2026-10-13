import assert from "node:assert/strict";
import { test } from "node:test";
import { pickRestoredFrames, restoredLimit } from "./frames.ts";

test("limite: 25% de 107 fotos", () => {
  assert.equal(restoredLimit(107, 0.25), 27);
  assert.equal(restoredLimit(3, 0.05), 1);
});

test("sem cortar quando cabe tudo", () => {
  assert.deepEqual(pickRestoredFrames([3, 2], 10), [[0, 1, 2], [0, 1]]);
});

test("cota proporcional ao tamanho do clipe, espaçada e sem as pontas", () => {
  const [a, b] = pickRestoredFrames([89, 177], 30);
  assert.equal(a.length + b.length, 30);
  assert.equal(a.length, 10);
  assert.equal(b.length, 20);
  for (const [sel, n] of [[a, 89], [b, 177]] as const) {
    assert.ok(sel[0] > 0 && sel[sel.length - 1] < n - 1, "sem as pontas");
    const gaps = sel.slice(1).map((v, i) => v - sel[i]);
    assert.ok(Math.max(...gaps) - Math.min(...gaps) <= 1, `espaçamento ${gaps}`);
  }
});

test("todo clipe entra com pelo menos 1 frame", () => {
  const sel = pickRestoredFrames([500, 9], 10);
  assert.equal(sel[1].length, 1);
  assert.equal(sel[0].length + sel[1].length, 10);
});
