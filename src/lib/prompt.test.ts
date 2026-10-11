import assert from "node:assert/strict";
import { test } from "node:test";
import { eraseAt, promptInTrim, withMarks } from "./prompt.ts";

const pt = (x: number, y: number, label: 0 | 1 = 1) => ({ x, y, label });

test("Shift+clique apaga o ponto mais perto, senão a caixa", () => {
  const m = { box: [0.2, 0.2, 0.8, 0.8] as [number, number, number, number], points: [pt(0.5, 0.5), pt(0.52, 0.5, 0)] };
  assert.deepEqual(eraseAt(m, 0.515, 0.5, 1).points, [pt(0.5, 0.5)]);
  // dentro da caixa, longe dos pontos: some a caixa
  assert.equal(eraseAt(m, 0.3, 0.3, 1).box, null);
  // fora de tudo: nada muda
  assert.equal(eraseAt(m, 0.95, 0.95, 1), m);
  // quadro largo (2:1): 0,04 de altura vale 0,02 de largura
  assert.equal(eraseAt({ box: null, points: [pt(0.5, 0.5)] }, 0.5, 0.57, 2).points.length, 0);
});

test("marcação vale só dentro do trecho; sem marcas volta ao automático", () => {
  const p = { time: 2, box: null, points: [pt(0.5, 0.5)] };
  assert.ok(promptInTrim(p, { start: 1, end: null }));
  assert.ok(!promptInTrim(p, { start: 2.5, end: null }));
  assert.ok(!promptInTrim(p, { start: 0, end: 1.5 }));
  assert.ok(promptInTrim({ ...p, time: null, image: "a.png" }, { start: 5, end: 6 }));
  assert.equal(withMarks(p, { box: null, points: [] }), null);
  assert.deepEqual(withMarks(p, { box: [0, 0, 1, 1], points: [] }), { ...p, box: [0, 0, 1, 1], points: [] });
});
