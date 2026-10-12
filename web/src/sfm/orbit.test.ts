// Correções da órbita numa órbita sintética com poses conhecidas.
import assert from "node:assert/strict";
import { test } from "node:test";
import type { Pose } from "./geometry";
import { cross, mulTVec3, mulVec3, transpose3, mul3, type Vec3 } from "./linalg.ts";
import { fillMissing, flattenOrbit, refinePose } from "./orbit.ts";
import type { SfmModel } from "./reconstruct";

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const unit = (a: Vec3): Vec3 => {
  const l = Math.hypot(...a);
  return [a[0] / l, a[1] / l, a[2] / l];
};

/** Câmera em `c` olhando para a origem, "para cima" ≈ +z (COLMAP: y para baixo). */
function lookAt(c: Vec3): Pose {
  const z = unit(sub([0, 0, 0], c));
  const x = unit(cross(z, [0, 0, -1]));
  const y = cross(z, x);
  const R = Float64Array.from([...x, ...y, ...z]);
  const t = mulVec3(R, c);
  return { R, t: [-t[0], -t[1], -t[2]] };
}

const centerOf = (p: Pose): Vec3 => {
  const c = mulTVec3(p.R, p.t);
  return [-c[0], -c[1], -c[2]];
};
const angleBetween = (a: Pose, b: Pose) => {
  const M = mul3(a.R, transpose3(b.R));
  return Math.acos(Math.min(1, Math.max(-1, (M[0] + M[4] + M[8] - 1) / 2)));
};

/** 96 frames a 24 fps (4 s), velocidade angular variando, raio 4, altura 1. */
function orbit(heightAt = (_i: number) => 1) {
  const n = 96;
  const at = Array.from({ length: n }, (_, i) => i / 24);
  const truth = at.map((s, i) => {
    const th = 2 * Math.PI * 0.85 * (s / 4) * (0.8 + 0.2 * (s / 4));
    return lookAt([4 * Math.cos(th), 4 * Math.sin(th), heightAt(i)]);
  });
  const model: SfmModel = {
    f: 500, cx: 320, cy: 240, width: 640, height: 480,
    poses: truth.map((p) => ({ R: p.R.slice(), t: [...p.t] as Vec3 })),
    points: [
      { X: [0.1, 0, 0], rgb: [0, 0, 0], obs: [], error: 0 },
      { X: [0, 0.1, 0.2], rgb: [0, 0, 0], obs: [], error: 0 },
      { X: [-0.1, -0.1, -0.1], rgb: [0, 0, 0], obs: [], error: 0 },
    ],
    stats: {},
  };
  return { at, truth, model };
}

test("fillMissing recupera frames perdidos no meio e nas pontas", () => {
  const { at, truth, model } = orbit();
  const lost = [0, 1, 20, 21, 22, 23, 50, 94, 95];
  for (const i of lost) model.poses[i] = null;
  assert.deepEqual(fillMissing(model, at).inferred, lost);
  for (const i of lost) {
    const err = Math.hypot(...sub(centerOf(model.poses[i]!), centerOf(truth[i])));
    assert.ok(err < 0.005 * 4, `frame ${i}: centro a ${(err / 4 * 100).toFixed(2)}% do raio`);
    const ang = (angleBetween(model.poses[i]!, truth[i]) * 180) / Math.PI;
    assert.ok(ang < 0.3, `frame ${i}: orientação a ${ang.toFixed(2)}°`);
  }
  // No meio da lacuna a órbita é seguida, não a corda: raio preservado.
  const c = centerOf(model.poses[21]!);
  assert.ok(Math.abs(Math.hypot(c[0], c[1]) - 4) < 0.01);
});

test("flattenOrbit nivela as câmeras e mantém o sujeito no centro", () => {
  const { model } = orbit((i) => 1 + 0.3 * Math.sin(i / 5));
  // Pontos observados pelas câmeras, para o BA dos pontos.
  const xy = model.poses.map((p) => {
    const out: number[] = [];
    for (const pt of model.points) {
      const c = mulVec3(p!.R, pt.X);
      const z = c[2] + p!.t[2];
      out.push((500 * (c[0] + p!.t[0])) / z + 320, (500 * (c[1] + p!.t[1])) / z + 240);
    }
    return out;
  });
  model.points.forEach((pt, j) => (pt.obs = model.poses.map((_, i) => [i, j] as [number, number])));
  const dev = flattenOrbit(model, xy);
  assert.ok(dev != null && dev > 0.05);
  // Todas as câmeras no mesmo plano.
  const cs = model.poses.map((p) => centerOf(p!));
  const n = unit(cross(sub(cs[30], cs[0]), sub(cs[60], cs[0])));
  for (const c of cs) assert.ok(Math.abs(n.reduce((a, v, k) => a + v * (c[k] - cs[0][k]), 0)) < 1e-6, "coplanares");
  // A origem continua no centro da imagem de cada câmera.
  for (const p of model.poses) {
    const c = p!.t;
    assert.ok(Math.hypot(c[0] / c[2], c[1] / c[2]) < 1e-9);
  }
  assert.ok(Number.isFinite(Number(model.stats.meanError)));
});

test("refinePose volta de um palpite grosseiro mesmo com matches errados", () => {
  const { truth, model } = orbit();
  const pose = truth[30];
  const c = centerOf(pose);
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31) - 0.5;
  const n = 200;
  const uv = new Float64Array(2 * n), X = new Float64Array(3 * n);
  for (let j = 0; j < n; j++) {
    const x: Vec3 = [rnd() * 2, rnd() * 2, rnd() * 2];
    const p = mulVec3(pose.R, x).map((v, k) => v + pose.t[k]);
    uv[2 * j] = (500 * p[0]) / p[2] + 320 + rnd();
    uv[2 * j + 1] = (500 * p[1]) / p[2] + 240 + rnd();
    if (j % 5 === 0) {
      uv[2 * j] += rnd() * 400;
      uv[2 * j + 1] += rnd() * 400;
    }
    X.set(x, 3 * j);
  }
  // Palpite a ~3% do raio e ~1,5° da verdade; prior fraco.
  const c0: Vec3 = [c[0] + 0.08, c[1] - 0.06, c[2] + 0.05];
  const guess = lookAt(c0);
  const r = refinePose(model, guess.R, c0, 1, { uv, X, n });
  assert.ok(r);
  const err = Math.hypot(...sub(centerOf(r.pose), c)) / 4;
  assert.ok(err < 0.002, `centro a ${(err * 100).toFixed(3)}% do raio`);
  assert.ok((angleBetween(r.pose, pose) * 180) / Math.PI < 0.1);
  assert.ok(r.inliers >= 150);
  assert.equal(refinePose(model, guess.R, c0, 1, { uv: uv.subarray(0, 20), X: X.subarray(0, 30), n: 10 }), null);
});
