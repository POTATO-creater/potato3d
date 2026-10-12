import assert from "node:assert/strict";
import { test } from "node:test";
import {
  clipFrames,
  clipLens,
  DEFAULT_CLIP,
  matToQuat,
  orbitFrame,
  poseToWorld,
  presetPath,
  quatToMat,
  viewPath,
  type CameraPose,
  type Quat,
  type Vec3,
} from "./trajectory.ts";

// Mundo com "para cima" = -Y, como costuma sair do COLMAP.
const UP: Vec3 = [0, -1, 0];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit = (a: Vec3): Vec3 => {
  const n = Math.hypot(...a);
  return [a[0] / n, a[1] / n, a[2] / n];
};
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Pose COLMAP de uma câmera em (azimute, elevação, raio) olhando a origem. */
function lookAt(azDeg: number, elDeg: number, r: number, name = ""): CameraPose {
  const az = (azDeg * Math.PI) / 180, el = (elDeg * Math.PI) / 180;
  const c: Vec3 = [r * Math.cos(el) * Math.sin(az), -r * Math.sin(el), r * Math.cos(el) * Math.cos(az)];
  const f = unit(sub([0, 0, 0], c));
  const down = unit(sub([0, 0, 0], sub(UP, [f[0] * dot(UP, f), f[1] * dot(UP, f), f[2] * dot(UP, f)])));
  const right = cross(down, f);
  // Câmera → mundo tem colunas (direita, baixo, frente); mundo → câmera é a transposta.
  const R: [Vec3, Vec3, Vec3] = [right, down, f];
  const qvec = matToQuat(R);
  const tvec: Vec3 = [0, 1, 2].map((i) => -dot(R[i], c)) as Vec3;
  return { name, qvec, tvec };
}

const orbit = Array.from({ length: 60 }, (_, i) => lookAt(i * 6, 10, 3, `${String(i).padStart(5, "0")}.jpg`));
const lens = { width: 960, height: 1920, fx: 1600, fy: 1600 };
const elevation = (p: { qvec: Quat; tvec: Vec3 }) => {
  const c = poseToWorld(p).center;
  return (Math.asin(dot(unit(c), UP)) * 180) / Math.PI;
};
const angleBetween = (a: Quat, b: Quat) => {
  const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  return (2 * Math.acos(Math.min(1, d)) * 180) / Math.PI;
};

test("quatérnio ↔ matriz ida e volta", () => {
  for (const p of orbit.slice(0, 10)) assert.deepEqual(matToQuat(quatToMat(p.qvec)).map((v) => v.toFixed(9)), p.qvec.map((v) => v.toFixed(9)));
});

test("órbita: centro na origem e up recuperado", () => {
  const { center, up } = orbitFrame(orbit);
  assert.ok(Math.hypot(...center) < 1e-6, `centro ${center}`);
  assert.ok(dot(up, UP) > 0.999);
});

test("tamanho do clipe: múltiplos de 32 e focal na mesma escala", () => {
  assert.deepEqual(clipLens(lens, 960), { size: [480, 960], intrinsics: { fx: 800, fy: 800, cx: 240, cy: 480 } });
  const { size } = clipLens({ width: 1920, height: 1080, fx: 1500, fy: 1500 }, 960);
  assert.deepEqual(size, [960, 544]);
});

test("preset −30°: pontas nas fotos, meio 30° abaixo, raio mantido", () => {
  const plan = presetPath(orbit, lens, { ...DEFAULT_CLIP, center: 20, elevDeg: -30 });
  const p = plan.poses;
  assert.equal(p.length, 97);
  // 48 fotos de ponta a ponta (meia foto por frame), dando a volta na órbita fechada.
  assert.equal(p[0].anchor, orbit[56].name);
  assert.equal(p[96].anchor, orbit[44].name);
  assert.ok(angleBetween(p[0].qvec, orbit[56].qvec) < 1e-4);
  assert.ok(Math.abs(elevation(p[48]) - (10 - 30)) < 0.5, `elevação no meio ${elevation(p[48])}`);
  assert.ok(Math.abs(Math.hypot(...poseToWorld(p[48]).center) - 3) < 1e-6);
  for (let i = 1; i < p.length; i++) assert.ok(angleBetween(p[i - 1].qvec, p[i].qvec) < 4, `salto no frame ${i}`);
});

test("vista mirada: meio igual ao alvo, pontas em fotos, sem saltos", () => {
  const target = lookAt(93, -35, 2.4);
  const plan = viewPath(orbit, lens, target, { ...DEFAULT_CLIP, span: 16, sweepDeg: 10 });
  const p = plan.poses;
  assert.ok(p[0].anchor && p[96].anchor);
  // No frame do meio a varredura é zero: a pose é exatamente a mirada.
  assert.ok(angleBetween(p[48].qvec, target.qvec) < 1e-4);
  assert.ok(Math.hypot(...sub(poseToWorld(p[48]).center, poseToWorld(target).center)) < 1e-6);
  for (let i = 1; i < p.length; i++) assert.ok(angleBetween(p[i - 1].qvec, p[i].qvec) < 5, `salto no frame ${i}`);
  // A mistura passa por fora do sujeito (raio nunca menor que o menor dos dois).
  for (const q of p) assert.ok(Math.hypot(...poseToWorld(q).center) > 2.39);
});

test("duração em frames: 8n + 1 mais próximo", () => {
  assert.equal(clipFrames(97 / 24, 24), 97);
  assert.equal(clipFrames(3.2, 30), 97);
  assert.equal(clipFrames(8, 30), 241);
  assert.equal(clipFrames(0.1, 30), 17);
});

test("volta 360° por baixo: começa e termina na mesma foto, passa por todas", () => {
  const frames = 241;
  const plan = presetPath(orbit, lens, { ...DEFAULT_CLIP, frames, center: 10 + orbit.length / 2, span: orbit.length, elevDeg: -25 });
  const p = plan.poses;
  assert.equal(p[0].anchor, orbit[10].name);
  assert.equal(p[frames - 1].anchor, orbit[10].name);
  assert.ok(Math.abs(elevation(p[frames >> 1]) - (10 - 25)) < 0.5);
  for (let i = 1; i < p.length; i++) assert.ok(angleBetween(p[i - 1].qvec, p[i].qvec) < 4, `salto no frame ${i}`);
});

test("onda 360°: desce numa metade e sobe na outra", () => {
  const frames = 241;
  const plan = presetPath(orbit, lens, { ...DEFAULT_CLIP, frames, center: orbit.length / 2, span: orbit.length, elevDeg: -20, shape: "wave" });
  const els = plan.poses.map(elevation);
  assert.ok(Math.min(...els.slice(0, frames >> 1)) < 10 - 18, "metade de baixo");
  assert.ok(Math.max(...els.slice(frames >> 1)) > 10 + 18, "metade de cima");
  assert.ok(Math.abs(els[0] - 10) < 1e-6 && Math.abs(els[frames - 1] - 10) < 1e-6);
  assert.equal(plan.poses[0].anchor, orbit[0].name);
});
