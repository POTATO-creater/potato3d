import assert from "node:assert/strict";
import { test } from "node:test";
import {
  clipFrames,
  orbitDistance,
  peakSpeed,
  clipLens,
  DEFAULT_CLIP,
  matToQuat,
  orbitFrame,
  poseToWorld,
  presetPath,
  tiltTurnPath,
  turnPath,
  quatToMat,
  poseStill,
  recordedPath,
  recordedSeconds,
  viewPath,
  type CameraPose,
  type Quat,
  type Vec3,
} from "./trajectory.ts";

// World with "up" = -Y, as COLMAP usually outputs.
const UP: Vec3 = [0, -1, 0];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit = (a: Vec3): Vec3 => {
  const n = Math.hypot(...a);
  return [a[0] / n, a[1] / n, a[2] / n];
};
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** COLMAP pose of a camera at (azimuth, elevation, radius) looking at the origin. */
function lookAt(azDeg: number, elDeg: number, r: number, name = ""): CameraPose {
  const az = (azDeg * Math.PI) / 180, el = (elDeg * Math.PI) / 180;
  const c: Vec3 = [r * Math.cos(el) * Math.sin(az), -r * Math.sin(el), r * Math.cos(el) * Math.cos(az)];
  const f = unit(sub([0, 0, 0], c));
  const down = unit(sub([0, 0, 0], sub(UP, [f[0] * dot(UP, f), f[1] * dot(UP, f), f[2] * dot(UP, f)])));
  const right = cross(down, f);
  // Camera → world has columns (right, down, forward); world → camera is the transpose.
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

test("quaternion ↔ matrix round trip", () => {
  for (const p of orbit.slice(0, 10)) assert.deepEqual(matToQuat(quatToMat(p.qvec)).map((v) => v.toFixed(9)), p.qvec.map((v) => v.toFixed(9)));
});

test("orbit: center at origin and up recovered", () => {
  const { center, up } = orbitFrame(orbit);
  assert.ok(Math.hypot(...center) < 1e-6, `center ${center}`);
  assert.ok(dot(up, UP) > 0.999);
});

test("clip size: multiples of 32 and focal length at the same scale", () => {
  assert.deepEqual(clipLens(lens, 960), { size: [480, 960], intrinsics: { fx: 800, fy: 800, cx: 240, cy: 480 } });
  const { size } = clipLens({ width: 1920, height: 1080, fx: 1500, fy: 1500 }, 960);
  assert.deepEqual(size, [960, 544]);
});

test("preset −30°: ends at photos, middle 30° below, radius kept", () => {
  const plan = presetPath(orbit, lens, { ...DEFAULT_CLIP, center: 20, elevDeg: -30 });
  const p = plan.poses;
  assert.equal(p.length, 97);
  // 48 photos end to end (half a photo per frame), wrapping around the closed orbit.
  assert.equal(p[0].anchor, orbit[56].name);
  assert.equal(p[96].anchor, orbit[44].name);
  assert.ok(angleBetween(p[0].qvec, orbit[56].qvec) < 1e-4);
  assert.ok(Math.abs(elevation(p[48]) - (10 - 30)) < 0.5, `elevation at middle ${elevation(p[48])}`);
  assert.ok(Math.abs(Math.hypot(...poseToWorld(p[48]).center) - 3) < 1e-6);
  for (let i = 1; i < p.length; i++) assert.ok(angleBetween(p[i - 1].qvec, p[i].qvec) < 4, `jump at frame ${i}`);
});

test("target view: middle equals target, ends in photo directions, no zoom-in", () => {
  const target = lookAt(93, -35, 2.4);
  const plan = viewPath(orbit, lens, target, { ...DEFAULT_CLIP, span: 16, sweepDeg: 10 });
  const p = plan.poses;
  // Ends: exact orientation of a photo, but at the view's distance.
  for (const end of [p[0], p[96]]) assert.ok(orbit.some((c) => angleBetween(end.qvec, c.qvec) < 1e-4));
  for (const q of p) assert.ok(Math.abs(Math.hypot(...poseToWorld(q).center) - 2.4) < 1e-6, "constant distance");
  // At the middle frame the sweep is zero: the pose is exactly the target.
  assert.ok(angleBetween(p[48].qvec, target.qvec) < 1e-4);
  assert.ok(Math.hypot(...sub(poseToWorld(p[48]).center, poseToWorld(target).center)) < 1e-6);
  for (let i = 1; i < p.length; i++) assert.ok(angleBetween(p[i - 1].qvec, p[i].qvec) < 5, `jump at frame ${i}`);
});

test("duration in frames: nearest 8n + 1", () => {
  assert.equal(clipFrames(97 / 24, 24), 97);
  assert.equal(clipFrames(3.2, 30), 97);
  assert.equal(clipFrames(8, 30), 241);
  assert.equal(clipFrames(0.1, 30), 17);
});

test("360° turn from below: starts and ends at the same photo, passes all of them", () => {
  const frames = 241;
  const plan = presetPath(orbit, lens, { ...DEFAULT_CLIP, frames, center: 10 + orbit.length / 2, span: orbit.length, elevDeg: -25 });
  const p = plan.poses;
  assert.equal(p[0].anchor, orbit[10].name);
  assert.equal(p[frames - 1].anchor, orbit[10].name);
  assert.ok(Math.abs(elevation(p[frames >> 1]) - (10 - 25)) < 0.5);
  for (let i = 1; i < p.length; i++) assert.ok(angleBetween(p[i - 1].qvec, p[i].qvec) < 4, `jump at frame ${i}`);
});

test("360° wave: goes down in one half and up in the other", () => {
  const frames = 241;
  const plan = presetPath(orbit, lens, { ...DEFAULT_CLIP, ramp: 0.1, frames, center: orbit.length / 2, span: orbit.length, elevDeg: -20, shape: "wave" });
  const els = plan.poses.map(elevation);
  assert.ok(Math.min(...els.slice(0, frames >> 1)) < 10 - 18, "lower half");
  assert.ok(Math.max(...els.slice(frames >> 1)) > 10 + 18, "upper half");
  assert.ok(Math.abs(els[0] - 10) < 1e-6 && Math.abs(els[frames - 1] - 10) < 1e-6);
  assert.equal(plan.poses[0].anchor, orbit[0].name);
});

test("requested radius: whole clip at that distance, no false anchors", () => {
  const plan = presetPath(orbit, lens, { ...DEFAULT_CLIP, center: 20, span: 20, elevDeg: -30, radius: 2 });
  for (const q of plan.poses) assert.ok(Math.abs(Math.hypot(...poseToWorld(q).center) - 2) < 1e-6);
  assert.equal(plan.poses[0].anchor, null);
  assert.ok(Math.abs(orbitDistance(orbit, lookAt(40, 0, 1.7)) - 1.7) < 1e-6);
});

test("speed: flat 4 s turn gives 90°/s; without held ends the ramp is slower", () => {
  const flat = presetPath(orbit, lens, { ...DEFAULT_CLIP, center: 30, span: orbit.length, elevDeg: 0 });
  assert.ok(Math.abs(peakSpeed(flat.poses, 24) - 90) < 2, `${peakSpeed(flat.poses, 24)}`);
  const target = lookAt(93, -35, 3);
  const old = viewPath(orbit, lens, target, { ...DEFAULT_CLIP, hold: 0.12, ramp: 0.23, span: 10, sweepDeg: 0 });
  const now = viewPath(orbit, lens, target, { ...DEFAULT_CLIP, span: 10, sweepDeg: 0 });
  assert.ok(peakSpeed(now.poses, 24) < 0.65 * peakSpeed(old.poses, 24));
});

test("irregularly spaced photos (and the last repeating the first): constant speed", () => {
  // Steps alternating 3° and 9°, plus a final photo almost at the first one's position.
  const az: number[] = [];
  for (let a = 0; a < 360 - 1e-9; ) {
    az.push(a);
    a += az.length % 2 ? 3 : 9;
  }
  az.push(359.97);
  const cams = az.map((a, i) => lookAt(a, 10, 3, `${String(i).padStart(5, "0")}.jpg`));
  const plan = presetPath(cams, lens, { ...DEFAULT_CLIP, frames: 121, center: cams.length / 2, span: cams.length, elevDeg: 0 });
  const p = plan.poses;
  assert.equal(p[0].anchor, cams[0].name);
  assert.equal(p[120].anchor, cams[0].name);
  const steps = p.slice(1).map((q, i) => angleBetween(p[i].qvec, q.qvec));
  const mean = 360 / 120;
  for (const s of steps) assert.ok(Math.abs(s - mean) < 0.05, `step ${s.toFixed(3)}° (expected ${mean}°)`);
});

test("360° turn from view: starts and ends at it, same distance and height, in orbit direction", () => {
  const target = lookAt(40, -25, 2);
  const plan = turnPath(orbit, lens, target, { ...DEFAULT_CLIP, frames: 121 });
  const p = plan.poses;
  assert.ok(angleBetween(p[0].qvec, target.qvec) < 1e-4 && angleBetween(p[120].qvec, target.qvec) < 1e-4);
  for (const q of p) {
    assert.ok(Math.abs(Math.hypot(...poseToWorld(q).center) - 2) < 1e-6);
    assert.ok(Math.abs(elevation(q) + 25) < 1e-6);
  }
  // The test orbit moves +6° of azimuth per photo: a quarter turn later, the camera is at 40° + 90°.
  const c = poseToWorld(p[30]).center;
  const az = (Math.atan2(c[0], c[2]) * 180) / Math.PI;
  assert.ok(Math.abs(az - 130) < 0.5, `azimuth ${az}`);
});

test("360° turn with radius: same view direction and elevation angle, at requested distance", () => {
  const target = lookAt(40, -25, 2);
  const p = turnPath(orbit, lens, target, { ...DEFAULT_CLIP, frames: 49, radius: 3 }).poses;
  assert.ok(angleBetween(p[0].qvec, target.qvec) < 1e-4);
  for (const q of p) {
    assert.ok(Math.abs(Math.hypot(...poseToWorld(q).center) - 3) < 1e-6);
    assert.ok(Math.abs(elevation(q) + 25) < 1e-6);
  }
});

test("chosen aspect: multiples of 32 and same vertical field of view as the lens", () => {
  const fovY = (k: { fy: number }, h: number) => 2 * Math.atan(h / 2 / k.fy);
  const lensFov = fovY(lens, lens.height);
  for (const [aspect, size] of [[16 / 9, [960, 544]], [1, [960, 960]], [9 / 16, [544, 960]]] as const) {
    const c = clipLens(lens, 960, aspect);
    assert.deepEqual(c.size, size);
    assert.ok(Math.abs(fovY(c.intrinsics, c.size[1]) - lensFov) < 1e-9);
    assert.equal(c.intrinsics.cx, c.size[0] / 2);
  }
  const plan = presetPath(orbit, lens, { ...DEFAULT_CLIP, center: 0, elevDeg: -20, aspect: 16 / 9 });
  assert.deepEqual(plan.size, [960, 544]);
});

// Height of a point (along "up") and where the camera looks.
const height = (p: { qvec: Quat; tvec: Vec3 }) => dot(poseToWorld(p).center, UP);
const forward = (p: { qvec: Quat; tvec: Vec3 }): Vec3 => {
  const w = poseToWorld(p);
  return [w.rot[0][2], w.rot[1][2], w.rot[2][2]];
};
const upright = (p: { qvec: Quat; tvec: Vec3 }) => {
  const w = poseToWorld(p);
  return dot([-w.rot[0][1], -w.rot[1][1], -w.rot[2][1]], UP);
};
const photoHeight = 3 * Math.sin((10 * Math.PI) / 180);
/** Pose looking at a point (UP = −Y, no roll). */
function aim(c: Vec3, at: Vec3): { qvec: Quat; tvec: Vec3 } {
  const f = unit(sub(at, c));
  const down = unit(sub([0, 0, 0], sub(UP, [f[0] * dot(UP, f), f[1] * dot(UP, f), f[2] * dot(UP, f)])));
  const R: [Vec3, Vec3, Vec3] = [cross(down, f), down, f];
  return { qvec: matToQuat(R), tvec: [0, 1, 2].map((i) => -dot(R[i], c)) as Vec3 };
}

test("tilted 360° turn: straight lift from photo height to the view, turn there, descent to start", () => {
  const target = lookAt(40, 40, 2);
  const p = tiltTurnPath(orbit, lens, target, { ...DEFAULT_CLIP, frames: 241 }).poses;
  const n = p.length;
  assert.ok(Math.abs(height(p[0]) - photoHeight) < 1e-6, `starts at photo height (${height(p[0])})`);
  assert.ok(angleBetween(p[0].qvec, p[n - 1].qvec) < 1e-4 && Math.abs(height(p[n - 1]) - photoHeight) < 1e-6, "ends at the start");
  // Straight lift: the first frames stay at the view's azimuth, always aiming
  // at the center (the view's aim point), and reach its tilt.
  const az = (q: { qvec: Quat; tvec: Vec3 }) => { const c = poseToWorld(q).center; return (Math.atan2(c[0], c[2]) * 180) / Math.PI; };
  let k = 0;
  while (k < n && Math.abs(az(p[k]) - 40) < 1e-6) k++;
  assert.ok(k > 3 && k < n / 3, `lift over ${k} frames`);
  assert.ok(Math.abs(dot(forward(p[0]), UP)) < 1e-9 && upright(p[0]) > 0.9999, "starts level");
  for (let i = 1; i < k; i++) assert.ok(elevation(p[i]) >= elevation(p[i - 1]) - 1e-9, "only rises");
  assert.ok(Math.abs(elevation(p[k - 1]) - 40) < 1, `reaches the tilt (${elevation(p[k - 1])})`);
  assert.ok(Math.abs(elevation(p[n >> 1]) - 40) < 1e-6, `middle at the tilt (${elevation(p[n >> 1])})`);
  for (let i = 1; i < n; i++) assert.ok(angleBetween(p[i - 1].qvec, p[i].qvec) < 6, `jump at frame ${i}`);
});

test("tilted 360° turn: view from above and behind aiming at the head starts upright, aiming at the head", () => {
  const head: Vec3 = [0, -0.7, 0];
  const az = Math.PI, el = (50 * Math.PI) / 180, r = 1;
  const target = aim([r * Math.cos(el) * Math.sin(az), -r * Math.sin(el), r * Math.cos(el) * Math.cos(az)], head);
  const p = tiltTurnPath(orbit, lens, target, DEFAULT_CLIP).poses;
  assert.ok(upright(p[0]) > 0.9, `upright (${upright(p[0])})`);
  assert.ok(Math.abs(height(p[0]) - photoHeight) < 1e-6, `at photo height (${height(p[0])})`);
  assert.ok(Math.abs(dot(forward(p[0]), UP)) < 1e-9, "level");
  // Facing the character (the side the view looks toward).
  const toHead = unit(sub(head, poseToWorld(p[0]).center));
  assert.ok(dot(forward(p[0]), toHead) > 0.9, "facing it");
});

test("tilted 360° turn: view from below starts level and reaches the view at the end of the lift", () => {
  const az = (20 * Math.PI) / 180, el = (-35 * Math.PI) / 180, r = 1.5;
  const target = aim([r * Math.cos(el) * Math.sin(az), -r * Math.sin(el), r * Math.cos(el) * Math.cos(az)], [0, -0.3, 0]);
  const p = tiltTurnPath(orbit, lens, target, { ...DEFAULT_CLIP, frames: 241 }).poses;
  assert.ok(Math.abs(dot(forward(p[0]), UP)) < 1e-9 && upright(p[0]) > 0.9999, "starts level and upright");
  assert.ok(Math.abs(height(p[0]) - photoHeight) < 1e-6, "at photo height");
  assert.ok(p.some((q) => angleBetween(q.qvec, target.qvec) < 0.5), "passes through the view");
  for (let i = 1; i < p.length; i++) assert.ok(angleBetween(p[i - 1].qvec, p[i].qvec) < 6, `jump at frame ${i}`);
});


test("recorded motion: pauses trimmed, resampled by time, ends on the recorded poses", () => {
  // 1 s still, 2 s turning 0° → 90° at an uneven pace, 1 s of tiny damping
  // drift, then still. Samples at ~60 Hz with jitter.
  const samples = [];
  let t = 0;
  for (let i = 0; t < 4500; i++) {
    const u = Math.min(1, Math.max(0, (t - 1000) / 2000));
    const drift = t > 3000 && t < 4000 ? 1e-4 * Math.sin(t) : 0;
    samples.push({ t, ...lookAt(90 * u * u + drift, 10, 3) });
    t += 16 + (i % 3);
  }
  assert.ok(Math.abs(recordedSeconds(samples) - 2) < 0.05, `duration ${recordedSeconds(samples)}`);
  const plan = recordedPath(samples, lens, { ...DEFAULT_CLIP, frames: 49, center: [0, 0, 0] })!;
  const p = plan.poses;
  assert.equal(p.length, 49);
  assert.ok(angleBetween(p[0].qvec, lookAt(0, 10, 3).qvec) < 0.2, "starts at the first pose");
  assert.ok(angleBetween(p[48].qvec, lookAt(90, 10, 3).qvec) < 0.2, "ends at the last pose");
  // Halfway in time is a quarter of the way in angle (u² pace kept).
  assert.ok(angleBetween(p[24].qvec, lookAt(22.5, 10, 3).qvec) < 0.5, "keeps the recorded pace");
  assert.ok(Math.abs(elevation(p[24]) - 10) < 0.1);
  assert.ok(p.every((q) => q.anchor === null));
});

test("recorded motion: nothing moved → no path", () => {
  const still = Array.from({ length: 30 }, (_, i) => ({ t: i * 16, ...lookAt(30, 10, 3) }));
  assert.equal(recordedPath(still, lens, { ...DEFAULT_CLIP, center: [0, 0, 0] }), null);
  assert.equal(recordedSeconds(still), 0);
});

test("recorded motion: uniform speed evens out the pace and drops pauses", () => {
  // 0° → 45° fast, a 1 s pause, then 45° → 90° slowly.
  const samples = [];
  for (let t = 0; t <= 4000; t += 16) {
    const az = t < 500 ? (45 * t) / 500 : t < 1500 ? 45 : 45 + (45 * (t - 1500)) / 2500;
    samples.push({ t, ...lookAt(az, 10, 3) });
  }
  const steps = (u: number) => {
    const p = recordedPath(samples, lens, { ...DEFAULT_CLIP, frames: 49, uniform: u, center: [0, 0, 0] })!.poses;
    return p.slice(1).map((q, i) => angleBetween(p[i].qvec, q.qvec));
  };
  const timed = steps(0), even = steps(1);
  assert.ok(Math.max(...timed) / Math.min(...timed.filter((d) => d > 1e-3)) > 4, "recorded pace is uneven");
  assert.ok(timed.some((d) => d < 1e-3), "the recorded pause holds the camera");
  assert.ok(Math.max(...even) / Math.min(...even) < 1.05, `constant speed (${Math.min(...even)}–${Math.max(...even)})`);
  const half = steps(0.5);
  const held = (d: number[]) => d.filter((x) => x < 1e-3).length;
  assert.ok(Math.max(...half) < Math.max(...timed), "the blend slows the fast part");
  assert.ok(held(half) > 0 && held(half) < held(timed), "and shortens the pause");
});

test("pose still: damping-sized nudges count as still", () => {
  assert.ok(poseStill(lookAt(30, 10, 3), lookAt(30 + 1e-5, 10, 3)));
  assert.ok(!poseStill(lookAt(30, 10, 3), lookAt(30.1, 10, 3)));
});

test("recorded motion: a zoom counts as movement at uniform speed and in the peak speed", () => {
  // Fast zoom from 3 to 0.75 (0.2 s), then a slow 60° turn up close (3 s).
  const samples = [];
  for (let t = 0; t <= 3200; t += 16) {
    const r = t < 200 ? 3 * Math.pow(0.25, t / 200) : 0.75;
    const az = t < 200 ? 0 : (60 * (t - 200)) / 3000;
    samples.push({ t, ...lookAt(az, 10, r) });
  }
  const opts = { ...DEFAULT_CLIP, frames: 97, center: [0, 0, 0] as Vec3 };
  const timed = recordedPath(samples, lens, { ...opts, uniform: 0 })!.poses;
  const even = recordedPath(samples, lens, { ...opts, uniform: 1 })!.poses;
  const zoomFrames = (p: typeof even) => p.filter((q) => Math.hypot(...poseToWorld(q).center) > 0.76).length;
  // ln 4 ≈ 79° of image change vs 60° of turn: the zoom gets over half the clip.
  assert.ok(zoomFrames(timed) < 10, `recorded: zoom in a few frames (${zoomFrames(timed)})`);
  assert.ok(zoomFrames(even) > 45, `uniform: zoom spread out (${zoomFrames(even)})`);
  const fast = peakSpeed(timed, 24, 3, [0, 0, 0]);
  assert.ok(fast > 5 * peakSpeed(timed, 24), `zoom shows in the peak (${fast})`);
  assert.ok(peakSpeed(even, 24, 3, [0, 0, 0]) < fast / 5);
});

test("recorded motion: stabilize removes shake, keeps the ends and the overall move", () => {
  // 90° turn in 3 s at a steady pace, with a hand shake (~9 Hz) in azimuth,
  // elevation and aim.
  const samples = [];
  for (let t = 0; t <= 3000; t += 16) {
    const shake = (f: number, ph: number) => (t > 0 && t < 3000 ? 1.5 * Math.sin((2 * Math.PI * f * t) / 1000 + ph) : 0);
    const cam = poseToWorld(lookAt((90 * t) / 3000 + shake(9, 0), 10 + shake(7, 1), 3)).center;
    samples.push({ t, ...aim(cam, [shake(11, 2) * 0.02, shake(8, 3) * 0.02, 0]) });
  }
  const opts = { ...DEFAULT_CLIP, frames: 97, center: [0, 0, 0] as Vec3 };
  // Jerk: change in step angle from frame to frame.
  const jerk = (stabilize: number) => {
    const p = recordedPath(samples, lens, { ...opts, stabilize })!.poses;
    const steps = p.slice(1).map((q, i) => angleBetween(p[i].qvec, q.qvec));
    return { p, jerk: Math.max(...steps.slice(1).map((d, i) => Math.abs(d - steps[i]))) };
  };
  const raw = jerk(0), smooth = jerk(0.5);
  assert.ok(smooth.jerk < raw.jerk / 5, `shake reduced (${raw.jerk.toFixed(3)} → ${smooth.jerk.toFixed(3)})`);
  assert.ok(angleBetween(smooth.p[0].qvec, raw.p[0].qvec) < 1e-6, "same first frame");
  assert.ok(angleBetween(smooth.p[96].qvec, raw.p[96].qvec) < 1e-6, "same last frame");
  assert.ok(Math.abs(elevation(smooth.p[48]) - 10) < 0.5, `middle on the path (${elevation(smooth.p[48])})`);
});

// Anchor off the photos' center: up at the "head" and a bit to the side.
const PIVOT: Vec3 = [0.3, -0.8, 0.2];
const aimsAt = (p: { qvec: Quat; tvec: Vec3 }, at: Vec3) => dot(forward(p), unit(sub(at, poseToWorld(p).center)));
const axisDist = (p: { qvec: Quat; tvec: Vec3 }, at: Vec3) => {
  const r = sub(poseToWorld(p).center, at);
  return Math.hypot(...sub(r, [UP[0] * dot(r, UP), UP[1] * dot(r, UP), UP[2] * dot(r, UP)]));
};

test("anchor: 360° turn circles the anchor's vertical axis, aiming at it", () => {
  const view = aim(poseToWorld(lookAt(40, 25, 2)).center, PIVOT);
  const p = turnPath(orbit, lens, view, { ...DEFAULT_CLIP, pivot: PIVOT }).poses;
  const d0 = axisDist(p[0], PIVOT);
  for (const q of p) {
    assert.ok(Math.abs(axisDist(q, PIVOT) - d0) < 1e-6, "same distance from the anchor's axis");
    assert.ok(aimsAt(q, PIVOT) > 0.9999, "aims at the anchor");
  }
});

test("anchor: tilted turn circles the anchor", () => {
  const view = aim(poseToWorld(lookAt(40, 45, 2)).center, PIVOT);
  const p = tiltTurnPath(orbit, lens, view, { ...DEFAULT_CLIP, pivot: PIVOT }).poses;
  const mid = p[p.length >> 1];
  assert.ok(aimsAt(mid, PIVOT) > 0.999, `middle aims at the anchor (${aimsAt(mid, PIVOT)})`);
  assert.ok(Math.abs(axisDist(mid, PIVOT) - axisDist(view, PIVOT)) < 1e-6, "at the view's distance from the anchor's axis");
});

test("anchor: presets leave and return to real photos and aim at the anchor in between", () => {
  for (const shape of ["arc", "wave"] as const) {
    const plan = presetPath(orbit, lens, { ...DEFAULT_CLIP, center: 20, span: 60, ramp: shape === "wave" ? 0 : 0.25, elevDeg: -30, shape, pivot: PIVOT });
    const p = plan.poses;
    assert.ok(p[0].anchor && p[p.length - 1].anchor, `${shape}: ends at photos`);
    assert.ok(aimsAt(p[p.length >> 2], PIVOT) > 0.9999, `${shape}: aims at the anchor`);
    for (let i = 1; i < p.length; i++) assert.ok(angleBetween(p[i - 1].qvec, p[i].qvec) < 8, `${shape}: jump at frame ${i}`);
  }
  // Without an anchor nothing changes.
  const a = presetPath(orbit, lens, { ...DEFAULT_CLIP, center: 20, elevDeg: -30 }).poses;
  const b = presetPath(orbit, lens, { ...DEFAULT_CLIP, center: 20, elevDeg: -30, pivot: undefined }).poses;
  assert.deepEqual(a, b);
});
