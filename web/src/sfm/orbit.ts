// Post-SfM orbit corrections, the same as src-tauri/src/scene.rs: flat orbit
// (flatten_orbit) and inferred poses for frames the reconstruction did not
// register (fill_missing).

import { bundleAdjust } from "./ba.ts";
import type { Pose } from "./geometry";
import { cross, dot, expSO3, type Mat3, mul3, mulTVec3, mulVec3, quatFromMat, transpose3, type Vec3 } from "./linalg.ts";
import type { PairMatches } from "./matcher";
import type { SfmModel } from "./reconstruct";

const sub = (a: ArrayLike<number>, b: ArrayLike<number>): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: ArrayLike<number>, b: ArrayLike<number>): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: ArrayLike<number>, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const unit = (a: ArrayLike<number>): Vec3 => scale(a, 1 / Math.max(Math.hypot(a[0], a[1], a[2]), 1e-12));
const centerOf = (p: Pose): Vec3 => scale(mulTVec3(p.R, p.t), -1);
const poseAt = (R: Mat3, c: Vec3): Pose => ({ R, t: scale(mulVec3(R, c), -1) });

function median(v: number[]) {
  if (!v.length) return 0;
  const s = [...v].sort((a, b) => a - b);
  return s[s.length >> 1];
}

/** Where the cameras look: median of the points (like scene.rs: read). */
function target(model: SfmModel, centers: Vec3[]): Vec3 {
  if (!model.points.length) return scale(centers.reduce((a, c) => add(a, c), [0, 0, 0] as Vec3), 1 / centers.length);
  return [0, 1, 2].map((k) => median(model.points.map((p) => p.X[k]))) as Vec3;
}

/** Orbit plane normal (Newell: sum of cross products of consecutive cameras
 *  around the mean), or null if the cameras don't form an arc. */
function orbitNormal(centers: Vec3[]): Vec3 | null {
  const k = centers.length;
  const mean = scale(centers.reduce((a, c) => add(a, c), [0, 0, 0] as Vec3), 1 / k);
  let n: Vec3 = [0, 0, 0];
  for (let i = 1; i < k; i++) n = add(n, cross(sub(centers[i - 1], mean), sub(centers[i], mean)));
  const spread = centers.reduce((s, c) => s + dot(sub(c, mean), sub(c, mean)), 0) / k;
  return Math.hypot(...n) < 1e-3 * spread * k ? null : unit(n);
}

/** Smallest rotation taking unit vector `a` to `b` (Rodrigues). */
function rotationBetween(a: Vec3, b: Vec3): Mat3 {
  const v = cross(a, b);
  const K = Float64Array.from([0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0]);
  const K2 = mul3(K, K);
  const s = 1 / (1 + dot(a, b));
  const M = Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (let i = 0; i < 9; i++) M[i] += K[i] + K2[i] * s;
  return M;
}

/**
 * Forces a flat orbit: projects each camera center onto the orbit plane and
 * rotates the camera so the subject stays at the same pixel; then re-fits only
 * the 3D points (poses and focal fixed). `xy[i]`: keypoints of image i. Returns
 * the largest height deviation removed, as a fraction of the radius, or null
 * if the cameras don't form an arc.
 */
export function flattenOrbit(model: SfmModel, xy: ArrayLike<number>[]): number | null {
  const reg = model.poses.flatMap((p, i) => (p ? [i] : []));
  if (reg.length < 3) return null;
  const centers = reg.map((i) => centerOf(model.poses[i]!));
  const normal = orbitNormal(centers);
  if (!normal) return null;
  const o = target(model, centers);
  const mean = scale(centers.reduce((a, c) => add(a, c), [0, 0, 0] as Vec3), 1 / centers.length);
  let maxH = 0;
  reg.forEach((i, k) => {
    const c = centers[k];
    const h = dot(sub(c, mean), normal);
    maxH = Math.max(maxH, Math.abs(h));
    const c2 = sub(c, scale(normal, h));
    const R = mul3(model.poses[i]!.R, rotationBetween(unit(sub(o, c2)), unit(sub(o, c))));
    model.poses[i] = poseAt(R, c2);
  });

  // Points: BA with only the points varying.
  const obsImg: number[] = [], obsPt: number[] = [], obsUV: number[] = [];
  model.points.forEach((p, j) => {
    for (const [i, k] of p.obs) {
      obsImg.push(i);
      obsPt.push(j);
      obsUV.push(xy[i][2 * k], xy[i][2 * k + 1]);
    }
  });
  const points = Float64Array.from(model.points.flatMap((p) => p.X));
  if (obsImg.length) {
    bundleAdjust({
      f: model.f, cx: model.cx, cy: model.cy,
      R: model.poses.map((p) => p?.R ?? null),
      t: model.poses.map((p) => p?.t ?? null),
      varCams: [], points,
      obsImg: Int32Array.from(obsImg), obsPt: Int32Array.from(obsPt), obsUV: Float64Array.from(obsUV),
      refineFocal: false, huber: 2, iters: 15,
    });
  }
  let errSum = 0, nObs = 0;
  model.points.forEach((p, j) => {
    p.X = [points[3 * j], points[3 * j + 1], points[3 * j + 2]];
    let e = 0;
    for (const [i, k] of p.obs) {
      const pose = model.poses[i]!;
      const c = add(mulVec3(pose.R, p.X), pose.t);
      e += c[2] > 0 ? Math.hypot((model.f * c[0]) / c[2] + model.cx - xy[i][2 * k], (model.f * c[1]) / c[2] + model.cy - xy[i][2 * k + 1]) : 1e3;
    }
    p.error = e / Math.max(p.obs.length, 1);
    errSum += e;
    nObs += p.obs.length;
  });
  model.stats.meanError = errSum / Math.max(nObs, 1);
  const radius = median(centers.map((c) => Math.hypot(...sub(c, o))));
  return maxH / Math.max(radius, 1e-12);
}

/** Rotation of θ about unit axis `n`. */
const about = (n: Vec3, th: number) => expSO3(scale(n, th));

type Quat = [number, number, number, number];

function slerp(a: Quat, b: Quat, u: number): Quat {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const s = d < 0 ? -1 : 1;
  d *= s;
  let wa = 1 - u, wb = u * s;
  if (d < 0.9995) {
    const th = Math.acos(d);
    wa = Math.sin((1 - u) * th) / Math.sin(th);
    wb = (Math.sin(u * th) / Math.sin(th)) * s;
  }
  const q = a.map((v, k) => wa * v + wb * b[k]) as Quat;
  const n = Math.hypot(...q);
  return q.map((v) => v / n) as Quat;
}

function matFromQuat([w, x, y, z]: Quat): Mat3 {
  return Float64Array.from([
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ]);
}

/** Cubic Hermite with non-uniform Catmull-Rom tangents: follows the
 *  neighbours' speed on both sides of the gap. */
function hermite(s: number[], y: number[], a: number, b: number, at: number) {
  const slope = (i: number, j: number) => (y[j] - y[i]) / (s[j] - s[i]);
  const ma = a > 0 ? slope(a - 1, b) : slope(a, b);
  const mb = b < s.length - 1 ? slope(a, b + 1) : slope(a, b);
  const h = s[b] - s[a];
  const u = (at - s[a]) / h;
  const u2 = u * u, u3 = u2 * u;
  return (2 * u3 - 3 * u2 + 1) * y[a] + (u3 - 2 * u2 + u) * h * ma + (-2 * u3 + 3 * u2) * y[b] + (u3 - u2) * h * mb;
}

/** Least-squares slope of y over s. */
function fitSlope(s: number[], y: number[]) {
  const n = s.length;
  const ms = s.reduce((a, v) => a + v, 0) / n, my = y.reduce((a, v) => a + v, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (s[i] - ms) * (y[i] - my);
    den += (s[i] - ms) ** 2;
  }
  return den > 0 ? num / den : 0;
}

/** Orbit angle, radius and height at time `t`: Hermite between the
 *  neighbouring registered frames; before the first or after the last, the
 *  angle is extrapolated with the speed of the 4 end frames, using their mean
 *  radius and height. */
function orbitAt(s: number[], th: number[], rad: number[], hgt: number[], t: number): [number, number, number] {
  const b = s.findIndex((v) => v > t);
  if (b > 0) return [th, rad, hgt].map((y) => hermite(s, y, b - 1, b, t)) as [number, number, number];
  const end = b === 0 ? 0 : s.length - 1;
  const ks = end === 0 ? [0, 1, 2, 3].filter((k) => k < s.length) : [end - 3, end - 2, end - 1, end].filter((k) => k >= 0);
  const mean = (y: number[]) => ks.reduce((a, k) => a + y[k], 0) / ks.length;
  return [th[end] + fitSlope(ks.map((k) => s[k]), ks.map((k) => th[k])) * (t - s[end]), mean(rad), mean(hgt)];
}

/** A frame's 2D (pixels) → 3D correspondences, from its matches with
 *  registered neighbours. */
export interface Correspondences {
  uv: Float64Array;
  X: Float64Array;
  n: number;
}

/** Typical noise of a matched 2D point (pixels): weighs the correspondences
 *  against the trajectory confidence (`refinePose`). */
const KEYPOINT_SIGMA = 0.5;
/** Minimum inliers (≤ 4 px) to accept a refined pose. */
const REFINE_MIN_INLIERS = 15;

function reproj(model: SfmModel, R: Mat3, t: Vec3, c: Correspondences, j: number) {
  const p = add(mulVec3(R, c.X.subarray(3 * j, 3 * j + 3)), t);
  if (p[2] <= 1e-9) return Infinity;
  return Math.hypot((model.f * p[0]) / p[2] + model.cx - c.uv[2 * j], (model.f * p[1]) / p[2] + model.cy - c.uv[2 * j + 1]);
}

/** Solves the 6×6 system `A x = b` (elimination with partial pivoting). */
function solve6(A: number[][], b: number[]): number[] | null {
  for (let col = 0; col < 6; col++) {
    let piv = col;
    for (let r = col + 1; r < 6; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    if (Math.abs(A[piv][col]) < 1e-12) return null;
    [A[col], A[piv]] = [A[piv], A[col]];
    [b[col], b[piv]] = [b[piv], b[col]];
    for (let r = col + 1; r < 6; r++) {
      const f = A[r][col] / A[col][col];
      for (let k = col; k < 6; k++) A[r][k] -= f * A[col][k];
      b[r] -= f * b[col];
    }
  }
  const x = new Array(6).fill(0);
  for (let r = 5; r >= 0; r--) {
    let acc = b[r];
    for (let k = r + 1; k < 6; k++) acc -= A[r][k] * x[k];
    x[r] = acc / A[r][r];
  }
  return x;
}

/**
 * Refines the inferred pose (world→camera rotation `R0`, center `c0`) against
 * the correspondences, starting from it: each round only uses those the
 * current pose projects nearby (threshold from 32 px down to 4 px), in a damped
 * Gauss-Newton with Huber loss, so wrong matches far from the inferred pose
 * don't pull the camera. The inferred center enters as a prior with deviation
 * `sigma`: where the visible points leave a direction poorly constrained, the
 * trajectory wins; where they constrain it well, the points do. Same algorithm
 * as scene.rs: refine_pose.
 */
export function refinePose(
  model: SfmModel,
  R0: Mat3,
  c0: Vec3,
  sigma: number,
  corr: Correspondences,
): { pose: Pose; inliers: number } | null {
  let R = R0;
  let t = scale(mulVec3(R0, c0), -1);
  // Cost in px²: the prior becomes (σ_px/σ)²·|center − c0|².
  const prior = (KEYPOINT_SIGMA / Math.max(sigma, 1e-12)) ** 2;
  const centerAt = (R: Mat3, t: Vec3) => scale(mulTVec3(R, t), -1);
  const all = [...Array(corr.n).keys()];
  for (const thr of [32, 16, 8, 4]) {
    const sel = all.filter((j) => reproj(model, R, t, corr, j) < thr);
    if (sel.length < REFINE_MIN_INLIERS) return null;
    const delta = thr / 2;
    const cost = (R: Mat3, t: Vec3) => {
      const d = sub(centerAt(R, t), c0);
      let sum = prior * dot(d, d);
      for (const j of sel) {
        const e = Math.min(reproj(model, R, t, corr, j), 1e3);
        sum += e <= delta ? e * e : 2 * delta * e - delta * delta;
      }
      return sum;
    };
    let lambda = 1e-3;
    let current = cost(R, t);
    for (let it = 0; it < 100; it++) {
      const H = Array.from({ length: 6 }, () => new Array(6).fill(0));
      const g = new Array(6).fill(0);
      for (const j of sel) {
        const RX = mulVec3(R, corr.X.subarray(3 * j, 3 * j + 3));
        const p = add(RX, t);
        if (p[2] <= 1e-9) continue;
        const iz = 1 / p[2];
        const e = [model.f * p[0] * iz + model.cx - corr.uv[2 * j], model.f * p[1] * iz + model.cy - corr.uv[2 * j + 1]];
        const nrm = Math.hypot(e[0], e[1]);
        const w = nrm <= delta ? 1 : delta / nrm;
        // d(pixel)/d(p) · [−[RX]× | I], with p = exp(δr)·RX + t + δt.
        const jp = [[model.f * iz, 0, -model.f * p[0] * iz * iz], [0, model.f * iz, -model.f * p[1] * iz * iz]];
        const sk = [[0, RX[2], -RX[1]], [-RX[2], 0, RX[0]], [RX[1], -RX[0], 0]];
        const J = [0, 1].map((a) => [
          ...[0, 1, 2].map((b) => jp[a][0] * sk[0][b] + jp[a][1] * sk[1][b] + jp[a][2] * sk[2][b]),
          ...jp[a],
        ]);
        for (let a = 0; a < 6; a++) {
          g[a] += w * (J[0][a] * e[0] + J[1][a] * e[1]);
          for (let b = 0; b < 6; b++) H[a][b] += w * (J[0][a] * J[0][b] + J[1][a] * J[1][b]);
        }
      }
      // Prior: center = −Rᵀt; d(center) = −Rᵀ[t]×·δr − Rᵀ·δt.
      const Rt = transpose3(R);
      const d = sub(centerAt(R, t), c0);
      const tx = Float64Array.from([0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0]);
      const Rtx = mul3(Rt, tx);
      const Jc = [0, 1, 2].map((a) => [-Rtx[3 * a], -Rtx[3 * a + 1], -Rtx[3 * a + 2], -Rt[3 * a], -Rt[3 * a + 1], -Rt[3 * a + 2]]);
      for (let a = 0; a < 6; a++) {
        g[a] += prior * (Jc[0][a] * d[0] + Jc[1][a] * d[1] + Jc[2][a] * d[2]);
        for (let b = 0; b < 6; b++) H[a][b] += prior * (Jc[0][a] * Jc[0][b] + Jc[1][a] * Jc[1][b] + Jc[2][a] * Jc[2][b]);
      }
      const damped = H.map((row, a) => row.map((v, b) => (a === b ? v + lambda * Math.max(v, 1e-9) : v)));
      const step = solve6(damped, g.map((v) => -v));
      if (!step) break;
      const R2 = mul3(expSO3(step.slice(0, 3)), R);
      const t2 = add(t, step.slice(3));
      const next = cost(R2, t2);
      if (next < current) {
        const done = current - next < 1e-9 * Math.max(current, 1e-12);
        [R, t, current] = [R2, t2, next];
        lambda = Math.max(lambda / 10, 1e-9);
        if (done) break;
      } else lambda *= 10;
    }
  }
  const inliers = all.filter((j) => reproj(model, R, t, corr, j) < 4).length;
  return inliers >= REFINE_MIN_INLIERS ? { pose: { R, t }, inliers } : null;
}

/** Angle (radians) of the relative rotation between two orientations. */
function rotationAngle(a: Mat3, b: Mat3) {
  const M = mul3(a, transpose3(b));
  return Math.acos(Math.min(1, Math.max(-1, (M[0] + M[4] + M[8] - 1) / 2)));
}

export interface FillOptions {
  /** Flat orbit: the refined pose is put back onto the orbit plane. */
  flat?: boolean;
  /** Keyed by unregistered frame index; frames without an entry are only inferred. */
  correspondences?: Map<number, Correspondences>;
}

/**
 * Gives poses to frames the SfM did not register, from their neighbours: in
 * orbit coordinates (angle about the axis, radius and height), via cubic
 * Hermite at each frame's timestamp, i.e. following the camera's speed and
 * spacing; orientation relative to the orbit comes from slerp. At the ends, it
 * extrapolates with the angular speed of the last frames. Without a defined
 * arc, centers are interpolated linearly. The inferred pose is then refined
 * against the 3D points the frame sees (`refinePose`), if there are enough
 * correspondences and the result doesn't stray too far from it.
 * `at[i]`: timestamp (or index) of frame i, increasing. Same algorithm as
 * scene.rs: fill_missing.
 */
export function fillMissing(
  model: SfmModel,
  at: number[],
  opts: FillOptions = {},
): { inferred: number[]; refined: [number, number][] } {
  const reg = model.poses.flatMap((p, i) => (p ? [i] : []));
  const missing = model.poses.flatMap((p, i) => (p ? [] : [i]));
  const refined: [number, number][] = [];
  if (reg.length < 3 || !missing.length) return { inferred: [], refined };
  const centers = reg.map((i) => centerOf(model.poses[i]!));
  const s = reg.map((i) => at[i]);
  const n = orbitNormal(centers) ?? [0, 0, 0];
  const linear = !n.some((v) => v !== 0);
  const o = target(model, centers);
  const radius = median(centers.map((c) => Math.hypot(...sub(c, o))));

  // Plane basis: e1 points toward the first camera.
  const d0 = sub(centers[0], o);
  const e1 = unit(sub(d0, scale(n, dot(d0, n))));
  const e2 = cross(n, e1);
  const th: number[] = [], rad: number[] = [], hgt: number[] = [];
  const raw = centers.map((c) => {
    const d = sub(c, o);
    const hh = dot(d, n);
    const p = sub(d, scale(n, hh));
    rad.push(Math.hypot(...p));
    hgt.push(hh);
    return Math.atan2(dot(p, e2), dot(p, e1));
  });
  // Unwrap following the angular speed (median of neighbouring steps): a gap
  // can be more than half a turn (half the orbit unregistered), and taking the
  // shortest way there sends the inferred cameras back over the registered arc.
  const wrap = (a: number) => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));
  const speed = median(raw.slice(1).map((a, k) => wrap(a - raw[k]) / Math.max(s[k + 1] - s[k], 1e-9)));
  raw.forEach((a, k) => {
    const expected = k ? th[k - 1] + speed * (s[k] - s[k - 1]) : a;
    th.push(a + 2 * Math.PI * Math.round((expected - a) / (2 * Math.PI)));
  });
  // "On-orbit" orientation: the camera rotated back to θ = 0.
  const q0: Quat[] = [];
  reg.forEach((i, k) => {
    const R0 = linear ? model.poses[i]!.R : mul3(model.poses[i]!.R, about(n, th[k]));
    let q = quatFromMat(R0) as Quat;
    const prev = q0[q0.length - 1];
    if (prev && prev[0] * q[0] + prev[1] * q[1] + prev[2] * q[2] + prev[3] * q[3] < 0) q = q.map((v) => -v) as Quat;
    q0.push(q);
  });

  // Trajectory confidence, measured on the registered frames themselves: each
  // one predicted without it (in the middle, from neighbours; at the ends,
  // extrapolated).
  const atCenter = ([theta, r, h]: [number, number, number]) =>
    add(o, add(add(scale(e1, r * Math.cos(theta)), scale(e2, r * Math.sin(theta))), scale(n, h)));
  const loo = (k: number) => {
    const drop = (y: number[]) => y.filter((_, j) => j !== k);
    return Math.hypot(...sub(atCenter(orbitAt(drop(s), drop(th), drop(rad), drop(hgt), s[k])), centers[k]));
  };
  const last = reg.length - 1;
  const step = median(s.slice(1).map((v, k) => v - s[k]));
  const [errMid, errEnd] =
    linear || reg.length < 6
      ? [0.05 * radius, 0.05 * radius]
      : [median(Array.from({ length: last - 1 }, (_, k) => loo(k + 1))), Math.max(loo(0), loo(last))];
  // From the error measured with a single frame left out: in the middle it
  // grows with the gap (≈ linear on a smooth orbit); at the ends, with the
  // square of the extrapolated distance.
  const sigmaAt = (t: number) => {
    const b = s.findIndex((v) => v > t);
    const sigma =
      b > 0
        ? errMid * Math.max(1, (s[b] - s[b - 1]) / (2 * step))
        : errEnd * Math.max(1, Math.abs(t - s[b === 0 ? 0 : last]) / step) ** 2;
    return Math.max(sigma, 1e-4 * radius);
  };

  for (const i of missing) {
    const t = at[i];
    const b = s.findIndex((v) => v > t);
    let q: Quat, lin: Vec3;
    if (b > 0) {
      const u = (t - s[b - 1]) / (s[b] - s[b - 1]);
      q = slerp(q0[b - 1], q0[b], u);
      lin = add(centers[b - 1], scale(sub(centers[b], centers[b - 1]), u));
    } else {
      // Before the first or after the last registered frame.
      const end = b === 0 ? 0 : last;
      const [k0, k1] = end === 0 ? [0, Math.min(3, last)] : [Math.max(0, last - 3), last];
      q = q0[end];
      lin = add(centers[end], scale(sub(centers[k1], centers[k0]), (t - s[end]) / (s[k1] - s[k0])));
    }
    const orbit = orbitAt(s, th, rad, hgt, t);
    let R = linear ? matFromQuat(q) : mul3(matFromQuat(q), transpose3(about(n, orbit[0])));
    let c = linear ? lin : atCenter(orbit);
    const corr = opts.correspondences?.get(i);
    const r = corr && refinePose(model, R, c, sigmaAt(t), corr);
    if (r) {
      const c2 = centerOf(r.pose);
      // Too far from the trajectory: bad matches, keep the inferred pose.
      if (Math.hypot(...sub(c2, c)) <= 0.2 * radius && rotationAngle(r.pose.R, R) <= (15 * Math.PI) / 180) {
        [R, c] = [r.pose.R, c2];
        if (opts.flat && !linear) {
          // Back onto the plane, with the subject at the same pixel.
          const flat = sub(c, scale(n, dot(sub(c, o), n) - orbit[2]));
          R = mul3(R, rotationBetween(unit(sub(o, flat)), unit(sub(o, c))));
          c = flat;
        }
        refined.push([i, r.inliers]);
      }
    }
    model.poses[i] = poseAt(R, c);
  }
  return { inferred: missing, refined };
}

/** Correspondences for unregistered frames: their matches with registered
 *  frames, carried over to the 3D points those frames see. A keypoint matched
 *  to different points keeps the most voted one. Same rule as colmap.rs:
 *  correspondences. */
export function correspondences(model: SfmModel, xy: ArrayLike<number>[], matches: PairMatches[]): Map<number, Correspondences> {
  const seen = new Map<number, Map<number, number>>(); // image → keypoint → point
  model.points.forEach((p, pi) => {
    for (const [i, k] of p.obs) {
      let m = seen.get(i);
      if (!m) seen.set(i, (m = new Map()));
      m.set(k, pi);
    }
  });
  const votes = new Map<number, Map<number, Map<number, number>>>();
  for (const m of matches) {
    const aReg = !!model.poses[m.a] && seen.has(m.a), bReg = !!model.poses[m.b] && seen.has(m.b);
    // Exactly one of the two outside the model.
    if (aReg === bReg || (aReg ? model.poses[m.b] : model.poses[m.a])) continue;
    const [missing, km, kr, reg] = aReg ? [m.b, m.ib, m.ia, m.a] : [m.a, m.ia, m.ib, m.b];
    const pts = seen.get(reg)!;
    let v = votes.get(missing);
    if (!v) votes.set(missing, (v = new Map()));
    for (let j = 0; j < km.length; j++) {
      const p = pts.get(kr[j]);
      if (p === undefined) continue;
      let c = v.get(km[j]);
      if (!c) v.set(km[j], (c = new Map()));
      c.set(p, (c.get(p) ?? 0) + 1);
    }
  }
  const out = new Map<number, Correspondences>();
  for (const [i, v] of votes) {
    const uv = new Float64Array(2 * v.size), X = new Float64Array(3 * v.size);
    let n = 0;
    for (const [k, c] of v) {
      const p = [...c].reduce((a, b) => (b[1] > a[1] ? b : a))[0];
      uv[2 * n] = xy[i][2 * k];
      uv[2 * n + 1] = xy[i][2 * k + 1];
      X.set(model.points[p].X, 3 * n);
      n++;
    }
    out.set(i, { uv, X, n });
  }
  return out;
}
