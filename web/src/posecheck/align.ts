// Pose correction of one frame against a render of a model that never saw it
// (the other fold): a 6-DOF Gauss-Newton step that moves the camera so that
//  - contour: the render's silhouette edge lands on the frame's mask edge
//    (edge points lifted to 3D with the render's depth, residual = the mask's
//    signed distance where they project), and
//  - interior: the render's texture lands on the frame's (direct photometric
//    alignment of points inside both silhouettes, coarse to fine).
// Both blocks carry the same total weight. The contour alone fixes doubled
// edges but drags frontal views off their texture; the interior keeps them.

import { choleskySolve, expSO3, mul3, mulVec3, type Mat3 } from "../sfm/linalg.ts";
import { bilinear, blur, erodeCross, halve, signedDistance, sobel } from "./image.ts";

export interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

/** The held-out model seen from the frame's current pose, at the frame's
 *  analysis size: Brush's `renderColor` (RGBA, colour premultiplied) and
 *  `renderDepth` (expected depth). */
export interface ViewRender {
  w: number;
  h: number;
  rgba: Float32Array;
  depth: Float32Array;
}

interface Level {
  w: number;
  h: number;
  gray: Float32Array;
  gx: Float32Array;
  gy: Float32Array;
}

/** The frame, prepared once for all its steps. */
export interface FrameTarget {
  w: number;
  h: number;
  /** Mask (alpha > 0.5), 0/1. */
  mask: Uint8Array;
  /** How far a silhouette edge pixel placed here is from the mask's own edge
   *  pixels (px, > 0 outside the mask, 0 on its inner boundary), and its
   *  gradient. */
  sdf: Float32Array;
  sdfGx: Float32Array;
  sdfGy: Float32Array;
  /** Gray pyramid, finest first. */
  levels: Level[];
}

/** Gray composited over this (the background never enters the interior term). */
const BACKGROUND = 17 / 255;

export interface AlignOptions {
  /** Pyramid levels of the interior term. */
  levels: number;
  /** Gauss-Newton iterations per level. */
  iterations: number;
  /** Robust loss of the contour term, in px at 640 px: Huber width and cutoff. */
  contourHuber: number;
  contourCut: number;
  /** Same for the interior term (px of image motion along the gradient). */
  interiorHuber: number;
  interiorCut: number;
  /** Weight of the interior block relative to the contour's (0 = contour only). */
  interiorWeight: number;
  /** Interior points: one per `grid`² pixels, with a gray gradient ≥ `minGradient` per px. */
  grid: number;
  minGradient: number;
  /** Where the model's edge sits relative to the masks when the poses agree
   *  (px, > 0 outside): short trainings render a soft halo of 1–2 px around
   *  the subject. Subtracted from the contour residuals (see edgeOffset). */
  contourBias: number;
  /** Pull toward the pose of the render, as a fraction of the data's mean
   *  curvature (rotation and translation apart). Keeps a step from sliding
   *  along what the images barely constrain (orbiting a round subject), and
   *  shrinks the well-constrained motion by about this fraction. */
  damping: number;
}

export const DEFAULT_ALIGN: AlignOptions = {
  levels: 3,
  iterations: 4,
  contourHuber: 2,
  contourCut: 15,
  interiorHuber: 1,
  interiorCut: 8,
  interiorWeight: 1,
  grid: 3,
  minGradient: 0.01,
  contourBias: 0,
  damping: 0.05,
};

function pyramid(gray: Float32Array, w: number, h: number, n: number): Level[] {
  const out: Level[] = [];
  let img = blur(gray, w, h);
  for (let l = 0; l < n; l++) {
    const { gx, gy } = sobel(img, w, h);
    out.push({ w, h, gray: img, gx, gy });
    if (l + 1 < n) {
      const next = halve(img, w, h);
      [img, w, h] = [blur(next.img, next.w, next.h), next.w, next.h];
    }
  }
  return out;
}

/** `rgba`: the frame with the mask in alpha (straight, 0–255). */
export function prepareTarget(rgba: ArrayLike<number>, w: number, h: number, levels = DEFAULT_ALIGN.levels): FrameTarget {
  const mask = new Uint8Array(w * h);
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const a = rgba[4 * i + 3] / 255;
    mask[i] = a > 0.5 ? 1 : 0;
    const c = (rgba[4 * i] + rgba[4 * i + 1] + rgba[4 * i + 2]) / (3 * 255);
    gray[i] = c * a + BACKGROUND * (1 - a);
  }
  // The signed distance is −1 on the mask's inner boundary: shifting the
  // inside by 1 makes an aligned edge pixel cost 0 (the plain distance would
  // push every silhouette ~½ px outward).
  const sdf = signedDistance(mask, w, h);
  for (let i = 0; i < w * h; i++) if (sdf[i] < 0) sdf[i] += 1;
  const { gx, gy } = sobel(sdf, w, h);
  return { w, h, mask, sdf, sdfGx: gx, sdfGy: gy, levels: pyramid(gray, w, h, levels) };
}

/** d(projection)/d(ω, v) of camera point p for a left update p' = exp(ω) p + v:
 *  rows u and v, 6 each, into `ju` and `jv`. */
function projJacobian(p: ArrayLike<number>, K: Intrinsics, ju: Float64Array, jv: Float64Array) {
  const [x, y, z] = [p[0], p[1], p[2]];
  const iz = 1 / z, iz2 = iz * iz;
  // du/dp = [fx/z, 0, -fx x/z²]; dp/dω = -[p]×, dp/dv = I
  const a = K.fx * iz, c = -K.fx * x * iz2;
  const b = K.fy * iz, d = -K.fy * y * iz2;
  // -[p]× = [[0, z, -y], [-z, 0, x], [y, -x, 0]]
  ju[0] = c * y;
  ju[1] = a * z - c * x;
  ju[2] = -a * y;
  ju[3] = a;
  ju[4] = 0;
  ju[5] = c;
  jv[0] = -b * z + d * y;
  jv[1] = -d * x;
  jv[2] = b * x;
  jv[3] = 0;
  jv[4] = b;
  jv[5] = d;
}

/** Accumulates w·JᵀJ and w·Jᵀr. */
function accumulate(H: Float64Array, g: Float64Array, J: Float64Array, r: number, w: number) {
  for (let i = 0; i < 6; i++) {
    const wi = w * J[i];
    g[i] += wi * r;
    for (let j = i; j < 6; j++) H[i * 6 + j] += wi * J[j];
  }
}

/** IRLS weight of the robust loss: Huber of width k, nothing beyond `cut`. */
const huber = (r: number, k: number, cut: number) => {
  const a = Math.abs(r);
  return a >= cut ? 0 : a <= k ? 1 : k / a;
};

/** The loss itself (constant beyond `cut`). */
const huberCost = (r: number, k: number, cut: number) => {
  const a = Math.min(Math.abs(r), cut);
  return a <= k ? (a * a) / 2 : k * (a - k / 2);
};

function median(v: Float64Array, n: number) {
  if (!n) return 0;
  const s = v.slice(0, n).sort();
  return n % 2 ? s[n >> 1] : (s[(n >> 1) - 1] + s[n >> 1]) / 2;
}

/** Interquartile mean of the signed distance (px, > 0 outside the mask) of
 *  the silhouette's edge pixels (alpha > 0.5) to the mask's edge, among those
 *  within `cut` px (an edge around a hole or a see-through strand lies deep
 *  inside the mask; the distances come in whole-pixel steps, hence no plain median).
 *  Over many frames, its median is the model's halo (AlignOptions.contourBias). */
export function edgeOffset(alpha: Float32Array, target: FrameTarget, cut = DEFAULT_ALIGN.contourCut * (Math.max(target.w, target.h) / 640)): number {
  const { w, h } = target;
  const rm = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) rm[i] = alpha[i] > 0.5 ? 1 : 0;
  const er = erodeCross(rm, w, h, 1);
  const v: number[] = [];
  for (let i = 0; i < w * h; i++) if (rm[i] && !er[i] && Math.abs(target.sdf[i]) < cut) v.push(target.sdf[i]);
  if (!v.length) return 0;
  v.sort((a, b) => a - b);
  const a = Math.floor(v.length / 4), b = Math.max(a + 1, Math.ceil((3 * v.length) / 4));
  let sum = 0;
  for (let i = a; i < b; i++) sum += v[i];
  return sum / (b - a);
}

export interface StepResult {
  /** Camera-frame update (ω, v): R ← exp(ω) R, t ← exp(ω) t + v. */
  dx: Float64Array;
  /** Points used by each block. */
  contourPoints: number;
  interiorPoints: number;
  /** edgeOffset before the step (px; > 0: the model sticks out). */
  contourOffset: number;
  /** The aligned loss at the render's own pose (mean robust loss of each
   *  block, weighted like the step): how well this pose fits, comparable
   *  between renders of the same frame. */
  cost: number;
}

/** StepResult.cost of the render's pose, without a step. */
export const alignCost = (render: ViewRender, target: FrameTarget, K: Intrinsics, o: AlignOptions = DEFAULT_ALIGN) =>
  alignStep(render, target, K, { ...o, iterations: 0 }).cost;

/** One alignment of the frame to the render (several Gauss-Newton
 *  iterations on a fixed render). */
export function alignStep(render: ViewRender, target: FrameTarget, K: Intrinsics, o: AlignOptions = DEFAULT_ALIGN): StepResult {
  const { w, h } = target;
  if (render.w !== w || render.h !== h) throw new Error("render and frame sizes differ");
  const n = w * h;
  const alpha = new Float32Array(n), gray = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = render.rgba[4 * i + 3];
    alpha[i] = a;
    gray[i] = (render.rgba[4 * i] + render.rgba[4 * i + 1] + render.rgba[4 * i + 2]) / 3 + BACKGROUND * (1 - Math.min(a, 1));
  }
  const rm = new Uint8Array(n);
  for (let i = 0; i < n; i++) rm[i] = alpha[i] > 0.5 ? 1 : 0;
  const scale = Math.max(w, h) / 640;
  const back = (x: number, y: number, d: number, out: Float64Array, k: number) => {
    out[3 * k] = ((x + 0.5 - K.cx) / K.fx) * d;
    out[3 * k + 1] = ((y + 0.5 - K.cy) / K.fy) * d;
    out[3 * k + 2] = d;
  };

  // Contour points: the render's edge pixels, with the depth read 2 px
  // inside (edge pixels mix the surface with the background).
  const er = erodeCross(rm, w, h, 1);
  const { gx: ax, gy: ay } = sobel(alpha, w, h);
  let nc = 0;
  for (let i = 0; i < n; i++) if (rm[i] && !er[i]) nc++;
  const Pc = new Float64Array(3 * nc);
  let kc = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!rm[i] || er[i]) continue;
      const gn = Math.hypot(ax[i], ay[i]) + 1e-6;
      const ix = Math.min(Math.max(Math.round(x + (2 * ax[i]) / gn), 0), w - 1);
      const iy = Math.min(Math.max(Math.round(y + (2 * ay[i]) / gn), 0), h - 1);
      const d = render.depth[iy * w + ix];
      if (!(d > 0)) continue;
      back(x, y, d, Pc, kc++);
    }
  const contourOffset = edgeOffset(alpha, target, o.contourCut * scale);

  // Interior points: well inside the render's silhouette and inside the mask.
  const inner = erodeCross(rm, w, h, 3);
  const { gx: rgx, gy: rgy } = sobel(blur(gray, w, h), w, h);
  const pick: number[] = [];
  for (let y = 0; y < h; y += o.grid)
    for (let x = 0; x < w; x += o.grid) {
      const i = y * w + x;
      if (inner[i] && target.mask[i] && render.depth[i] > 0 && Math.hypot(rgx[i], rgy[i]) >= o.minGradient) pick.push(i);
    }
  const nf = o.interiorWeight > 0 ? pick.length : 0;
  const Pf = new Float64Array(3 * nf);
  for (let k = 0; k < nf; k++) back(pick[k] % w, Math.floor(pick[k] / w), render.depth[pick[k]], Pf, k);
  // Reference intensities on the render's own pyramid.
  const L = Math.min(o.levels, target.levels.length);
  const ref = pyramid(gray, w, h, L);
  const refAt: Float64Array[] = ref.map((lv, l) => {
    const s = 2 ** l;
    const out = new Float64Array(nf);
    for (let k = 0; k < nf; k++) {
      const x = pick[k] % w, y = Math.floor(pick[k] / w);
      out[k] = bilinear(lv.gray, lv.w, lv.h, (x + 0.5) / s - 0.5, (y + 0.5) / s - 0.5);
    }
    return out;
  });

  const x6 = new Float64Array(6);
  /** Damping (see AlignOptions), set from the first iteration's curvature. */
  let prior: number[] | null = null;
  const ju = new Float64Array(6), jv = new Float64Array(6), J = new Float64Array(6);
  const p = new Float64Array(3);
  const resid = new Float64Array(nf), proj = new Float64Array(2 * nf);
  const eps = o.minGradient / 2;
  const kHub = o.contourHuber * scale, kCut = o.contourCut * scale;
  const transform = (P: Float64Array, k: number, R: Mat3) => {
    const q = mulVec3(R, [P[3 * k], P[3 * k + 1], P[3 * k + 2]]);
    p[0] = q[0] + x6[3];
    p[1] = q[1] + x6[4];
    p[2] = q[2] + x6[5];
  };

  // The loss at the render's pose, finest level.
  let cost = 0;
  {
    let c = 0, kn = 0;
    for (let k = 0; k < kc; k++) {
      const P = [Pc[3 * k], Pc[3 * k + 1], Pc[3 * k + 2]];
      const u = (K.fx * P[0]) / P[2] + K.cx, v = (K.fy * P[1]) / P[2] + K.cy;
      if (u < 0 || v < 0 || u > w || v > h) continue;
      c += huberCost(bilinear(target.sdf, w, h, u - 0.5, v - 0.5) - o.contourBias, kHub, kCut);
      kn++;
    }
    if (kn) cost += c / kn;
    const lv = target.levels[0];
    let m = 0;
    const g = new Float64Array(nf);
    for (let k = 0; k < nf; k++) {
      const u = (K.fx * Pf[3 * k]) / Pf[3 * k + 2] + K.cx, v = (K.fy * Pf[3 * k + 1]) / Pf[3 * k + 2] + K.cy;
      if (u < 0 || v < 0 || u > w || v > h) continue;
      const X = u - 0.5, Y = v - 0.5;
      const gx = bilinear(lv.gx, lv.w, lv.h, X, Y), gy = bilinear(lv.gy, lv.w, lv.h, X, Y);
      g[m] = Math.sqrt(gx * gx + gy * gy + eps * eps);
      resid[m++] = bilinear(lv.gray, lv.w, lv.h, X, Y) - refAt[0][k];
    }
    if (m) {
      const bias = median(resid, m);
      let f = 0;
      for (let k = 0; k < m; k++) f += huberCost((resid[k] - bias) / g[k], o.interiorHuber * scale, o.interiorCut * scale);
      cost += (o.interiorWeight * f) / m;
    }
  }

  for (let l = L - 1; l >= 0; l--) {
    const lv = target.levels[l];
    const s = 2 ** l;
    const fHub = o.interiorHuber * scale * s, fCut = o.interiorCut * scale * s;
    for (let it = 0; it < o.iterations; it++) {
      const R = expSO3(x6);
      const Hc = new Float64Array(36), gc = new Float64Array(6);
      const Hf = new Float64Array(36), gf = new Float64Array(6);
      let wc = 0, wf = 0;
      for (let k = 0; k < kc; k++) {
        transform(Pc, k, R);
        if (p[2] <= 0) continue;
        const u = (K.fx * p[0]) / p[2] + K.cx, v = (K.fy * p[1]) / p[2] + K.cy;
        if (u < 0 || v < 0 || u > w || v > h) continue;
        const r = bilinear(target.sdf, w, h, u - 0.5, v - 0.5) - o.contourBias;
        const wk = huber(r, kHub, kCut);
        if (!wk) continue;
        const sx = bilinear(target.sdfGx, w, h, u - 0.5, v - 0.5), sy = bilinear(target.sdfGy, w, h, u - 0.5, v - 0.5);
        projJacobian(p, K, ju, jv);
        for (let j = 0; j < 6; j++) J[j] = sx * ju[j] + sy * jv[j];
        accumulate(Hc, gc, J, r, wk);
        wc += wk;
      }
      // Interior: first the residuals, to remove the median brightness offset.
      let m = 0;
      for (let k = 0; k < nf; k++) {
        transform(Pf, k, R);
        if (p[2] <= 0) continue;
        const u = (K.fx * p[0]) / p[2] + K.cx, v = (K.fy * p[1]) / p[2] + K.cy;
        if (u < 0 || v < 0 || u > w || v > h) continue;
        proj[2 * k] = u;
        proj[2 * k + 1] = v;
        resid[m++] = bilinear(lv.gray, lv.w, lv.h, u / s - 0.5, v / s - 0.5) - refAt[l][k];
      }
      const bias = median(resid, m);
      for (let k = 0; k < nf; k++) {
        transform(Pf, k, R);
        if (p[2] <= 0) continue;
        const u = proj[2 * k], v = proj[2 * k + 1];
        if (u < 0 || v < 0 || u > w || v > h) continue;
        const X = u / s - 0.5, Y = v / s - 0.5;
        const gx = bilinear(lv.gx, lv.w, lv.h, X, Y), gy = bilinear(lv.gy, lv.w, lv.h, X, Y);
        const gn = Math.sqrt(gx * gx + gy * gy + eps * eps);
        // Normal flow in full-size px: the image motion along the gradient
        // that explains the residual.
        const r = ((bilinear(lv.gray, lv.w, lv.h, X, Y) - refAt[l][k] - bias) / gn) * s;
        const wk = huber(r, fHub, fCut);
        if (!wk) continue;
        projJacobian(p, K, ju, jv);
        for (let j = 0; j < 6; j++) J[j] = (gx * ju[j] + gy * jv[j]) / gn;
        accumulate(Hf, gf, J, r, wk);
        wf += wk;
      }
      if (!wc && !wf) break;
      const H = new Float64Array(36), g = new Float64Array(6);
      const sc = wc ? 1 / wc : 0, sf = wf ? o.interiorWeight / wf : 0;
      for (let i = 0; i < 36; i++) H[i] = sc * Hc[i] + sf * Hf[i];
      if (!prior) {
        const r = (H[0] + H[7] + H[14]) / 3, t = (H[21] + H[28] + H[35]) / 3;
        prior = [r, r, r, t, t, t].map((x) => o.damping * x);
      }
      for (let i = 0; i < 6; i++) g[i] = sc * gc[i] + sf * gf[i] + prior[i] * x6[i];
      for (let i = 0; i < 6; i++) H[i * 6 + i] += prior[i];
      for (let i = 0; i < 6; i++) for (let j = 0; j < i; j++) H[i * 6 + j] = H[j * 6 + i];
      for (let i = 0; i < 6; i++) H[i * 6 + i] += 1e-9 + 1e-6 * H[i * 6 + i];
      const dx = choleskySolve(H, g, 6);
      if (!dx) break;
      let big = 0;
      for (let i = 0; i < 6; i++) {
        x6[i] -= dx[i];
        big = Math.max(big, Math.abs(dx[i]));
      }
      if (big < 1e-7) break;
    }
  }
  return { dx: x6, contourPoints: kc, interiorPoints: nf, contourOffset, cost };
}

/** Which of the poses a frame went through to keep, by their `cost`s (the
 *  first is where it started): the cheapest, if it beats the start by the
 *  relative `margin`. A frame that already fits stays put instead of
 *  wandering along what the screens barely see. */
export function choosePose(costs: readonly number[], margin: number): number {
  let best = 0;
  for (let i = 1; i < costs.length; i++) if (costs[i] < costs[best]) best = i;
  return costs[best] < costs[0] * (1 - margin) ? best : 0;
}

/** Angle between two rotations (radians). */
export function rotationAngle(A: Mat3, B: Mat3): number {
  let tr = 0;
  for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) tr += A[i * 3 + k] * B[i * 3 + k]; // trace(A Bᵀ)
  return Math.acos(Math.min(1, Math.max(-1, (tr - 1) / 2)));
}

/** World → camera pose after the update `dx` (see StepResult). */
export function applyStep(R: Mat3, t: ArrayLike<number>, dx: ArrayLike<number>): { R: Mat3; t: [number, number, number] } {
  const dR = expSO3([dx[0], dx[1], dx[2]]);
  const t2 = mulVec3(dR, t);
  return { R: mul3(dR, R), t: [t2[0] + dx[3], t2[1] + dx[4], t2[2] + dx[5]] };
}
