// Geometria de múltiplas vistas: matriz fundamental (RANSAC), decomposição da
// essencial, triangulação e PnP. Coordenadas "normalizadas" são
// ((u - cx) / f, (v - cy) / f).

import {
  det3,
  expSO3,
  mat3,
  type Mat3,
  mul3,
  mulVec3,
  nearestRotation,
  nullVector,
  svd3,
  symEig,
  transpose3,
  type Vec3,
} from "./linalg";

export interface Pose {
  R: Mat3; // mundo → câmera
  t: Vec3;
}

/** PRNG determinístico (mulberry32): RANSAC reprodutível. */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sample(n: number, k: number, rand: () => number): number[] {
  const out: number[] = [];
  while (out.length < k) {
    const i = Math.floor(rand() * n);
    if (!out.includes(i)) out.push(i);
  }
  return out;
}

/** Normalização de Hartley: centro na origem, distância média √2. */
function hartley(p: Float64Array, n: number): Mat3 {
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) {
    mx += p[2 * i];
    my += p[2 * i + 1];
  }
  mx /= n;
  my /= n;
  let d = 0;
  for (let i = 0; i < n; i++) d += Math.hypot(p[2 * i] - mx, p[2 * i + 1] - my);
  const s = Math.SQRT2 / Math.max(d / n, 1e-12);
  return mat3(s, 0, -s * mx, 0, s, -s * my, 0, 0, 1);
}

/** F (pixels) pelo algoritmo de 8 pontos normalizado, com posto 2. */
function eightPoint(p1: Float64Array, p2: Float64Array, idx: ArrayLike<number>, T1: Mat3, T2: Mat3): Mat3 {
  const m = idx.length;
  const A = new Float64Array(m * 9);
  for (let r = 0; r < m; r++) {
    const i = idx[r];
    const x1 = T1[0] * p1[2 * i] + T1[2], y1 = T1[4] * p1[2 * i + 1] + T1[5];
    const x2 = T2[0] * p2[2 * i] + T2[2], y2 = T2[4] * p2[2 * i + 1] + T2[5];
    A.set([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1], r * 9);
  }
  const f = nullVector(A, m, 9);
  let F = Float64Array.from(f) as Mat3;
  // posto 2: F ← F − (F v₃) v₃ᵀ, com v₃ o menor autovetor de FᵀF
  const { V } = symEig(mul3(transpose3(F), F), 3);
  const v3: Vec3 = [V[0], V[3], V[6]];
  const Fv = mulVec3(F, v3);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) F[i * 3 + j] -= Fv[i] * v3[j];
  F = mul3(mul3(transpose3(T2), F), T1);
  return F;
}

function sampson(F: Mat3, x1: number, y1: number, x2: number, y2: number) {
  const a0 = F[0] * x1 + F[1] * y1 + F[2];
  const a1 = F[3] * x1 + F[4] * y1 + F[5];
  const a2 = F[6] * x1 + F[7] * y1 + F[8];
  const b0 = F[0] * x2 + F[3] * y2 + F[6];
  const b1 = F[1] * x2 + F[4] * y2 + F[7];
  const e = x2 * a0 + y2 * a1 + a2;
  return (e * e) / (a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1 + 1e-30);
}

/** Matriz fundamental robusta (RANSAC + reajuste nos inliers). Limiar em pixels. */
export function ransacFundamental(
  p1: Float64Array,
  p2: Float64Array,
  n: number,
  thresholdPx = 1.5,
  seed = 1,
): { F: Mat3; inliers: Uint8Array; count: number } | null {
  if (n < 12) return null;
  const T1 = hartley(p1, n), T2 = hartley(p2, n);
  const thr2 = thresholdPx * thresholdPx;
  const rand = rng(seed);
  let best: Mat3 | null = null;
  let bestCount = 0;
  let maxIter = 2000;
  const score = (F: Mat3) => {
    let c = 0;
    for (let i = 0; i < n; i++) if (sampson(F, p1[2 * i], p1[2 * i + 1], p2[2 * i], p2[2 * i + 1]) < thr2) c++;
    return c;
  };
  for (let it = 0; it < maxIter; it++) {
    const F = eightPoint(p1, p2, sample(n, 8, rand), T1, T2);
    const c = score(F);
    if (c > bestCount) {
      bestCount = c;
      best = F;
      const w = c / n;
      maxIter = Math.min(2000, Math.ceil(Math.log(0.001) / Math.log(Math.max(1 - w ** 8, 1e-12))));
    }
  }
  if (!best) return null;
  // reajuste iterativo nos inliers
  let F = best;
  for (let k = 0; k < 3; k++) {
    const idx: number[] = [];
    for (let i = 0; i < n; i++) if (sampson(F, p1[2 * i], p1[2 * i + 1], p2[2 * i], p2[2 * i + 1]) < thr2) idx.push(i);
    if (idx.length < 8) break;
    const Fr = eightPoint(p1, p2, idx, T1, T2);
    if (score(Fr) >= idx.length) F = Fr;
    else break;
  }
  const inliers = new Uint8Array(n);
  let count = 0;
  for (let i = 0; i < n; i++)
    if (sampson(F, p1[2 * i], p1[2 * i + 1], p2[2 * i], p2[2 * i + 1]) < thr2) {
      inliers[i] = 1;
      count++;
    }
  return { F, inliers, count };
}

/** Ponto 3D a partir de ≥2 observações normalizadas (DLT). */
export function triangulate(poses: Pose[], xs: ArrayLike<number>): Vec3 | null {
  const m = poses.length;
  const A = new Float64Array(m * 2 * 4);
  for (let k = 0; k < m; k++) {
    const { R, t } = poses[k];
    const x = xs[2 * k], y = xs[2 * k + 1];
    for (let j = 0; j < 3; j++) {
      A[(2 * k) * 4 + j] = x * R[6 + j] - R[j];
      A[(2 * k + 1) * 4 + j] = y * R[6 + j] - R[3 + j];
    }
    A[(2 * k) * 4 + 3] = x * t[2] - t[0];
    A[(2 * k + 1) * 4 + 3] = y * t[2] - t[1];
  }
  const X = nullVector(A, m * 2, 4);
  if (Math.abs(X[3]) < 1e-12) return null;
  return [X[0] / X[3], X[1] / X[3], X[2] / X[3]];
}

export const toCam = (p: Pose, X: ArrayLike<number>): Vec3 => {
  const c = mulVec3(p.R, X);
  return [c[0] + p.t[0], c[1] + p.t[1], c[2] + p.t[2]];
};

export const center = (p: Pose): Vec3 => {
  const Rt = transpose3(p.R);
  const c = mulVec3(Rt, p.t);
  return [-c[0], -c[1], -c[2]];
};

/**
 * Pose relativa da câmera 2 a partir de F e da focal: as 4 soluções de E e a
 * que deixa mais pontos na frente das duas câmeras.
 */
export function relativePose(
  F: Mat3,
  f: number,
  cx: number,
  cy: number,
  n1: Float64Array,
  n2: Float64Array,
  idx: number[],
): { pose: Pose; good: number } {
  const K = mat3(f, 0, cx, 0, f, cy, 0, 0, 1);
  const E = mul3(mul3(transpose3(K), F), K);
  const { U, V } = svd3(E);
  const fix = (M: Mat3) => {
    if (det3(M) < 0) for (let i = 0; i < 3; i++) M[i * 3 + 2] = -M[i * 3 + 2];
  };
  fix(U);
  fix(V);
  const W = mat3(0, -1, 0, 1, 0, 0, 0, 0, 1);
  const Vt = transpose3(V);
  const R1 = mul3(mul3(U, W), Vt);
  const R2 = mul3(mul3(U, transpose3(W)), Vt);
  const u3: Vec3 = [U[2], U[5], U[8]];
  const id: Pose = { R: mat3(1, 0, 0, 0, 1, 0, 0, 0, 1), t: [0, 0, 0] };
  let best = { pose: id, good: -1 };
  for (const R of [R1, R2])
    for (const s of [1, -1]) {
      const pose: Pose = { R, t: [s * u3[0], s * u3[1], s * u3[2]] };
      let good = 0;
      for (const i of idx) {
        const X = triangulate([id, pose], [n1[2 * i], n1[2 * i + 1], n2[2 * i], n2[2 * i + 1]]);
        if (X && X[2] > 0 && toCam(pose, X)[2] > 0) good++;
      }
      if (good > best.good) best = { pose, good };
    }
  return best;
}

/** Ângulo (graus) entre os raios de duas câmeras até X. */
export function triAngle(c1: ArrayLike<number>, c2: ArrayLike<number>, X: ArrayLike<number>) {
  const a: Vec3 = [c1[0] - X[0], c1[1] - X[1], c1[2] - X[2]];
  const b: Vec3 = [c2[0] - X[0], c2[1] - X[1], c2[2] - X[2]];
  const c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (Math.hypot(...a) * Math.hypot(...b) + 1e-30);
  return (Math.acos(Math.min(1, Math.max(-1, c))) * 180) / Math.PI;
}

/** Refino Gauss-Newton (Huber) de uma pose a partir de 2D–3D normalizados. */
export function refinePose(
  pose: Pose,
  X: Float64Array,
  x: Float64Array,
  n: number,
  huber: number,
  iters = 10,
): Pose {
  let R = pose.R, t: Vec3 = [...pose.t];
  for (let it = 0; it < iters; it++) {
    const H = new Float64Array(36), g = new Float64Array(6);
    for (let i = 0; i < n; i++) {
      const Xw: Vec3 = [X[3 * i], X[3 * i + 1], X[3 * i + 2]];
      const RX = mulVec3(R, Xw);
      const c: Vec3 = [RX[0] + t[0], RX[1] + t[1], RX[2] + t[2]];
      if (c[2] <= 1e-9) continue;
      const iz = 1 / c[2];
      const rx = c[0] * iz - x[2 * i], ry = c[1] * iz - x[2 * i + 1];
      const e = Math.hypot(rx, ry);
      const w = e <= huber ? 1 : huber / e;
      // d(proj)/d(c) e d(c)/d(δr) = −[RX]×, d(c)/d(δt) = I
      const P = [iz, 0, -c[0] * iz * iz, 0, iz, -c[1] * iz * iz];
      const Sx = [0, RX[2], -RX[1], -RX[2], 0, RX[0], RX[1], -RX[0], 0]; // −[RX]×
      const J = new Float64Array(12);
      for (let r = 0; r < 2; r++) {
        for (let k = 0; k < 3; k++) {
          J[r * 6 + k] = P[r * 3] * Sx[k] + P[r * 3 + 1] * Sx[3 + k] + P[r * 3 + 2] * Sx[6 + k];
          J[r * 6 + 3 + k] = P[r * 3 + k];
        }
      }
      const res = [rx, ry];
      for (let r = 0; r < 2; r++)
        for (let a = 0; a < 6; a++) {
          g[a] += w * J[r * 6 + a] * res[r];
          for (let b = 0; b < 6; b++) H[a * 6 + b] += w * J[r * 6 + a] * J[r * 6 + b];
        }
    }
    for (let a = 0; a < 6; a++) H[a * 6 + a] *= 1 + 1e-6;
    // resolve H δ = −g (6×6 via eliminação)
    const d = solveDense(H, g.map((v) => -v), 6);
    if (!d) break;
    R = mul3(expSO3([d[0], d[1], d[2]]), R);
    t = [t[0] + d[3], t[1] + d[4], t[2] + d[5]];
    if (Math.hypot(...d) < 1e-10) break;
  }
  return { R, t };
}

function solveDense(A: Float64Array, b: Float64Array, n: number): Float64Array | null {
  const M = Float64Array.from(A), y = Float64Array.from(b);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r * n + c]) > Math.abs(M[p * n + c])) p = r;
    if (Math.abs(M[p * n + c]) < 1e-300) return null;
    if (p !== c) {
      for (let k = 0; k < n; k++) [M[c * n + k], M[p * n + k]] = [M[p * n + k], M[c * n + k]];
      [y[c], y[p]] = [y[p], y[c]];
    }
    for (let r = c + 1; r < n; r++) {
      const f = M[r * n + c] / M[c * n + c];
      for (let k = c; k < n; k++) M[r * n + k] -= f * M[c * n + k];
      y[r] -= f * y[c];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = y[r];
    for (let k = r + 1; k < n; k++) s -= M[r * n + k] * y[k];
    y[r] = s / M[r * n + r];
  }
  return y;
}

/** Pose por DLT de 6 pontos (normalizados), com os pontos 3D recentrados. */
function dltPose(X: Float64Array, x: Float64Array, idx: number[]): Pose | null {
  const m = idx.length;
  let cx = 0, cy = 0, cz = 0;
  for (const i of idx) {
    cx += X[3 * i];
    cy += X[3 * i + 1];
    cz += X[3 * i + 2];
  }
  cx /= m;
  cy /= m;
  cz /= m;
  let s = 0;
  for (const i of idx) s += Math.hypot(X[3 * i] - cx, X[3 * i + 1] - cy, X[3 * i + 2] - cz);
  s = m / Math.max(s, 1e-12);
  const A = new Float64Array(m * 2 * 12);
  idx.forEach((i, r) => {
    const Xn = [(X[3 * i] - cx) * s, (X[3 * i + 1] - cy) * s, (X[3 * i + 2] - cz) * s, 1];
    const u = x[2 * i], v = x[2 * i + 1];
    for (let k = 0; k < 4; k++) {
      A[(2 * r) * 12 + k] = Xn[k];
      A[(2 * r) * 12 + 8 + k] = -u * Xn[k];
      A[(2 * r + 1) * 12 + 4 + k] = Xn[k];
      A[(2 * r + 1) * 12 + 8 + k] = -v * Xn[k];
    }
  });
  const p = nullVector(A, m * 2, 12);
  let M = mat3(p[0], p[1], p[2], p[4], p[5], p[6], p[8], p[9], p[10]);
  let p4: Vec3 = [p[3], p[7], p[11]];
  if (det3(M) < 0) {
    M = M.map((v) => -v) as Mat3;
    p4 = [-p4[0], -p4[1], -p4[2]];
  }
  const { S } = svd3(M);
  const scale = (S[0] + S[1] + S[2]) / 3;
  if (!(scale > 1e-12)) return null;
  const R = nearestRotation(M);
  // x ~ M (X−c)s + p4 = λ(R X + t), com M = λR/s: λ = scale·s e t = p4/λ − R c
  const Rc = mulVec3(R, [cx, cy, cz]);
  const t: Vec3 = [p4[0] / scale / s - Rc[0], p4[1] / scale / s - Rc[1], p4[2] / scale / s - Rc[2]];
  return { R, t };
}

function inliersOf(pose: Pose, X: Float64Array, x: Float64Array, n: number, thr: number) {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = toCam(pose, [X[3 * i], X[3 * i + 1], X[3 * i + 2]]);
    if (c[2] <= 0) continue;
    if (Math.hypot(c[0] / c[2] - x[2 * i], c[1] / c[2] - x[2 * i + 1]) < thr) out.push(i);
  }
  return out;
}

/**
 * Registra uma câmera a partir de correspondências 2D–3D (normalizadas).
 * Tenta refinar a pose inicial (vídeo: a do vizinho) e um RANSAC por DLT;
 * fica com a que tiver mais inliers.
 */
export function solvePnP(
  X: Float64Array,
  x: Float64Array,
  n: number,
  thr: number,
  init: Pose | null,
  seed = 1,
): { pose: Pose; inliers: number[] } | null {
  if (n < 6) return null;
  const cands: Pose[] = [];
  if (init) cands.push(refinePose(init, X, x, n, thr * 2, 15));
  const rand = rng(seed);
  let best: { pose: Pose; inliers: number[] } | null = null;
  const consider = (p: Pose) => {
    const inl = inliersOf(p, X, x, n, thr);
    if (!best || inl.length > best.inliers.length) best = { pose: p, inliers: inl };
  };
  cands.forEach(consider);
  let maxIter = 500;
  for (let it = 0; it < maxIter; it++) {
    const p = dltPose(X, x, sample(n, 6, rand));
    if (!p) continue;
    const inl = inliersOf(p, X, x, n, thr);
    if (!best || inl.length > (best as { inliers: number[] }).inliers.length) {
      best = { pose: p, inliers: inl };
      const w = inl.length / n;
      maxIter = Math.min(500, Math.ceil(Math.log(0.001) / Math.log(Math.max(1 - w ** 6, 1e-12))));
    }
  }
  if (!best) return null;
  const b = best as { pose: Pose; inliers: number[] };
  // refino final nos inliers
  const Xi = new Float64Array(b.inliers.length * 3), xi = new Float64Array(b.inliers.length * 2);
  b.inliers.forEach((i, k) => {
    Xi.set(X.subarray(3 * i, 3 * i + 3), 3 * k);
    xi.set(x.subarray(2 * i, 2 * i + 2), 2 * k);
  });
  const pose = refinePose(b.pose, Xi, xi, b.inliers.length, thr, 10);
  return { pose, inliers: inliersOf(pose, X, x, n, thr) };
}

