// Bundle adjustment: Levenberg–Marquardt with a Schur complement.
// Shared SIMPLE_PINHOLE camera (focal f, fixed principal point), world→camera
// poses with left update R ← exp(δr) R, t ← t + δt.
// Huber loss (IRLS) in pixels.

import { choleskySolve, expSO3, inv3, type Mat3, mul3, type Vec3 } from "./linalg.ts";

export interface BAProblem {
  f: number;
  cx: number;
  cy: number;
  R: (Mat3 | null)[];
  t: (Vec3 | null)[];
  /** Images whose poses vary; the other observed ones stay fixed. */
  varCams: number[];
  /** 3 per point; all observed points vary (modified in place). */
  points: Float64Array;
  obsImg: Int32Array;
  obsPt: Int32Array;
  obsUV: Float64Array;
  refineFocal: boolean;
  huber: number;
  iters: number;
}

export interface BAResult {
  f: number;
  rmsBefore: number;
  rmsAfter: number;
  iterations: number;
}

function huberCost(e2: number, d: number) {
  const e = Math.sqrt(e2);
  return e <= d ? e2 : 2 * d * e - d * d;
}

export function bundleAdjust(p: BAProblem): BAResult {
  const nObs = p.obsImg.length;
  const camIdx = new Map<number, number>();
  p.varCams.forEach((img, k) => camIdx.set(img, k));
  const nc = p.varCams.length;
  const nC = 6 * nc + (p.refineFocal ? 1 : 0);
  const fIdx = 6 * nc;

  // variable points and their observations
  const ptIdx = new Map<number, number>();
  const ptList: number[] = [];
  for (let o = 0; o < nObs; o++) {
    const j = p.obsPt[o];
    if (!ptIdx.has(j)) {
      ptIdx.set(j, ptList.length);
      ptList.push(j);
    }
  }
  const np = ptList.length;
  const obsByPt: number[][] = Array.from({ length: np }, () => []);
  for (let o = 0; o < nObs; o++) obsByPt[ptIdx.get(p.obsPt[o])!].push(o);

  const evalCost = (R: (Mat3 | null)[], t: (Vec3 | null)[], f: number, X: Float64Array) => {
    let cost = 0, sq = 0;
    for (let o = 0; o < nObs; o++) {
      const i = p.obsImg[o], j = p.obsPt[o];
      const Ri = R[i]!, ti = t[i]!;
      const x = X[3 * j], y = X[3 * j + 1], z = X[3 * j + 2];
      const cz = Ri[6] * x + Ri[7] * y + Ri[8] * z + ti[2];
      if (cz <= 1e-9) {
        cost += p.huber * 1e3;
        sq += 1e6;
        continue;
      }
      const cxv = Ri[0] * x + Ri[1] * y + Ri[2] * z + ti[0];
      const cyv = Ri[3] * x + Ri[4] * y + Ri[5] * z + ti[1];
      const ex = (f * cxv) / cz + p.cx - p.obsUV[2 * o];
      const ey = (f * cyv) / cz + p.cy - p.obsUV[2 * o + 1];
      const e2 = ex * ex + ey * ey;
      cost += huberCost(e2, p.huber);
      sq += e2;
    }
    return { cost, rms: Math.sqrt(sq / Math.max(nObs, 1)) };
  };

  let R = p.R.slice(), t = p.t.slice(), f = p.f;
  let X = p.points;
  let cur = evalCost(R, t, f, X);
  const rmsBefore = cur.rms;
  let lambda = 1e-4;
  let it = 0;

  // per-observation buffers
  const Jc = new Float64Array(nObs * 12); // 2×6
  const Jf = new Float64Array(nObs * 2);
  const Jp = new Float64Array(nObs * 6); // 2×3
  const res = new Float64Array(nObs * 2);
  const wts = new Float64Array(nObs);

  for (; it < p.iters; it++) {
    // --- Jacobians ---
    for (let o = 0; o < nObs; o++) {
      const i = p.obsImg[o], j = p.obsPt[o];
      const Ri = R[i]!, ti = t[i]!;
      const x = X[3 * j], y = X[3 * j + 1], z = X[3 * j + 2];
      const rx = Ri[0] * x + Ri[1] * y + Ri[2] * z;
      const ry = Ri[3] * x + Ri[4] * y + Ri[5] * z;
      const rz = Ri[6] * x + Ri[7] * y + Ri[8] * z;
      const c0 = rx + ti[0], c1 = ry + ti[1], c2 = rz + ti[2];
      if (c2 <= 1e-9) {
        wts[o] = 0;
        continue;
      }
      const iz = 1 / c2;
      const ex = f * c0 * iz + p.cx - p.obsUV[2 * o];
      const ey = f * c1 * iz + p.cy - p.obsUV[2 * o + 1];
      res[2 * o] = ex;
      res[2 * o + 1] = ey;
      const e = Math.hypot(ex, ey);
      wts[o] = e <= p.huber ? 1 : p.huber / e;
      // P = d(res)/d(c)
      const P00 = f * iz, P02 = -f * c0 * iz * iz, P11 = f * iz, P12 = -f * c1 * iz * iz;
      // d(res)/d(δr) = P · (−[RX]×);  −[RX]× = [[0, rz, −ry], [−rz, 0, rx], [ry, −rx, 0]]
      const b = o * 12;
      Jc[b + 0] = P02 * ry;
      Jc[b + 1] = P00 * rz - P02 * rx;
      Jc[b + 2] = -P00 * ry;
      Jc[b + 3] = P00;
      Jc[b + 4] = 0;
      Jc[b + 5] = P02;
      Jc[b + 6] = -P11 * rz + P12 * ry;
      Jc[b + 7] = -P12 * rx;
      Jc[b + 8] = P11 * rx;
      Jc[b + 9] = 0;
      Jc[b + 10] = P11;
      Jc[b + 11] = P12;
      Jf[2 * o] = c0 * iz;
      Jf[2 * o + 1] = c1 * iz;
      // d(res)/d(X) = P · R
      const q = o * 6;
      Jp[q + 0] = P00 * Ri[0] + P02 * Ri[6];
      Jp[q + 1] = P00 * Ri[1] + P02 * Ri[7];
      Jp[q + 2] = P00 * Ri[2] + P02 * Ri[8];
      Jp[q + 3] = P11 * Ri[3] + P12 * Ri[6];
      Jp[q + 4] = P11 * Ri[4] + P12 * Ri[7];
      Jp[q + 5] = P11 * Ri[5] + P12 * Ri[8];
    }

    // --- normal equations ---
    const U = new Float64Array(nC * nC);
    const gc = new Float64Array(nC);
    const V = new Float64Array(np * 9);
    const gp = new Float64Array(np * 3);
    // W per observation: (6 camera rows + focal row) × 3
    const Wc = new Float64Array(nObs * 18);
    const Wf = new Float64Array(nObs * 3);
    for (let o = 0; o < nObs; o++) {
      const w = wts[o];
      if (w === 0) continue;
      const k = camIdx.get(p.obsImg[o]);
      const jp = ptIdx.get(p.obsPt[o])!;
      const r0 = res[2 * o], r1 = res[2 * o + 1];
      const q = o * 6;
      for (let a = 0; a < 3; a++) {
        gp[jp * 3 + a] += w * (Jp[q + a] * r0 + Jp[q + 3 + a] * r1);
        for (let c = 0; c < 3; c++) V[jp * 9 + a * 3 + c] += w * (Jp[q + a] * Jp[q + c] + Jp[q + 3 + a] * Jp[q + 3 + c]);
      }
      const b = o * 12;
      if (k !== undefined) {
        const base = 6 * k;
        for (let a = 0; a < 6; a++) {
          const ja0 = Jc[b + a], ja1 = Jc[b + 6 + a];
          gc[base + a] += w * (ja0 * r0 + ja1 * r1);
          for (let c = 0; c < 6; c++) U[(base + a) * nC + base + c] += w * (ja0 * Jc[b + c] + ja1 * Jc[b + 6 + c]);
          for (let c = 0; c < 3; c++) Wc[o * 18 + a * 3 + c] = w * (ja0 * Jp[q + c] + ja1 * Jp[q + 3 + c]);
          if (p.refineFocal) {
            const v = w * (ja0 * Jf[2 * o] + ja1 * Jf[2 * o + 1]);
            U[(base + a) * nC + fIdx] += v;
            U[fIdx * nC + base + a] += v;
          }
        }
      }
      if (p.refineFocal) {
        const f0 = Jf[2 * o], f1 = Jf[2 * o + 1];
        gc[fIdx] += w * (f0 * r0 + f1 * r1);
        U[fIdx * nC + fIdx] += w * (f0 * f0 + f1 * f1);
        for (let c = 0; c < 3; c++) Wf[o * 3 + c] = w * (f0 * Jp[q + c] + f1 * Jp[q + 3 + c]);
      }
    }

    // --- LM step (try, adjust λ) ---
    let accepted = false;
    for (let tries = 0; tries < 8 && !accepted; tries++) {
      const S = Float64Array.from(U);
      for (let a = 0; a < nC; a++) S[a * nC + a] = U[a * nC + a] * (1 + lambda) + 1e-9;
      const bvec = new Float64Array(nC);
      for (let a = 0; a < nC; a++) bvec[a] = -gc[a];
      const Vinv: (Mat3 | null)[] = new Array(np);
      for (let jp = 0; jp < np; jp++) {
        const Vj = V.slice(jp * 9, jp * 9 + 9);
        for (let a = 0; a < 3; a++) Vj[a * 4] = Vj[a * 4] * (1 + lambda) + 1e-9;
        const Vi = inv3(Vj);
        Vinv[jp] = Vi;
        if (!Vi) continue;
        const obs = obsByPt[jp];
        // Y_o = W_o V⁻¹ (7×3 per observation: 6 camera + 1 focal)
        const rows: { row: number; Y: Float64Array; W: Float64Array }[] = [];
        for (const o of obs) {
          if (wts[o] === 0) continue;
          const k = camIdx.get(p.obsImg[o]);
          if (k !== undefined) {
            const Wb = Wc.subarray(o * 18, o * 18 + 18);
            const Y = new Float64Array(18);
            for (let a = 0; a < 6; a++)
              for (let c = 0; c < 3; c++)
                Y[a * 3 + c] = Wb[a * 3] * Vi[c] + Wb[a * 3 + 1] * Vi[3 + c] + Wb[a * 3 + 2] * Vi[6 + c];
            rows.push({ row: 6 * k, Y, W: Wb });
          }
        }
        // focal row: sum of the point's contributions
        let Wfj: Float64Array | null = null;
        let Yf: Float64Array | null = null;
        if (p.refineFocal) {
          Wfj = new Float64Array(3);
          for (const o of obs) for (let c = 0; c < 3; c++) Wfj[c] += Wf[o * 3 + c];
          Yf = new Float64Array(3);
          for (let c = 0; c < 3; c++) Yf[c] = Wfj[0] * Vi[c] + Wfj[1] * Vi[3 + c] + Wfj[2] * Vi[6 + c];
        }
        const g = gp.subarray(jp * 3, jp * 3 + 3);
        // S −= Y Wᵀ ;  b += Y g  (since b = −gc + W V⁻¹ gp)
        // multiple observations of the same point by the same camera don't occur (tracks are unique per image)
        for (const A of rows) {
          for (let a = 0; a < 6; a++) bvec[A.row + a] += A.Y[a * 3] * g[0] + A.Y[a * 3 + 1] * g[1] + A.Y[a * 3 + 2] * g[2];
          for (const B of rows) {
            for (let a = 0; a < 6; a++)
              for (let c = 0; c < 6; c++)
                S[(A.row + a) * nC + B.row + c] -=
                  A.Y[a * 3] * B.W[c * 3] + A.Y[a * 3 + 1] * B.W[c * 3 + 1] + A.Y[a * 3 + 2] * B.W[c * 3 + 2];
          }
          if (Yf && Wfj) {
            for (let a = 0; a < 6; a++) {
              const v = A.Y[a * 3] * Wfj[0] + A.Y[a * 3 + 1] * Wfj[1] + A.Y[a * 3 + 2] * Wfj[2];
              S[(A.row + a) * nC + fIdx] -= v;
              S[fIdx * nC + A.row + a] -= v;
            }
          }
        }
        if (Yf && Wfj) {
          S[fIdx * nC + fIdx] -= Yf[0] * Wfj[0] + Yf[1] * Wfj[1] + Yf[2] * Wfj[2];
          bvec[fIdx] += Yf[0] * g[0] + Yf[1] * g[1] + Yf[2] * g[2];
        }
      }
      const dc = nC > 0 ? choleskySolve(S, bvec, nC) : new Float64Array(0);
      if (!dc) {
        lambda *= 10;
        continue;
      }
      // points: dp = V⁻¹ (−gp − Wᵀ dc)
      const Xn = Float64Array.from(X);
      for (let jp = 0; jp < np; jp++) {
        const Vi = Vinv[jp];
        if (!Vi) continue;
        const rhs = [-gp[jp * 3], -gp[jp * 3 + 1], -gp[jp * 3 + 2]];
        for (const o of obsByPt[jp]) {
          if (wts[o] === 0) continue;
          const k = camIdx.get(p.obsImg[o]);
          if (k !== undefined)
            for (let a = 0; a < 6; a++) {
              const d = dc[6 * k + a];
              for (let c = 0; c < 3; c++) rhs[c] -= Wc[o * 18 + a * 3 + c] * d;
            }
          if (p.refineFocal) for (let c = 0; c < 3; c++) rhs[c] -= Wf[o * 3 + c] * dc[fIdx];
        }
        const j = ptList[jp];
        for (let c = 0; c < 3; c++) Xn[3 * j + c] += Vi[c * 3] * rhs[0] + Vi[c * 3 + 1] * rhs[1] + Vi[c * 3 + 2] * rhs[2];
      }
      const Rn = R.slice(), tn = t.slice();
      p.varCams.forEach((img, k) => {
        const d = dc.subarray(6 * k, 6 * k + 6);
        Rn[img] = mul3(expSO3([d[0], d[1], d[2]]), R[img]!);
        const ti = t[img]!;
        tn[img] = [ti[0] + d[3], ti[1] + d[4], ti[2] + d[5]];
      });
      const fn = p.refineFocal ? f + dc[fIdx] : f;
      const next = evalCost(Rn, tn, fn, Xn);
      if (next.cost < cur.cost) {
        const rel = (cur.cost - next.cost) / cur.cost;
        R = Rn;
        t = tn;
        f = fn;
        X = Xn;
        cur = next;
        lambda = Math.max(lambda / 3, 1e-9);
        accepted = true;
        if (rel < 1e-6) it = p.iters;
      } else {
        lambda *= 5;
      }
    }
    if (!accepted) break;
  }
  p.varCams.forEach((img) => {
    p.R[img] = R[img];
    p.t[img] = t[img];
  });
  p.points.set(X);
  return { f, rmsBefore, rmsAfter: cur.rms, iterations: it };
}
