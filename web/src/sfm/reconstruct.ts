// Incremental SfM for orbit video: geometric verification, tracks, initial
// pair, PnP registration, triangulation and bundle adjustment.

import type { Msg } from "../../../src/lib/api";
import { bundleAdjust } from "./ba";
import {
  center,
  type Pose,
  ransacFundamental,
  relativePose,
  solvePnP,
  toCam,
  triAngle,
  triangulate,
} from "./geometry";
import { cross, eye3, expSO3, inv3, type Mat3, mul3, mulVec3, symEig, transpose3, type Vec3 } from "./linalg";
import type { PairMatches } from "./matcher";

export interface ImageFeatures {
  name: string;
  width: number;
  height: number;
  n: number;
  xy: Float32Array;
  rgb: Uint8Array;
}

interface InitPair {
  a: number;
  b: number;
  pose: Pose;
  score: number;
}

export interface SfmModel {
  f: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
  poses: (Pose | null)[];
  points: { X: Vec3; rgb: [number, number, number]; obs: [number, number][]; error: number }[];
  stats: Record<string, number | string>;
}

const MAX_ERR = 4; // px
const MIN_ANGLE = 1.5; // degrees
const MIN_PNP = 20;

export function reconstruct(
  imgs: ImageFeatures[],
  matches: PairMatches[],
  log: (s: string) => void,
  progress: (frac: number, msg: Msg) => void,
  closedLoop = false,
): SfmModel {
  const N = imgs.length;
  const { width: W, height: H } = imgs[0];
  const cx = W / 2, cy = H / 2;
  const f0 = 1.2 * Math.max(W, H); // same initial guess as COLMAP

  // ---- 1. geometric verification ----
  const t0 = performance.now();
  const verified: { a: number; b: number; ia: Uint32Array; ib: Uint32Array }[] = [];
  for (const m of matches) {
    const n = m.ia.length;
    if (n < 20) continue;
    const p1 = new Float64Array(n * 2), p2 = new Float64Array(n * 2);
    for (let k = 0; k < n; k++) {
      p1[2 * k] = imgs[m.a].xy[2 * m.ia[k]];
      p1[2 * k + 1] = imgs[m.a].xy[2 * m.ia[k] + 1];
      p2[2 * k] = imgs[m.b].xy[2 * m.ib[k]];
      p2[2 * k + 1] = imgs[m.b].xy[2 * m.ib[k] + 1];
    }
    const r = ransacFundamental(p1, p2, n, 2.0, m.a * 7919 + m.b);
    if (!r || r.count < 20) continue;
    const ia: number[] = [], ib: number[] = [];
    for (let k = 0; k < n; k++)
      if (r.inliers[k]) {
        ia.push(m.ia[k]);
        ib.push(m.ib[k]);
      }
    verified.push({ a: m.a, b: m.b, ia: Uint32Array.from(ia), ib: Uint32Array.from(ib) });
  }
  log(`Geometric verification: ${verified.length}/${matches.length} pairs in ${((performance.now() - t0) / 1000).toFixed(1)} s`);

  // ---- 2. tracks (union-find) ----
  const off: number[] = [];
  let nodes = 0;
  for (const im of imgs) {
    off.push(nodes);
    nodes += im.n;
  }
  const parent = new Int32Array(nodes).map((_, i) => i);
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const nodeImg = new Int32Array(nodes);
  imgs.forEach((im, i) => nodeImg.fill(i, off[i], off[i] + im.n));
  // Closest pairs first (most reliable), and a merge that would put two
  // features of the same image in one track is refused: otherwise a single
  // wrong match (repetitive floor between frames far apart) joins two whole
  // tracks, and dropping the ambiguous observations afterwards takes the
  // right ones with it.
  const gapOf = (v: { a: number; b: number }) => (closedLoop ? Math.min(v.b - v.a, N - (v.b - v.a)) : v.b - v.a);
  const byTrust = [...verified].sort((x, y) => gapOf(x) - gapOf(y) || y.ia.length - x.ia.length);
  const members = new Map<number, number[]>(); // root → images in the track (sorted), if > 1
  const imagesOf = (r: number) => members.get(r) ?? [nodeImg[r]];
  let refused = 0;
  for (const v of byTrust)
    for (let k = 0; k < v.ia.length; k++) {
      const ra = find(off[v.a] + v.ia[k]), rb = find(off[v.b] + v.ib[k]);
      if (ra === rb) continue;
      const ia = imagesOf(ra), ib = imagesOf(rb);
      const merged: number[] = [];
      let i = 0, j = 0, clash = false;
      while (i < ia.length && j < ib.length) {
        if (ia[i] === ib[j]) {
          clash = true;
          break;
        }
        merged.push(ia[i] < ib[j] ? ia[i++] : ib[j++]);
      }
      if (clash) {
        refused++;
        continue;
      }
      while (i < ia.length) merged.push(ia[i++]);
      while (j < ib.length) merged.push(ib[j++]);
      parent[ra] = rb;
      members.delete(ra);
      members.set(rb, merged);
    }
  const groups = new Map<number, number[]>();
  for (let x = 0; x < nodes; x++) {
    const r = find(x);
    const g = groups.get(r);
    if (g) g.push(x);
    else groups.set(r, [x]);
  }
  const trackObs: [number, number][][] = [];
  const featTrack = imgs.map((im) => new Int32Array(im.n).fill(-1));
  let dropped = 0;
  for (const g0 of groups.values()) {
    if (g0.length < 2) continue;
    // Two features from the same image in the same track: a wrong match chained
    // different points. Only those observations are dropped, not the whole track.
    const count = new Map<number, number>();
    for (const x of g0) count.set(nodeImg[x], (count.get(nodeImg[x]) ?? 0) + 1);
    const g = g0.filter((x) => count.get(nodeImg[x]) === 1);
    dropped += g0.length - g.length;
    if (g.length < 2) continue;
    const id = trackObs.length;
    trackObs.push(g.map((x) => [nodeImg[x], x - off[nodeImg[x]]]));
    for (const x of g) featTrack[nodeImg[x]][x - off[nodeImg[x]]] = id;
  }
  const pairInliers = new Map(verified.map((v) => [`${v.a},${v.b}`, v.ia.length]));
  log(`Tracks: ${trackObs.length} (${refused} conflicting merges refused, ${dropped} ambiguous observations dropped)`);

  // ---- 3. initial pair candidates ----
  // Parallax (median triangulation angle ≥ 2°) and point count, not distance
  // in frames: in a video with few frames, 3 frames can already be 27°.
  const uv0 = (img: number, feat: number): [number, number] => [imgs[img].xy[2 * feat], imgs[img].xy[2 * feat + 1]];
  // The pairs with the most matches are almost always immediate neighbours with
  // little parallax (on a telephoto lens, < 1° between frames). Evaluate the
  // best of each distance band (√2 between bands, up to N/4), not just the ones
  // with the most matches.
  const candidates: InitPair[] = [];
  const bands = new Map<number, typeof verified>();
  for (const v of verified) {
    const gap = Math.min(v.b - v.a, N - (v.b - v.a));
    if (gap > N / 4) continue;
    const band = Math.floor(2 * Math.log2(gap));
    bands.get(band)?.push(v) ?? bands.set(band, [v]);
  }
  const perBand = Math.ceil(60 / Math.max(1, bands.size));
  const pool = [...bands.values()].flatMap((b) => b.sort((x, y) => y.ia.length - x.ia.length).slice(0, perBand));
  for (const v of pool) {
    const n = v.ia.length;
    const p1 = new Float64Array(n * 2), p2 = new Float64Array(n * 2);
    v.ia.forEach((k, j) => p1.set(uv0(v.a, k), 2 * j));
    v.ib.forEach((k, j) => p2.set(uv0(v.b, k), 2 * j));
    const n1 = p1.map((u, j) => (u - (j % 2 ? cy : cx)) / f0);
    const n2 = p2.map((u, j) => (u - (j % 2 ? cy : cx)) / f0);
    const F = ransacFundamental(p1, p2, n, 2.0, 99);
    if (!F) continue;
    const idx = [...Array(n).keys()].filter((k) => F.inliers[k]);
    const rel = relativePose(F.F, f0, cx, cy, n1, n2, idx);
    const id: Pose = { R: eye3(), t: [0, 0, 0] };
    const angles: number[] = [];
    for (const k of idx) {
      const X = triangulate([id, rel.pose], [n1[2 * k], n1[2 * k + 1], n2[2 * k], n2[2 * k + 1]]);
      if (X && X[2] > 0) angles.push(triAngle([0, 0, 0], center(rel.pose), X));
    }
    angles.sort((x, y) => x - y);
    const med = angles[angles.length >> 1] ?? 0;
    if (med >= 2 && rel.good >= 25) candidates.push({ a: v.a, b: v.b, pose: rel.pose, score: rel.good * Math.min(med, 6) });
  }
  candidates.sort((x, y) => y.score - x.score);
  // further attempts in other regions of the orbit: no repeated images
  const tries: InitPair[] = [];
  for (const c of candidates) {
    if (tries.some((t) => [t.a, t.b].includes(c.a) || [t.a, t.b].includes(c.b))) continue;
    tries.push(c);
    if (tries.length === 3) break;
  }
  if (!tries.length) throw new Error("No initial pair with enough parallax.");

  /** A full reconstruction from one initial pair. */
  const attempt = (init: InitPair): SfmModel => {
    let f = f0;
    // ---- state ----
    const R: (Mat3 | null)[] = new Array(N).fill(null);
    const T: (Vec3 | null)[] = new Array(N).fill(null);
    const pose = (i: number): Pose | null => (R[i] ? { R: R[i]!, t: T[i]! } : null);
    const trackPoint = new Int32Array(trackObs.length).fill(-1);
    const ptX: number[] = []; // 3 per point
    const ptTrack: number[] = [];
    const order: number[] = []; // registration order
    const uv = (img: number, feat: number): [number, number] => [imgs[img].xy[2 * feat], imgs[img].xy[2 * feat + 1]];
    const reprojErr = (img: number, feat: number, X: ArrayLike<number>) => {
      const c = toCam(pose(img)!, X);
      if (c[2] <= 0) return Infinity;
      const [u, v] = uv(img, feat);
      return Math.hypot((f * c[0]) / c[2] + cx - u, (f * c[1]) / c[2] + cy - v);
    };
    const norm = (img: number, feat: number): [number, number] => {
      const [u, v] = uv(img, feat);
      return [(u - cx) / f, (v - cy) / f];
    };

    /** Triangulates point-less tracks that have ≥2 registered observations. */
    const triangulateTracks = (tracks: Iterable<number>, maxErr = MAX_ERR) => {
      let added = 0;
      for (const tr of tracks) {
        if (trackPoint[tr] >= 0) continue;
        const obs = trackObs[tr].filter(([i]) => R[i]);
        if (obs.length < 2) continue;
        const X = triangulate(
          obs.map(([i]) => pose(i)!),
          obs.flatMap(([i, k]) => norm(i, k)),
        );
        if (!X) continue;
        let ok = 0, maxAng = 0;
        for (const [i, k] of obs) if (reprojErr(i, k, X) < maxErr) ok++;
        if (ok < 2 || ok < obs.length * 0.5) continue;
        const c0 = center(pose(obs[0][0])!);
        for (const [i] of obs.slice(1)) maxAng = Math.max(maxAng, triAngle(c0, center(pose(i)!), X));
        if (maxAng < MIN_ANGLE) continue;
        trackPoint[tr] = ptTrack.length;
        ptTrack.push(tr);
        ptX.push(...X);
        added++;
      }
      return added;
    };

    /** Builds and runs the BA. `maxErr` filters out far-off observations first. */
    const runBA = (varCams: number[], refineFocal: boolean, iters: number, maxErr: number) => {
      const pts = new Set<number>();
      if (varCams.length === order.length - 1 || varCams.length === order.length) {
        for (let p = 0; p < ptTrack.length; p++) pts.add(p);
      } else {
        for (const img of varCams)
          for (let k = 0; k < imgs[img].n; k++) {
            const tr = featTrack[img][k];
            if (tr >= 0 && trackPoint[tr] >= 0) pts.add(trackPoint[tr]);
          }
      }
      const obsImg: number[] = [], obsPt: number[] = [], obsUV: number[] = [];
      for (const p of pts) {
        const X = ptX.slice(3 * p, 3 * p + 3);
        for (const [i, k] of trackObs[ptTrack[p]]) {
          if (!R[i]) continue;
          if (reprojErr(i, k, X) > maxErr) continue;
          obsImg.push(i);
          obsPt.push(p);
          obsUV.push(...uv(i, k));
        }
      }
      const points = Float64Array.from(ptX);
      const r = bundleAdjust({
        f, cx, cy, R, t: T, varCams, points,
        obsImg: Int32Array.from(obsImg), obsPt: Int32Array.from(obsPt), obsUV: Float64Array.from(obsUV),
        refineFocal, huber: 2, iters,
      });
      for (let i = 0; i < points.length; i++) ptX[i] = points[i];
      f = r.f;
      return r;
    };

    /** Removes points with <2 good observations or a small angle. */
    const filterPoints = () => {
      let removed = 0;
      for (let p = 0; p < ptTrack.length; p++) {
        const tr = ptTrack[p];
        if (tr < 0) continue;
        const X = ptX.slice(3 * p, 3 * p + 3);
        const good = trackObs[tr].filter(([i, k]) => R[i] && reprojErr(i, k, X) < MAX_ERR);
        let maxAng = 0;
        if (good.length >= 2) {
          const c0 = center(pose(good[0][0])!);
          for (const [i] of good.slice(1)) maxAng = Math.max(maxAng, triAngle(c0, center(pose(i)!), X));
        }
        if (good.length < 2 || maxAng < MIN_ANGLE) {
          trackPoint[tr] = -1;
          ptTrack[p] = -1;
          removed++;
        }
      }
      // compact in place (a push(...arr) with hundreds of thousands of
      // coordinates overflows the call stack)
      let w = 0;
      for (let p = 0; p < ptTrack.length; p++) {
        const tr = ptTrack[p];
        if (tr < 0) continue;
        trackPoint[tr] = w;
        ptTrack[w] = tr;
        for (let c = 0; c < 3; c++) ptX[3 * w + c] = ptX[3 * p + c];
        w++;
      }
      ptTrack.length = w;
      ptX.length = 3 * w;
      return removed;
    };

    R[init.a] = eye3();
    T[init.a] = [0, 0, 0];
    R[init.b] = init.pose.R;
    T[init.b] = init.pose.t;
    order.push(init.a, init.b);
    const initTracks = new Set<number>();
    for (let k = 0; k < imgs[init.a].n; k++) if (featTrack[init.a][k] >= 0) initTracks.add(featTrack[init.a][k]);
    triangulateTracks(initTracks);
    runBA([init.b], false, 20, 1e9);
    log(`Initial pair ${imgs[init.a].name} + ${imgs[init.b].name}: ${ptTrack.length} points`);

    // ---- 4. incremental registration ----
    const failed = new Set<number>();
    let lastGlobal = 2;
    while (true) {
      // next image: the one with the most 2D–3D correspondences
      let next = -1, bestCount = 0;
      for (let i = 0; i < N; i++) {
        if (R[i] || failed.has(i)) continue;
        let c = 0;
        for (let k = 0; k < imgs[i].n; k++) {
          const tr = featTrack[i][k];
          if (tr >= 0 && trackPoint[tr] >= 0) c++;
        }
        if (c > bestCount) {
          bestCount = c;
          next = i;
        }
      }
      if (next < 0 || bestCount < MIN_PNP) break;

      const corr: [number, number][] = []; // (feat, point)
      for (let k = 0; k < imgs[next].n; k++) {
        const tr = featTrack[next][k];
        if (tr >= 0 && trackPoint[tr] >= 0) corr.push([k, trackPoint[tr]]);
      }
      const X = new Float64Array(corr.length * 3), x = new Float64Array(corr.length * 2);
      corr.forEach(([k, p], j) => {
        X.set(ptX.slice(3 * p, 3 * p + 3), 3 * j);
        x.set(norm(next, k), 2 * j);
      });
      // initial pose: that of the nearest registered neighbour in the sequence
      let neighbor: Pose | null = null;
      for (let d = 1; d < N && !neighbor; d++) neighbor = pose((next + d) % N) ?? pose((next - d + N) % N);
      const r = solvePnP(X, x, corr.length, MAX_ERR / f, neighbor, next + 1);
      if (!r || r.inliers.length < MIN_PNP) {
        failed.add(next);
        log(`PnP failed on ${imgs[next].name} (${r?.inliers.length ?? 0}/${corr.length} inliers)`);
        continue;
      }
      R[next] = r.pose.R;
      T[next] = r.pose.t;
      order.push(next);
      // new triangulations from this image's tracks
      const tracks = new Set<number>();
      for (let k = 0; k < imgs[next].n; k++) if (featTrack[next][k] >= 0) tracks.add(featTrack[next][k]);
      triangulateTracks(tracks);

      // local BA: last registered cameras (except the first, which is fixed)
      const local = order.slice(-8).filter((i) => i !== order[0]);
      runBA(local, false, 6, MAX_ERR * 2);
      if (order.length >= lastGlobal * 1.25 && order.length >= 8) {
        // With few cameras BA is cheap: iterate until the focal length settles.
        // On a telephoto lens (focal ~10× the width) the focal × depth valley is
        // nearly flat and 15 iterations per round leave the focal creeping up;
        // cameras registered while it is short end up with shrunken angle steps
        // and the orbit doesn't close (a "hole" opens).
        const g = runBA(order.slice(1), true, order.length < 20 ? 150 : 15, MAX_ERR * 2);
        filterPoints();
        lastGlobal = order.length;
        log(`Global BA with ${order.length} cameras: ${g.rmsBefore.toFixed(2)} → ${g.rmsAfter.toFixed(2)} px, f = ${f.toFixed(1)}`);
      }
      progress(order.length / N, { key: "poses.webRegistered", args: { i: order.length, n: N } });
      // an image that failed may have more points now
      if (failed.size && order.length % 5 === 0) failed.clear();
    }

    // ---- 5. close the orbit ----
    // Registration grows on both sides of the initial pair and each side
    // shrinks the angle steps a little (focal × depth × rotation nearly
    // compensate each other). Where the fronts meet, a jump of tens of degrees
    // is left, and the observations spanning it fall outside any BA threshold,
    // so BA can't close it alone. Rotate each camera about the orbit axis to
    // spread the excess across all steps and re-triangulate; the BA below
    // settles the focal with the orbit already closed.
    const closeOrbit = () => {
      const ring = [...Array(N).keys()].filter((i) => R[i]);
      if (ring.length < Math.max(12, 0.8 * N)) return;
      const C = ring.map((i) => center(pose(i)!));
      const m = [0, 1, 2].map((a) => C.reduce((s, c) => s + c[a], 0) / C.length);
      const cov = new Float64Array(9);
      for (const c of C)
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) cov[a * 3 + b] += (c[a] - m[a]) * (c[b] - m[b]);
      const { values, V } = symEig(cov, 3);
      if (values[0] > 0.2 * values[1]) return; // centers far from a plane: not an orbit
      const axis: Vec3 = [V[0], V[3], V[6]];
      const e1: Vec3 = [V[2], V[5], V[8]];
      const e2 = cross(axis, e1);
      // least-squares circle in the plane: x² + y² = 2ax + 2by + c
      const P = C.map((c) => {
        const d = [c[0] - m[0], c[1] - m[1], c[2] - m[2]];
        return [d[0] * e1[0] + d[1] * e1[1] + d[2] * e1[2], d[0] * e2[0] + d[1] * e2[1] + d[2] * e2[2]];
      });
      const A = new Float64Array(9), rhs = new Float64Array(3);
      for (const [x, y] of P) {
        const row = [2 * x, 2 * y, 1], r2 = x * x + y * y;
        for (let a = 0; a < 3; a++) {
          rhs[a] += row[a] * r2;
          for (let b = 0; b < 3; b++) A[a * 3 + b] += row[a] * row[b];
        }
      }
      const Ai = inv3(A);
      if (!Ai) return;
      const sol = mulVec3(Ai, rhs);
      const o: Vec3 = [0, 1, 2].map((a) => m[a] + sol[0] * e1[a] + sol[1] * e2[a]) as Vec3;
      const theta = P.map(([x, y]) => Math.atan2(y - sol[1], x - sol[0]));
      const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
      let steps = ring.map((_, k) => wrap(theta[(k + 1) % ring.length] - theta[k]));
      const sorted = [...steps].sort((x, y) => x - y);
      const dir = Math.sign(sorted[sorted.length >> 1]) || 1;
      steps = steps.map((d) => d * dir);
      const total = steps.reduce((s, d) => s + d, 0);
      if (Math.abs(total - 2 * Math.PI) > 0.2) return; // not exactly one turn
      let seam = 0;
      steps.forEach((d, k) => d > steps[seam] && (seam = k));
      const around: number[] = [];
      for (let d = -4; d <= 4; d++) if (d) around.push(steps[(seam + d + ring.length) % ring.length]);
      around.sort((x, y) => x - y);
      const expected = Math.max(0, (around[3] + around[4]) / 2);
      const excess = steps[seam] - expected;
      const deg = (r: number) => (r * 180) / Math.PI;
      if (excess < (2 * Math.PI) / 180 || steps[seam] < 3 * Math.max(expected, sorted[sorted.length >> 1] * dir)) return;
      // Only close if the pair across the jump matches as well as real
      // neighbours: in a video that doesn't go all the way around, the jump is
      // real and matches poorly.
      const inliers = (i: number, j: number) => pairInliers.get(`${Math.min(i, j)},${Math.max(i, j)}`) ?? 0;
      const consecutive = ring.map((i, k) => inliers(i, ring[(k + 1) % ring.length])).sort((x, y) => x - y);
      if (inliers(ring[seam], ring[(seam + 1) % ring.length]) < 0.5 * consecutive[consecutive.length >> 1]) return;
      // new angle for each camera, starting from the one right after the jump
      const scale = (total - expected) / (total - steps[seam]);
      const delta = new Float64Array(ring.length);
      let cum = 0;
      for (let j = 0; j < ring.length; j++) {
        const k = (seam + 1 + j) % ring.length;
        delta[k] = (scale - 1) * cum;
        cum += steps[k];
      }
      const ref = delta[ring.indexOf(order[0])]; // the first camera stays fixed in BA
      ring.forEach((i, k) => {
        const Q = expSO3(axis.map((a) => a * dir * (delta[k] - ref)));
        const c = C[k];
        const qc = mulVec3(Q, [c[0] - o[0], c[1] - o[1], c[2] - o[2]]);
        const cNew: Vec3 = [o[0] + qc[0], o[1] + qc[1], o[2] + qc[2]];
        R[i] = mul3(R[i]!, transpose3(Q));
        const t = mulVec3(R[i]!, cNew);
        T[i] = [-t[0], -t[1], -t[2]];
      });
      // the old points follow the pre-closure geometry: redo them all
      trackPoint.fill(-1);
      ptX.length = 0;
      ptTrack.length = 0;
      triangulateTracks(trackObs.keys(), 24);
      log(
        `Orbit closed: ${deg(steps[seam]).toFixed(1)}° jump between ${imgs[ring[seam]].name} and ` +
          `${imgs[ring[(seam + 1) % ring.length]].name} spread out (steps ×${scale.toFixed(3)}), ${ptTrack.length} points`,
      );
    };
    if (closedLoop) closeOrbit();

    // ---- 6. wrap-up: recover observations with a decreasing threshold ----
    for (const thr of [24, 12, 6, MAX_ERR]) {
      triangulateTracks(trackObs.keys());
      const g = runBA(order.slice(1), true, 25, thr);
      filterPoints();
      log(`Final BA (threshold ${thr} px): ${g.rmsBefore.toFixed(2)} → ${g.rmsAfter.toFixed(2)} px, f = ${f.toFixed(1)}`);
    }

    // ---- output ----
    const points: SfmModel["points"] = [];
    let errSum = 0, nObs = 0;
    for (let p = 0; p < ptTrack.length; p++) {
      const X: Vec3 = [ptX[3 * p], ptX[3 * p + 1], ptX[3 * p + 2]];
      const obs = trackObs[ptTrack[p]].filter(([i, k]) => R[i] && reprojErr(i, k, X) < MAX_ERR);
      if (obs.length < 2) continue;
      let r = 0, g = 0, b = 0, e = 0;
      for (const [i, k] of obs) {
        r += imgs[i].rgb[3 * k];
        g += imgs[i].rgb[3 * k + 1];
        b += imgs[i].rgb[3 * k + 2];
        e += reprojErr(i, k, X);
      }
      const m = obs.length;
      points.push({ X, rgb: [Math.round(r / m), Math.round(g / m), Math.round(b / m)], obs, error: e / m });
      errSum += e;
      nObs += m;
    }
    const registered = R.filter(Boolean).length;
    return {
      f, cx, cy, width: W, height: H,
      poses: R.map((_, i) => pose(i)),
      points,
      stats: { registered, total: N, points: points.length, observations: nObs, meanError: errSum / Math.max(nObs, 1), focal: f },
    };
  };

  let best: SfmModel | null = null;
  for (const init of tries) {
    if (best) log(`Only ${best.stats.registered}/${N} cameras: trying another initial pair`);
    const m = attempt(init);
    if (!best || Number(m.stats.registered) > Number(best.stats.registered)) best = m;
    if (Number(best.stats.registered) >= 0.9 * N) break;
  }
  return best!;
}

