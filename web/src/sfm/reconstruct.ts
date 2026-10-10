// SfM incremental para vídeo orbital: verificação geométrica, tracks,
// par inicial, registro por PnP, triangulação e bundle adjustment.

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
import { eye3, type Mat3, type Vec3 } from "./linalg";
import type { PairMatches } from "./matcher";

export interface ImageFeatures {
  name: string;
  width: number;
  height: number;
  n: number;
  xy: Float32Array;
  rgb: Uint8Array;
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
const MIN_ANGLE = 1.5; // graus
const MIN_PNP = 20;

export function reconstruct(
  imgs: ImageFeatures[],
  matches: PairMatches[],
  log: (s: string) => void,
  progress: (frac: number, msg: string) => void,
): SfmModel {
  const N = imgs.length;
  const { width: W, height: H } = imgs[0];
  const cx = W / 2, cy = H / 2;
  let f = 1.2 * Math.max(W, H); // mesmo chute inicial do COLMAP

  // ---- 1. verificação geométrica ----
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
  log(`Verificação geométrica: ${verified.length}/${matches.length} pares em ${((performance.now() - t0) / 1000).toFixed(1)} s`);

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
  for (const v of verified)
    for (let k = 0; k < v.ia.length; k++) {
      const ra = find(off[v.a] + v.ia[k]), rb = find(off[v.b] + v.ib[k]);
      if (ra !== rb) parent[ra] = rb;
    }
  const nodeImg = new Int32Array(nodes);
  imgs.forEach((im, i) => nodeImg.fill(i, off[i], off[i] + im.n));
  const groups = new Map<number, number[]>();
  for (let x = 0; x < nodes; x++) {
    const r = find(x);
    const g = groups.get(r);
    if (g) g.push(x);
    else groups.set(r, [x]);
  }
  const trackObs: [number, number][][] = [];
  const featTrack = imgs.map((im) => new Int32Array(im.n).fill(-1));
  for (const g of groups.values()) {
    if (g.length < 2) continue;
    const seen = new Set<number>();
    let conflict = false;
    for (const x of g) {
      if (seen.has(nodeImg[x])) conflict = true;
      seen.add(nodeImg[x]);
    }
    if (conflict) continue; // dois pontos da mesma imagem: track ambíguo
    const id = trackObs.length;
    trackObs.push(g.map((x) => [nodeImg[x], x - off[nodeImg[x]]]));
    for (const x of g) featTrack[nodeImg[x]][x - off[nodeImg[x]]] = id;
  }
  log(`Tracks: ${trackObs.length}`);

  // ---- estado ----
  const R: (Mat3 | null)[] = new Array(N).fill(null);
  const T: (Vec3 | null)[] = new Array(N).fill(null);
  const pose = (i: number): Pose | null => (R[i] ? { R: R[i]!, t: T[i]! } : null);
  const trackPoint = new Int32Array(trackObs.length).fill(-1);
  const ptX: number[] = []; // 3 por ponto
  const ptTrack: number[] = [];
  const order: number[] = []; // ordem de registro
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

  /** Triangula tracks sem ponto que tenham ≥2 observações registradas. */
  const triangulateTracks = (tracks: Iterable<number>) => {
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
      for (const [i, k] of obs) if (reprojErr(i, k, X) < MAX_ERR) ok++;
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

  /** Monta e roda o BA. `maxErr` filtra observações muito fora antes. */
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

  /** Remove pontos com <2 observações boas ou ângulo pequeno. */
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
    // compacta
    const keep = ptTrack.map((tr, p) => [tr, p]).filter(([tr]) => tr >= 0);
    const newX: number[] = [];
    const newTrack: number[] = [];
    for (const [tr, p] of keep) {
      trackPoint[tr] = newTrack.length;
      newTrack.push(tr);
      newX.push(ptX[3 * p], ptX[3 * p + 1], ptX[3 * p + 2]);
    }
    ptX.length = 0;
    ptX.push(...newX);
    ptTrack.length = 0;
    ptTrack.push(...newTrack);
    return removed;
  };

  // ---- 3. par inicial ----
  const byInliers = [...verified].filter((v) => Math.abs(v.b - v.a) >= 3 && Math.abs(v.b - v.a) <= N / 4);
  byInliers.sort((x, y) => y.ia.length - x.ia.length);
  let init: { a: number; b: number; pose: Pose; score: number } | null = null;
  for (const v of byInliers.slice(0, 40)) {
    const n1 = new Float64Array(v.ia.length * 2), n2 = new Float64Array(v.ia.length * 2);
    v.ia.forEach((k, j) => n1.set(norm(v.a, k), 2 * j));
    v.ib.forEach((k, j) => n2.set(norm(v.b, k), 2 * j));
    const p1 = new Float64Array(v.ia.length * 2), p2 = new Float64Array(v.ia.length * 2);
    v.ia.forEach((k, j) => p1.set(uv(v.a, k), 2 * j));
    v.ib.forEach((k, j) => p2.set(uv(v.b, k), 2 * j));
    const F = ransacFundamental(p1, p2, v.ia.length, 2.0, 99);
    if (!F) continue;
    const idx = [...Array(v.ia.length).keys()].filter((k) => F.inliers[k]);
    const rel = relativePose(F.F, f, cx, cy, n1, n2, idx);
    // ângulo mediano de triangulação
    const id: Pose = { R: eye3(), t: [0, 0, 0] };
    const angles: number[] = [];
    for (const k of idx) {
      const X = triangulate([id, rel.pose], [n1[2 * k], n1[2 * k + 1], n2[2 * k], n2[2 * k + 1]]);
      if (X && X[2] > 0) angles.push(triAngle([0, 0, 0], center(rel.pose), X));
    }
    angles.sort((x, y) => x - y);
    const med = angles[angles.length >> 1] ?? 0;
    const score = rel.good * Math.min(med, 6);
    if (med >= 2 && (!init || score > init.score)) init = { a: v.a, b: v.b, pose: rel.pose, score };
  }
  if (!init) throw new Error("Nenhum par inicial com paralaxe suficiente.");
  R[init.a] = eye3();
  T[init.a] = [0, 0, 0];
  R[init.b] = init.pose.R;
  T[init.b] = init.pose.t;
  order.push(init.a, init.b);
  const initTracks = new Set<number>();
  for (let k = 0; k < imgs[init.a].n; k++) if (featTrack[init.a][k] >= 0) initTracks.add(featTrack[init.a][k]);
  triangulateTracks(initTracks);
  runBA([init.b], false, 20, 1e9);
  log(`Par inicial ${imgs[init.a].name} + ${imgs[init.b].name}: ${ptTrack.length} pontos`);

  // ---- 4. registro incremental ----
  const failed = new Set<number>();
  let lastGlobal = 2;
  while (true) {
    // próxima imagem: a com mais correspondências 2D–3D
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

    const corr: [number, number][] = []; // (feat, ponto)
    for (let k = 0; k < imgs[next].n; k++) {
      const tr = featTrack[next][k];
      if (tr >= 0 && trackPoint[tr] >= 0) corr.push([k, trackPoint[tr]]);
    }
    const X = new Float64Array(corr.length * 3), x = new Float64Array(corr.length * 2);
    corr.forEach(([k, p], j) => {
      X.set(ptX.slice(3 * p, 3 * p + 3), 3 * j);
      x.set(norm(next, k), 2 * j);
    });
    // pose inicial: a do vizinho registrado mais próximo na sequência
    let neighbor: Pose | null = null;
    for (let d = 1; d < N && !neighbor; d++) neighbor = pose((next + d) % N) ?? pose((next - d + N) % N);
    const r = solvePnP(X, x, corr.length, MAX_ERR / f, neighbor, next + 1);
    if (!r || r.inliers.length < MIN_PNP) {
      failed.add(next);
      log(`PnP falhou em ${imgs[next].name} (${r?.inliers.length ?? 0}/${corr.length} inliers)`);
      continue;
    }
    R[next] = r.pose.R;
    T[next] = r.pose.t;
    order.push(next);
    // novas triangulações com tracks desta imagem
    const tracks = new Set<number>();
    for (let k = 0; k < imgs[next].n; k++) if (featTrack[next][k] >= 0) tracks.add(featTrack[next][k]);
    triangulateTracks(tracks);

    // BA local: últimas câmeras registradas (exceto a primeira, fixa)
    const local = order.slice(-8).filter((i) => i !== order[0]);
    runBA(local, false, 6, MAX_ERR * 2);
    if (order.length >= lastGlobal * 1.25 && order.length >= 8) {
      const g = runBA(order.slice(1), true, 15, MAX_ERR * 2);
      filterPoints();
      lastGlobal = order.length;
      log(`BA global com ${order.length} câmeras: ${g.rmsBefore.toFixed(2)} → ${g.rmsAfter.toFixed(2)} px, f = ${f.toFixed(1)}`);
    }
    progress(order.length / N, `Registradas ${order.length}/${N} câmeras`);
    // uma imagem que falhou pode ter mais pontos agora
    if (failed.size && order.length % 5 === 0) failed.clear();
  }

  // ---- 5. fechamento: reaproveita observações com limiar decrescente ----
  for (const thr of [24, 12, 6, MAX_ERR]) {
    triangulateTracks(trackObs.keys());
    const g = runBA(order.slice(1), true, 25, thr);
    filterPoints();
    log(`BA final (limiar ${thr} px): ${g.rmsBefore.toFixed(2)} → ${g.rmsAfter.toFixed(2)} px, f = ${f.toFixed(1)}`);
  }

  // ---- saída ----
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
}
