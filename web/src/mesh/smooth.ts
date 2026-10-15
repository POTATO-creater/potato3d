// Feature-preserving smoothing of the simplified mesh:
// - bilateral filtering of face normals (Zheng et al. 2011): neighbours across
//   a crease barely count, so the dress hem or the neckline survive while
//   bumps flatten; then vertices move to agree with the filtered normals;
// - untangle: Laplacian relaxation of only the folded faces (normal against
//   their ring), whose shading shows as pin-prick pits;
// - thin parts (stilettos, fingers: an inward ray hits the other side within
//   a few edges) stay frozen, or both would thin them until they break.

import type { Mesh } from "./extract.ts";
import { Bvh } from "./bvh.ts";
import { faceRing, type Rows, vertexFaces, vertexNeighbours } from "./topology.ts";

export interface SmoothOptions {
  /** Normal difference scale of the bilateral filter. */
  sigmaR: number;
  normalIterations: number;
  vertexIterations: number;
  /** Thin = inside thickness below this many mean edges. */
  thinEdges: number;
}

export const SMOOTH_DEFAULTS: SmoothOptions = { sigmaR: 0.15, normalIterations: 6, vertexIterations: 10, thinEdges: 2.5 };

export interface SmoothStats {
  thinVertices: number;
  foldsBefore: number;
  foldsAfter: number;
}

interface FaceGeometry {
  n: Float64Array;
  c: Float64Array;
  area: Float64Array;
}

function faceGeometry(P: Float32Array, I: Uint32Array): FaceGeometry {
  const nt = I.length / 3;
  const n = new Float64Array(3 * nt), c = new Float64Array(3 * nt), area = new Float64Array(nt);
  for (let t = 0; t < nt; t++) {
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], d = 3 * I[3 * t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[d] - P[a], vy = P[d + 1] - P[a + 1], vz = P[d + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz);
    area[t] = l / 2;
    n.set([nx / (l || 1), ny / (l || 1), nz / (l || 1)], 3 * t);
    for (let k = 0; k < 3; k++) c[3 * t + k] = (P[a + k] + P[b + k] + P[d + k]) / 3;
  }
  return { n, c, area };
}

/** Per face: its normal · the area-weighted mean normal of its ring. */
function foldDeviation(g: FaceGeometry, ring: Rows) {
  const nt = g.area.length, dev = new Float64Array(nt);
  for (let t = 0; t < nt; t++) {
    let x = 0, y = 0, z = 0;
    for (let j = ring.start[t]; j < ring.start[t + 1]; j++) {
      const u = ring.items[j], w = g.area[u];
      x += g.n[3 * u] * w;
      y += g.n[3 * u + 1] * w;
      z += g.n[3 * u + 2] * w;
    }
    const l = Math.hypot(x, y, z) || 1;
    dev[t] = (g.n[3 * t] * x + g.n[3 * t + 1] * y + g.n[3 * t + 2] * z) / l;
  }
  return dev;
}

const countFolds = (dev: Float64Array) => dev.reduce((s, d) => s + (d < 0 ? 1 : 0), 0);

/** Mean edge of a mesh of this area and triangle count if it were uniform. */
export function uniformEdge(P: Float32Array, I: Uint32Array) {
  const g = faceGeometry(P, I);
  const area = g.area.reduce((a, b) => a + b, 0);
  return Math.sqrt((4 * area) / (Math.sqrt(3) * (I.length / 3)));
}

/** Vertices where a ray into the surface (along −normal) hits the other side
 *  within `maxThickness`, plus one ring. */
export function thinVertices(P: Float32Array, I: Uint32Array, nb: Rows, vf: Rows, maxThickness: number) {
  const nv = P.length / 3, g = faceGeometry(P, I);
  const bvh = new Bvh(P, I);
  const thin = new Uint8Array(nv);
  for (let v = 0; v < nv; v++) {
    let x = 0, y = 0, z = 0;
    for (let j = vf.start[v]; j < vf.start[v + 1]; j++) {
      const t = vf.items[j], w = g.area[t];
      x += g.n[3 * t] * w;
      y += g.n[3 * t + 1] * w;
      z += g.n[3 * t + 2] * w;
    }
    const l = Math.hypot(x, y, z);
    if (!l) continue;
    const d = [-x / l, -y / l, -z / l];
    const own = (tri: number) => I[3 * tri] === v || I[3 * tri + 1] === v || I[3 * tri + 2] === v;
    if (bvh.cast([P[3 * v], P[3 * v + 1], P[3 * v + 2]], d, maxThickness, own).tri >= 0) thin[v] = 1;
  }
  const grown = thin.slice();
  for (let v = 0; v < nv; v++) if (thin[v]) for (let j = nb.start[v]; j < nb.start[v + 1]; j++) grown[nb.items[j]] = 1;
  return grown;
}

export function smoothMesh(m: Mesh, o: SmoothOptions = SMOOTH_DEFAULTS): { mesh: Mesh; stats: SmoothStats } {
  const I = m.indices, nv = m.positions.length / 3, nt = I.length / 3;
  const P = m.positions.slice();
  const vf = vertexFaces(I, nv), nb = vertexNeighbours(I, nv, vf), ring = faceRing(I, nv, vf);
  const edge = uniformEdge(P, I);
  const frozen = thinVertices(P, I, nb, vf, o.thinEdges * edge);
  const foldsBefore = countFolds(foldDeviation(faceGeometry(P, I), ring));

  // Bilateral normal filter (centroids and areas of the input).
  const g = faceGeometry(P, I);
  let meanEdge = 0;
  for (let t = 0; t < nt; t++) {
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1];
    meanEdge += Math.hypot(P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]);
  }
  const sigmaS = 1.5 * (meanEdge / nt);
  const s2 = 2 * sigmaS * sigmaS, r2 = 2 * o.sigmaR * o.sigmaR;
  let n = g.n;
  for (let it = 0; it < o.normalIterations; it++) {
    const next = new Float64Array(3 * nt);
    for (let t = 0; t < nt; t++) {
      let x = n[3 * t] * g.area[t], y = n[3 * t + 1] * g.area[t], z = n[3 * t + 2] * g.area[t];
      for (let j = ring.start[t]; j < ring.start[t + 1]; j++) {
        const u = ring.items[j];
        const dc = (g.c[3 * t] - g.c[3 * u]) ** 2 + (g.c[3 * t + 1] - g.c[3 * u + 1]) ** 2 + (g.c[3 * t + 2] - g.c[3 * u + 2]) ** 2;
        const dn = (n[3 * t] - n[3 * u]) ** 2 + (n[3 * t + 1] - n[3 * u + 1]) ** 2 + (n[3 * t + 2] - n[3 * u + 2]) ** 2;
        const w = g.area[u] * Math.exp(-dc / s2 - dn / r2);
        x += n[3 * u] * w;
        y += n[3 * u + 1] * w;
        z += n[3 * u + 2] * w;
      }
      const l = Math.hypot(x, y, z) || 1;
      next.set([x / l, y / l, z / l], 3 * t);
    }
    n = next;
  }
  // Vertices follow the filtered normals: x += mean over faces of n (n · (c − x)).
  const move = new Float64Array(3 * nv);
  for (let it = 0; it < o.vertexIterations; it++) {
    move.fill(0);
    for (let t = 0; t < nt; t++) {
      const cx = (P[3 * I[3 * t]] + P[3 * I[3 * t + 1]] + P[3 * I[3 * t + 2]]) / 3;
      const cy = (P[3 * I[3 * t] + 1] + P[3 * I[3 * t + 1] + 1] + P[3 * I[3 * t + 2] + 1]) / 3;
      const cz = (P[3 * I[3 * t] + 2] + P[3 * I[3 * t + 1] + 2] + P[3 * I[3 * t + 2] + 2]) / 3;
      for (let k = 0; k < 3; k++) {
        const v = I[3 * t + k];
        const d = n[3 * t] * (cx - P[3 * v]) + n[3 * t + 1] * (cy - P[3 * v + 1]) + n[3 * t + 2] * (cz - P[3 * v + 2]);
        move[3 * v] += n[3 * t] * d;
        move[3 * v + 1] += n[3 * t + 1] * d;
        move[3 * v + 2] += n[3 * t + 2] * d;
      }
    }
    for (let v = 0; v < nv; v++) {
      if (frozen[v]) continue;
      const deg = vf.start[v + 1] - vf.start[v] || 1;
      for (let k = 0; k < 3; k++) P[3 * v + k] += move[3 * v + k] / deg;
    }
  }

  untangle(P, I, nb, ring, frozen);
  const foldsAfter = countFolds(foldDeviation(faceGeometry(P, I), ring));
  return {
    mesh: { ...m, positions: P },
    stats: { thinVertices: frozen.reduce((s, x) => s + x, 0), foldsBefore, foldsAfter },
  };
}

/** Relaxes (half-way to the neighbours' mean) the vertices of faces that
 *  point away from their ring, plus one ring, until there are none. */
function untangle(P: Float32Array, I: Uint32Array, nb: Rows, ring: Rows, frozen: Uint8Array, iterations = 30) {
  const nv = P.length / 3;
  const sel = new Uint8Array(nv), grown = new Uint8Array(nv);
  for (let it = 0; it < iterations; it++) {
    const dev = foldDeviation(faceGeometry(P, I), ring);
    sel.fill(0);
    let any = false;
    for (let t = 0; t < dev.length; t++)
      if (dev[t] < 0.3) {
        sel[I[3 * t]] = sel[I[3 * t + 1]] = sel[I[3 * t + 2]] = 1;
        any = true;
      }
    if (!any) return;
    grown.set(sel);
    for (let v = 0; v < nv; v++) if (sel[v]) for (let j = nb.start[v]; j < nb.start[v + 1]; j++) grown[nb.items[j]] = 1;
    const target: [number, number, number, number][] = [];
    for (let v = 0; v < nv; v++) {
      if (!grown[v] || frozen[v]) continue;
      const d = nb.start[v + 1] - nb.start[v];
      if (!d) continue;
      let x = 0, y = 0, z = 0;
      for (let j = nb.start[v]; j < nb.start[v + 1]; j++) {
        const u = 3 * nb.items[j];
        x += P[u];
        y += P[u + 1];
        z += P[u + 2];
      }
      target.push([v, x / d, y / d, z / d]);
    }
    for (const [v, x, y, z] of target) {
      P[3 * v] = 0.5 * P[3 * v] + 0.5 * x;
      P[3 * v + 1] = 0.5 * P[3 * v + 1] + 0.5 * y;
      P[3 * v + 2] = 0.5 * P[3 * v + 2] + 0.5 * z;
    }
  }
}
