// Hole filling for the parts no photo saw (top of the head, soles): each
// boundary loop is closed by ear clipping (smallest interior angle first),
// the patch is remeshed to an even edge length (split, collapse, valence
// flips, relaxation) and its new vertices are placed by biharmonic fairing,
// a smooth continuation of the surrounding curvature: a dome on a head,
// nearly flat soles. The observed surface never moves.

import { type Mesh, vertexNormals } from "./extract.ts";
import { boundaryLoops, type Rows, vertexFaces, vertexNeighbours } from "./topology.ts";

export interface FillOptions {
  /** Loops whose bounding box is wider than this (scene units) are an open
   *  border of the scene, not a hole. */
  maxExtent: number;
  /** Edge length inside the patches; at least the border's (the mesh is
   *  simplified afterwards, finer patches would only cost time). */
  edge?: number;
}

export interface FillStats {
  holes: number;
  filled: number;
  /** Vertices added inside the filled holes. */
  vertices: number;
}

class Grow {
  data: Float64Array;
  n = 0;
  constructor(init: ArrayLike<number>) {
    this.data = new Float64Array(Math.max(16, init.length * 1.25));
    this.data.set(init);
    this.n = init.length;
  }
  push(...xs: number[]) {
    if (this.n + xs.length > this.data.length) {
      const d = new Float64Array(this.data.length * 2);
      d.set(this.data);
      this.data = d;
    }
    for (const x of xs) this.data[this.n++] = x;
    return this.n / 3 - 1;
  }
}

const EDGE_KEY = 2 ** 26;
const edgeKey = (a: number, b: number) => (a < b ? a * EDGE_KEY + b : b * EDGE_KEY + a);

function sub(P: Float64Array, a: number, b: number) {
  return [P[3 * a] - P[3 * b], P[3 * a + 1] - P[3 * b + 1], P[3 * a + 2] - P[3 * b + 2]];
}
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: number[]) => Math.hypot(a[0], a[1], a[2]);

export function fillHoles(m: Mesh, o: FillOptions): { mesh: Mesh; stats: FillStats } {
  const nv0 = m.positions.length / 3;
  const vf = vertexFaces(m.indices, nv0);
  const nb = vertexNeighbours(m.indices, nv0, vf);
  const loops = boundaryLoops(m.indices, nv0, vf);
  const stats: FillStats = { holes: loops.length, filled: 0, vertices: 0 };
  if (!loops.length) return { mesh: m, stats };

  const P = new Grow(m.positions), C = new Grow(m.colors);
  const vn = vertexNormals(m);
  const newEdges = new Set<number>();
  const connected = (a: number, b: number) => {
    if (newEdges.has(edgeKey(a, b))) return true;
    if (a >= nv0 || b >= nv0) return false;
    for (let j = nb.start[a]; j < nb.start[a + 1]; j++) if (nb.items[j] === b) return true;
    return false;
  };
  const faces: number[] = [];

  for (const loop of loops) {
    let perimeter = 0;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < loop.length; i++) {
      perimeter += norm(sub(P.data, loop[i], loop[(i + 1) % loop.length]));
      for (let k = 0; k < 3; k++) {
        const x = P.data[3 * loop[i] + k];
        lo[k] = Math.min(lo[k], x);
        hi[k] = Math.max(hi[k], x);
      }
    }
    if (Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) > o.maxExtent) continue;
    const patch = clipEars(P.data, vn, loop, connected);
    if (!patch) continue;
    const first = P.n / 3;
    refine(P, C, patch, Math.max(perimeter / loop.length, o.edge ?? 0), connected);
    fair(P.data, patch, first, P.n / 3, nb, nv0);
    for (const t of patch) {
      faces.push(...t);
      for (let k = 0; k < 3; k++) newEdges.add(edgeKey(t[k], t[(k + 1) % 3]));
    }
    stats.filled++;
  }
  // new vertices that collapses orphaned are dropped
  const remap = new Int32Array(P.n / 3).fill(-1);
  for (let v = 0; v < nv0; v++) remap[v] = v;
  let nv = nv0;
  for (const v of faces) if (remap[v] < 0) remap[v] = nv++;
  stats.vertices = nv - nv0;
  const indices = new Uint32Array(m.indices.length + faces.length);
  indices.set(m.indices);
  indices.set(faces.map((v) => remap[v]), m.indices.length);
  const positions = new Float32Array(3 * nv), colors = new Float32Array(3 * nv);
  for (let v = 0; v < remap.length; v++) {
    if (remap[v] < 0) continue;
    positions.set(P.data.subarray(3 * v, 3 * v + 3), 3 * remap[v]);
    colors.set(C.data.subarray(3 * v, 3 * v + 3), 3 * remap[v]);
  }
  return { mesh: { positions, colors, indices }, stats };
}

/** Ear clipping of one loop, smallest interior angle first (measured around
 *  the local surface normal, so reflex corners wait). Null if it gets stuck. */
function clipEars(P: Float64Array, vn: Float32Array, loop: number[], connected: (a: number, b: number) => boolean): number[][] | null {
  const n = loop.length;
  const N = loop.map((v) => [vn[3 * v], vn[3 * v + 1], vn[3 * v + 2]]);
  const prev = loop.map((_, i) => (i + n - 1) % n), next = loop.map((_, i) => (i + 1) % n);
  const alive = new Uint8Array(n).fill(1);
  const angle = new Float64Array(n);
  const local = new Set<number>();
  const has = (a: number, b: number) => local.has(edgeKey(a, b)) || connected(a, b);
  const measure = (i: number) => {
    const v = loop[i], a = sub(P, loop[prev[i]], v), b = sub(P, loop[next[i]], v);
    // interior angle: from (next − v) counter-clockwise to (prev − v)
    let th = Math.atan2(dot(N[i], cross(b, a)), dot(b, a));
    if (th < 0) th += 2 * Math.PI;
    angle[i] = th;
  };
  for (let i = 0; i < n; i++) measure(i);
  const tris: number[][] = [];
  for (let left = n; left > 3; left--) {
    let best = -1;
    for (let i = 0; i < n; i++) {
      if (!alive[i] || (best >= 0 && angle[i] >= angle[best])) continue;
      const a = loop[prev[i]], b = loop[next[i]];
      if (a !== b && !has(a, b)) best = i;
    }
    if (best < 0) return null;
    const p = prev[best], q = next[best];
    const t = [loop[p], loop[best], loop[q]];
    tris.push(t);
    local.add(edgeKey(t[0], t[2]));
    const tn = cross(sub(P, t[1], t[0]), sub(P, t[2], t[0]));
    const l = norm(tn) || 1;
    for (const i of [p, q]) {
      const s = N[i].map((x, k) => x + tn[k] / l);
      const sl = norm(s) || 1;
      N[i] = s.map((x) => x / sl);
    }
    alive[best] = 0;
    next[p] = q;
    prev[q] = p;
    measure(p);
    measure(q);
  }
  const i = alive.indexOf(1);
  if (loop[prev[i]] === loop[next[i]]) return null;
  tris.push([loop[prev[i]], loop[i], loop[next[i]]]);
  return tris;
}

/** Isotropic remeshing of the patch towards edge length h: long interior
 *  edges (> 4/3 h) split at their midpoints, short ones (< 4/5 h) collapsed,
 *  flips evening out the valences, new vertices relaxed; until nothing
 *  splits. Border edges (shared with the mesh) stay as they are. */
function refine(P: Grow, C: Grow, tris: number[][], h: number, connected: (a: number, b: number) => boolean) {
  const first = P.n / 3;
  const edges = new Map<number, number[]>();
  /** patch triangles around each vertex */
  const around = new Map<number, Set<number>>();
  const link = (t: number) => {
    for (const v of tris[t]) {
      const s = around.get(v);
      if (s) s.add(t);
      else around.set(v, new Set([t]));
    }
    for (let k = 0; k < 3; k++) {
      const key = edgeKey(tris[t][k], tris[t][(k + 1) % 3]);
      const l = edges.get(key);
      if (l) l.push(t);
      else edges.set(key, [t]);
    }
  };
  const unlink = (t: number) => {
    for (const v of tris[t]) around.get(v)!.delete(t);
    for (let k = 0; k < 3; k++) {
      const key = edgeKey(tris[t][k], tris[t][(k + 1) % 3]);
      const l = edges.get(key)!;
      l.splice(l.indexOf(t), 1);
      if (!l.length) edges.delete(key);
    }
  };
  tris.forEach((_, t) => link(t));
  const len = (a: number, b: number) => norm(sub(P.data, a, b));
  /** The other triangle on edge (a, b) of triangle t, rotated so it reads (b, a, d). */
  const across = (t: number, k: number) => {
    const a = tris[t][k], b = tris[t][(k + 1) % 3];
    const l = edges.get(edgeKey(a, b))!;
    if (l.length !== 2) return null;
    const u = l[0] === t ? l[1] : l[0];
    const j = tris[u].indexOf(b);
    if (tris[u][(j + 1) % 3] !== a) return null;
    return { u, d: tris[u][(j + 2) % 3] };
  };

  for (let round = 0; round < 30; round++) {
    // one split per triangle of this round, then flips, so new slivers get
    // flipped before anyone splits them again
    let split = 0;
    const count = tris.length;
    for (let t = 0; t < count; t++) {
      if (!tris[t].length) continue;
      let k = 0, best = 0;
      for (let e = 0; e < 3; e++) {
        const l = len(tris[t][e], tris[t][(e + 1) % 3]);
        if (l > best) (best = l), (k = e);
      }
      if (best <= (4 / 3) * h) continue;
      const o = across(t, k);
      if (!o) continue;
      const a = tris[t][k], b = tris[t][(k + 1) % 3], c = tris[t][(k + 2) % 3];
      const pd = P.data, cd = C.data;
      const mid = P.push((pd[3 * a] + pd[3 * b]) / 2, (pd[3 * a + 1] + pd[3 * b + 1]) / 2, (pd[3 * a + 2] + pd[3 * b + 2]) / 2);
      C.push((cd[3 * a] + cd[3 * b]) / 2, (cd[3 * a + 1] + cd[3 * b + 1]) / 2, (cd[3 * a + 2] + cd[3 * b + 2]) / 2);
      unlink(t);
      unlink(o.u);
      tris[t] = [a, mid, c];
      tris[o.u] = [b, mid, o.d];
      tris.push([mid, b, c], [mid, a, o.d]);
      for (const x of [t, o.u, tris.length - 2, tris.length - 1]) link(x);
      split++;
    }
    // Collapse short edges (< 4/5 h) into their other end, when the patch
    // stays manifold and unfolded and no edge gets long. Loop vertices stay.
    for (let t = 0; t < tris.length; t++)
      for (let k = 0; k < 3 && tris[t].length; k++) {
        let q = tris[t][k], r = tris[t][(k + 1) % 3];
        if (len(q, r) >= 0.8 * h) continue;
        if (r < first) [q, r] = [r, q];
        if (r < first) continue;
        const o = across(t, k);
        if (!o) continue;
        const c = tris[t][(k + 2) % 3], d = o.d;
        const ring = (v: number) => {
          const n = new Set<number>();
          for (const u of around.get(v)!) for (const x of tris[u]) if (x !== v) n.add(x);
          return n;
        };
        const nq = ring(q);
        let ok = true;
        for (const x of ring(r)) {
          if (x === q || x === c || x === d) continue;
          if (nq.has(x) || connected(q, x) || len(q, x) > (4 / 3) * h) ok = false;
        }
        if (!ok) continue;
        const moved = [...around.get(r)!].filter((u) => u !== t && u !== o.u);
        for (const u of moved) {
          const tri = tris[u], after = tri.map((v) => (v === r ? q : v));
          const n0 = cross(sub(P.data, tri[1], tri[0]), sub(P.data, tri[2], tri[0]));
          const n1 = cross(sub(P.data, after[1], after[0]), sub(P.data, after[2], after[0]));
          if (dot(n0, n1) <= 0) ok = false;
        }
        if (!ok) continue;
        unlink(t);
        unlink(o.u);
        tris[t] = [];
        tris[o.u] = [];
        for (const u of moved) {
          unlink(u);
          tris[u] = tris[u].map((v) => (v === r ? q : v));
          link(u);
        }
      }
    // Valence flips (towards 6 around new vertices): ear clipping zips round
    // holes into fans whose hubs no angle criterion undoes.
    const deg = new Map<number, number>();
    for (const t of tris) for (const v of t) deg.set(v, (deg.get(v) ?? 0) + 1);
    const off = (v: number, change: number) => (v < first ? 0 : Math.abs(deg.get(v)! + change - 6));
    for (let pass = 0; pass < 50; pass++) {
      let flipped = 0;
      for (let t = 0; t < tris.length; t++)
        for (let k = 0; k < 3 && tris[t].length; k++) {
          const o = across(t, k);
          if (!o) continue;
          const a = tris[t][k], b = tris[t][(k + 1) % 3], c = tris[t][(k + 2) % 3], d = o.d;
          const gain = off(a, 0) + off(b, 0) + off(c, 0) + off(d, 0) - (off(a, -1) + off(b, -1) + off(c, 1) + off(d, 1));
          if (gain <= 0 || c === d || edges.has(edgeKey(c, d)) || connected(c, d)) continue;
          // only across a convex quad: both new triangles face like the old pair
          const n0 = cross(sub(P.data, b, a), sub(P.data, c, a)), n1 = cross(sub(P.data, a, b), sub(P.data, d, b));
          const nq = [n0[0] + n1[0], n0[1] + n1[1], n0[2] + n1[2]];
          if (dot(cross(sub(P.data, d, a), sub(P.data, c, a)), nq) <= 0 || dot(cross(sub(P.data, b, d), sub(P.data, c, d)), nq) <= 0) continue;
          unlink(t);
          unlink(o.u);
          tris[t] = [a, d, c];
          tris[o.u] = [d, b, c];
          link(t);
          link(o.u);
          for (const [v, x] of [[a, -1], [b, -1], [c, 1], [d, 1]]) deg.set(v, deg.get(v)! + x);
          flipped++;
          break;
        }
      if (!flipped) break;
    }
    // Tangential relaxation: new vertices to their neighbours' mean (the
    // positions are provisional, fairing sets the final ones).
    const nbrs = new Map<number, Set<number>>();
    for (const t of tris)
      for (let k = 0; k < t.length; k++) {
        const v = t[k];
        if (v < first) continue;
        const s = nbrs.get(v) ?? new Set<number>();
        s.add(t[(k + 1) % 3]).add(t[(k + 2) % 3]);
        nbrs.set(v, s);
      }
    for (let it = 0; it < 5; it++)
      for (const [v, s] of nbrs) {
        const m = [0, 0, 0];
        for (const u of s) for (let c = 0; c < 3; c++) m[c] += P.data[3 * u + c] / s.size;
        P.data.set(m, 3 * v);
      }
    if (!split) break;
  }
  const live = tris.filter((t) => t.length);
  tris.length = 0;
  tris.push(...live);
}

/** Biharmonic fairing of one patch's new vertices [first, end): L² x = 0 on
 *  them (uniform Laplacian) with the surrounding two rings fixed, so their
 *  curvature carries in; solved in the symmetric form K D⁻¹ K by
 *  Jacobi-preconditioned conjugate gradients from the refined patch. The
 *  patch's edges match its border's, as the uniform Laplacian assumes. */
function fair(P: Float64Array, tris: number[][], first: number, end: number, nb: Rows, nv0: number) {
  const nu = end - first;
  if (!nu) return;
  const patchAdj = new Map<number, Set<number>>();
  for (const t of tris)
    for (let k = 0; k < 3; k++) {
      const a = t[k], b = t[(k + 1) % 3];
      for (const [x, y] of [[a, b], [b, a]]) {
        const s = patchAdj.get(x);
        if (s) s.add(y);
        else patchAdj.set(x, new Set([y]));
      }
    }
  const neighbours = (v: number) => {
    const s = new Set(patchAdj.get(v));
    if (v < nv0) for (let j = nb.start[v]; j < nb.start[v + 1]; j++) s.add(nb.items[j]);
    return [...s];
  };
  // local numbering: unknowns first, then the other rows, then their neighbours
  const ids: number[] = [];
  const loc = new Map<number, number>();
  const add = (g: number) => {
    if (!loc.has(g)) loc.set(g, ids.push(g) - 1);
    return loc.get(g)!;
  };
  for (let v = first; v < end; v++) add(v);
  for (let v = first; v < end; v++) for (const w of neighbours(v)) add(w);
  const nr = ids.length;
  const rowN: number[][] = [];
  for (let i = 0; i < nr; i++) rowN.push(neighbours(ids[i]).map(add));
  const deg = rowN.map((n) => n.length || 1);
  const diag = new Float64Array(nu);
  for (let u = 0; u < nu; u++) diag[u] = deg[u] + rowN[u].reduce((s, i) => s + 1 / deg[i], 0);

  const full = new Float64Array(ids.length), r = new Float64Array(nr);
  /** out = K D⁻¹ K `full` (K = D − A, the uniform graph Laplacian; zero rows
   *  of this are L² x = 0 for L = D⁻¹K), restricted to the unknowns. */
  const apply = (out: Float64Array) => {
    for (let i = 0; i < nr; i++) {
      let s = deg[i] * full[i];
      for (const j of rowN[i]) s -= full[j];
      r[i] = s;
    }
    for (let u = 0; u < nu; u++) {
      let s = r[u];
      for (const i of rowN[u]) s -= r[i] / deg[i];
      out[u] = s;
    }
  };
  const b = new Float64Array(nu), x = new Float64Array(nu), res = new Float64Array(nu);
  const z = new Float64Array(nu), p = new Float64Array(nu), Ap = new Float64Array(nu);
  for (let c = 0; c < 3; c++) {
    // b = −LᵀL x_known
    full.fill(0);
    for (let i = nu; i < ids.length; i++) full[i] = P[3 * ids[i] + c];
    apply(b);
    full.fill(0);
    for (let u = 0; u < nu; u++) (b[u] = -b[u]), (full[u] = x[u] = P[3 * ids[u] + c]);
    apply(Ap);
    let bn = 0, rz = 0;
    for (let u = 0; u < nu; u++) {
      res[u] = b[u] - Ap[u];
      z[u] = res[u] / diag[u];
      p[u] = z[u];
      bn += b[u] * b[u];
      rz += res[u] * z[u];
    }
    const tol = 1e-16 * Math.max(bn, 1e-30);
    for (let it = 0; it < 10000; it++) {
      let rr = 0;
      for (let u = 0; u < nu; u++) rr += res[u] * res[u];
      if (rr <= tol) break;
      full.set(p);
      apply(Ap);
      let pAp = 0;
      for (let u = 0; u < nu; u++) pAp += p[u] * Ap[u];
      if (!(pAp > 0)) break;
      const alpha = rz / pAp;
      let rz2 = 0;
      for (let u = 0; u < nu; u++) {
        x[u] += alpha * p[u];
        res[u] -= alpha * Ap[u];
        z[u] = res[u] / diag[u];
        rz2 += res[u] * z[u];
      }
      const beta = rz2 / rz;
      rz = rz2;
      for (let u = 0; u < nu; u++) p[u] = z[u] + beta * p[u];
    }
    for (let u = 0; u < nu; u++) P[3 * ids[u] + c] = x[u];
  }
}
