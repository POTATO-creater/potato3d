// Bounding-volume hierarchy over a mesh's triangles, for ray casts.

const LEAF = 4;

export class Bvh {
  readonly P: Float32Array;
  readonly I: Uint32Array;
  /** Triangles in leaf order. */
  readonly order: Uint32Array;
  /** min xyz, max xyz per node */
  readonly box: Float32Array;
  /** Inner node: index of the first child (the second follows it). Leaf: first triangle in `order`. */
  readonly first: Uint32Array;
  /** Triangles in a leaf, 0 for inner nodes. */
  readonly count: Uint32Array;

  constructor(P: Float32Array, I: Uint32Array) {
    this.P = P;
    this.I = I;
    const nt = I.length / 3;
    const cen = new Float32Array(3 * nt);
    for (let t = 0; t < nt; t++)
      for (let k = 0; k < 3; k++) cen[3 * t + k] = (P[3 * I[3 * t] + k] + P[3 * I[3 * t + 1] + k] + P[3 * I[3 * t + 2] + k]) / 3;
    this.order = new Uint32Array(nt).map((_, i) => i);
    const cap = 2 * nt + 1;
    this.box = new Float32Array(6 * cap);
    this.first = new Uint32Array(cap);
    this.count = new Uint32Array(cap);
    let nodes = 1;
    const stack: [number, number, number][] = [[0, 0, nt]];
    while (stack.length) {
      const [node, a, b] = stack.pop()!;
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      const clo = [Infinity, Infinity, Infinity], chi = [-Infinity, -Infinity, -Infinity];
      for (let i = a; i < b; i++) {
        const t = this.order[i];
        for (let c = 0; c < 3; c++) {
          const v = 3 * I[3 * t + c];
          for (let k = 0; k < 3; k++) (lo[k] = Math.min(lo[k], P[v + k])), (hi[k] = Math.max(hi[k], P[v + k]));
        }
        for (let k = 0; k < 3; k++) (clo[k] = Math.min(clo[k], cen[3 * t + k])), (chi[k] = Math.max(chi[k], cen[3 * t + k]));
      }
      this.box.set([...lo, ...hi], 6 * node);
      if (b - a <= LEAF) {
        this.first[node] = a;
        this.count[node] = b - a;
        continue;
      }
      const ext = [0, 1, 2].map((k) => chi[k] - clo[k]);
      const axis = ext.indexOf(Math.max(...ext));
      this.order.subarray(a, b).sort((x, y) => cen[3 * x + axis] - cen[3 * y + axis]);
      const mid = (a + b) >> 1;
      this.first[node] = nodes;
      stack.push([nodes, a, mid], [nodes + 1, mid, b]);
      nodes += 2;
    }
  }

  /** Nearest hit along o + t·d (t in (0, tMax)), either side of the
   *  triangles; `skip(tri)` ignores triangles. Infinity when none. */
  cast(o: ArrayLike<number>, d: ArrayLike<number>, tMax = Infinity, skip?: (tri: number) => boolean): { t: number; tri: number } {
    const { P, I, box } = this;
    const inv = [1 / d[0], 1 / d[1], 1 / d[2]];
    let best = tMax, hit = -1;
    const stack = [0];
    while (stack.length) {
      const n = stack.pop()!;
      // slab test
      let t0 = 0, t1 = best;
      for (let k = 0; k < 3; k++) {
        let a = (box[6 * n + k] - o[k]) * inv[k], b = (box[6 * n + 3 + k] - o[k]) * inv[k];
        if (a > b) [a, b] = [b, a];
        t0 = Math.max(t0, a);
        t1 = Math.min(t1, b);
      }
      if (!(t0 <= t1)) continue;
      if (!this.count[n]) {
        stack.push(this.first[n], this.first[n] + 1);
        continue;
      }
      for (let i = this.first[n]; i < this.first[n] + this.count[n]; i++) {
        const tri = this.order[i];
        if (skip?.(tri)) continue;
        // Möller–Trumbore
        const a = 3 * I[3 * tri], b = 3 * I[3 * tri + 1], c = 3 * I[3 * tri + 2];
        const e1 = [P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]];
        const e2 = [P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]];
        const p = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
        const det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
        if (Math.abs(det) < 1e-20) continue;
        const s = [o[0] - P[a], o[1] - P[a + 1], o[2] - P[a + 2]];
        const u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) / det;
        if (u < 0 || u > 1) continue;
        const q = [s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]];
        const v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det;
        if (v < 0 || u + v > 1) continue;
        const t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det;
        if (t > 0 && t < best) (best = t), (hit = tri);
      }
    }
    return { t: hit < 0 ? Infinity : best, tri: hit };
  }
}
