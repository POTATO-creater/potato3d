// Surface extraction from the sparse TSDF: marching tetrahedra (each lattice
// cube split into 6 tetrahedra around its main diagonal — consistent between
// neighbouring cubes, so the surface is closed wherever the volume is
// observed), vertices shared through an edge hash, colour interpolated along
// the edges. Then: keep the large connected components.

import { BLOCK, blockKey, type Volume } from "./tsdf.ts";

export interface Mesh {
  /** xyz per vertex */
  positions: Float32Array;
  /** rgb per vertex, 0–1, sRGB (as in the photos) */
  colors: Float32Array;
  /** 3 indices per triangle, counter-clockwise seen from outside */
  indices: Uint32Array;
}

const F16 = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

/** Corners of a lattice cube: index = x + 2y + 4z. */
const CORNER = [0, 1, 2, 3, 4, 5, 6, 7].map((k) => [k & 1, (k >> 1) & 1, (k >> 2) & 1]);
/** 6 tetrahedra around the 0–7 diagonal. */
const TETS = [
  [0, 1, 3, 7],
  [0, 3, 2, 7],
  [0, 2, 6, 7],
  [0, 6, 4, 7],
  [0, 4, 5, 7],
  [0, 5, 1, 7],
];
/** Edge directions between tetrahedron corners, ordered so all components are ≥ 0. */
const DIRS = [
  [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
];
const dirIndex = (dx: number, dy: number, dz: number) => DIRS.findIndex((d) => d[0] === dx && d[1] === dy && d[2] === dz);

class Grow {
  data: Float32Array;
  n = 0;
  constructor(cap: number) {
    this.data = new Float32Array(cap);
  }
  push3(a: number, b: number, c: number) {
    if (this.n + 3 > this.data.length) {
      const d = new Float32Array(this.data.length * 2);
      d.set(this.data);
      this.data = d;
    }
    this.data[this.n++] = a;
    this.data[this.n++] = b;
    this.data[this.n++] = c;
  }
}
class GrowU32 {
  data: Uint32Array;
  n = 0;
  constructor(cap: number) {
    this.data = new Uint32Array(cap);
  }
  push(a: number) {
    if (this.n + 1 > this.data.length) {
      const d = new Uint32Array(this.data.length * 2);
      d.set(this.data);
      this.data = d;
    }
    this.data[this.n++] = a;
  }
}

/** Marching tetrahedra on the TSDF. Cubes with any corner seen fewer than
 *  `minWeight` times (or not allocated) are skipped. */
export function extractSurface(vol: Volume, minWeight = 2): Mesh {
  const B = BLOCK, BV = B * B * B;
  const nb = vol.blocks.length / 3;
  // Block lookup and the lattice bounding box (for edge keys).
  const bkey = blockKey;
  const index = new Map<number, number>();
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let b = 0; b < nb; b++) {
    const c = [vol.blocks[3 * b], vol.blocks[3 * b + 1], vol.blocks[3 * b + 2]];
    index.set(bkey(c[0], c[1], c[2]), b);
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], c[k]);
      max[k] = Math.max(max[k], c[k]);
    }
  }
  const nx = (max[0] - min[0] + 2) * B, ny = (max[1] - min[1] + 2) * B;
  const lin = (x: number, y: number, z: number) => x - min[0] * B + nx * (y - min[1] * B + ny * (z - min[2] * B));

  const pos = new Grow(1 << 20), col = new Grow(1 << 20), idx = new GrowU32(1 << 21);
  const edges = new Map<number, number>();
  const s = new Float32Array(8), w = new Float32Array(8), rgb = new Float32Array(24);
  const gxs = new Int32Array(8), gys = new Int32Array(8), gzs = new Int32Array(8);
  const local = new Int32Array(8); // voxel index (block-major) per corner

  const vertex = (a: number, b: number) => {
    // ordered endpoints: componentwise a ≤ b
    let ia = a, ib = b;
    if (gxs[a] > gxs[b] || gys[a] > gys[b] || gzs[a] > gzs[b]) [ia, ib] = [b, a];
    const k = lin(gxs[ia], gys[ia], gzs[ia]) * 7 + dirIndex(gxs[ib] - gxs[ia], gys[ib] - gys[ia], gzs[ib] - gzs[ia]);
    const found = edges.get(k);
    if (found !== undefined) return found;
    const t = s[ia] / (s[ia] - s[ib]);
    const v = vol.voxel;
    pos.push3(
      (gxs[ia] + t * (gxs[ib] - gxs[ia])) * v,
      (gys[ia] + t * (gys[ib] - gys[ia])) * v,
      (gzs[ia] + t * (gzs[ib] - gzs[ia])) * v,
    );
    col.push3(
      rgb[3 * ia] + t * (rgb[3 * ib] - rgb[3 * ia]),
      rgb[3 * ia + 1] + t * (rgb[3 * ib + 1] - rgb[3 * ia + 1]),
      rgb[3 * ia + 2] + t * (rgb[3 * ib + 2] - rgb[3 * ia + 2]),
    );
    const id = pos.n / 3 - 1;
    edges.set(k, id);
    return id;
  };

  const neighbors = new Int32Array(8);
  for (let b = 0; b < nb; b++) {
    const bx = vol.blocks[3 * b], by = vol.blocks[3 * b + 1], bz = vol.blocks[3 * b + 2];
    for (let k = 0; k < 8; k++) {
      const n = index.get(bkey(bx + CORNER[k][0], by + CORNER[k][1], bz + CORNER[k][2]));
      neighbors[k] = n === undefined ? -1 : n;
    }
    for (let lz = 0; lz < B; lz++) {
      for (let ly = 0; ly < B; ly++) {
        for (let lx = 0; lx < B; lx++) {
          let ok = true, neg = 0;
          for (let k = 0; k < 8 && ok; k++) {
            let x = lx + CORNER[k][0], y = ly + CORNER[k][1], z = lz + CORNER[k][2];
            const nk = (x >= B ? 1 : 0) + (y >= B ? 2 : 0) + (z >= B ? 4 : 0);
            const blk = neighbors[nk];
            if (blk < 0) {
              ok = false;
              break;
            }
            if (x >= B) x -= B;
            if (y >= B) y -= B;
            if (z >= B) z -= B;
            const vi = blk * BV + x + B * (y + B * z);
            const sw = vol.tsdfw[vi];
            w[k] = F16[sw >>> 16];
            if (w[k] < minWeight) ok = false;
            s[k] = F16[sw & 0xffff] || 1e-6; // exact zeros count as outside
            if (s[k] < 0) neg++;
            local[k] = vi;
          }
          if (!ok || neg === 0 || neg === 8) continue;
          for (let k = 0; k < 8; k++) {
            gxs[k] = (bx * B) + lx + CORNER[k][0];
            gys[k] = (by * B) + ly + CORNER[k][1];
            gzs[k] = (bz * B) + lz + CORNER[k][2];
            const c = vol.color[local[k]];
            rgb[3 * k] = (c & 255) / 255;
            rgb[3 * k + 1] = ((c >>> 8) & 255) / 255;
            rgb[3 * k + 2] = ((c >>> 16) & 255) / 255;
          }
          for (const tet of TETS) emitTet(tet);
        }
      }
    }
  }

  function emitTet(t: number[]) {
    const ins: number[] = [], outs: number[] = [];
    for (const k of t) (s[k] < 0 ? ins : outs).push(k);
    if (ins.length === 0 || outs.length === 0) return;
    // Polygons as lists of lattice edges (corner pairs).
    let polys: [number, number][][];
    if (ins.length === 1 || outs.length === 1) {
      const [lone, others] = ins.length === 1 ? [ins[0], outs] : [outs[0], ins];
      polys = [others.map((o) => [lone, o] as [number, number])];
    } else {
      // quad: i0-o0, i0-o1, i1-o1, i1-o0 (a closed loop around the tetrahedron)
      const [i0, i1] = ins, [o0, o1] = outs;
      const q: [number, number][] = [[i0, o0], [i0, o1], [i1, o1], [i1, o0]];
      polys = [[q[0], q[1], q[2]], [q[0], q[2], q[3]]];
    }
    // Outward = from the inside corners towards the outside ones. The winding
    // is decided on the lattice-edge midpoints, which never degenerate (the
    // interpolated vertices can, when the surface passes through a corner).
    let gx = 0, gy = 0, gz = 0;
    for (const k of outs) (gx += gxs[k] / outs.length), (gy += gys[k] / outs.length), (gz += gzs[k] / outs.length);
    for (const k of ins) (gx -= gxs[k] / ins.length), (gy -= gys[k] / ins.length), (gz -= gzs[k] / ins.length);
    const mid = ([a, b]: [number, number]) => [gxs[a] + gxs[b], gys[a] + gys[b], gzs[a] + gzs[b]];
    for (const tri of polys) {
      const [pa, pb, pc] = tri.map(mid);
      const ux = pb[0] - pa[0], uy = pb[1] - pa[1], uz = pb[2] - pa[2];
      const vx = pc[0] - pa[0], vy = pc[1] - pa[1], vz = pc[2] - pa[2];
      const flip = (uy * vz - uz * vy) * gx + (uz * vx - ux * vz) * gy + (ux * vy - uy * vx) * gz < 0;
      const [a, b, c] = tri.map(([i, j]) => vertex(i, j));
      if (a === b || b === c || a === c) continue;
      if (flip) (idx.push(a), idx.push(c), idx.push(b));
      else (idx.push(a), idx.push(b), idx.push(c));
    }
  }

  return {
    positions: pos.data.slice(0, pos.n),
    colors: col.data.slice(0, col.n),
    indices: idx.data.slice(0, idx.n),
  };
}

/** Keeps the connected components with at least `minFraction` of the largest
 *  one's triangles (floaters and stray bits go), and drops unused vertices. */
export function keepLargeComponents(m: Mesh, minFraction = 0.01): Mesh {
  const nv = m.positions.length / 3, nt = m.indices.length / 3;
  const parent = new Int32Array(nv).map((_, i) => i);
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  for (let t = 0; t < nt; t++) {
    const a = find(m.indices[3 * t]), b = find(m.indices[3 * t + 1]), c = find(m.indices[3 * t + 2]);
    parent[b] = a;
    parent[find(c)] = a;
  }
  const size = new Map<number, number>();
  for (let t = 0; t < nt; t++) {
    const r = find(m.indices[3 * t]);
    size.set(r, (size.get(r) ?? 0) + 1);
  }
  const largest = Math.max(0, ...size.values());
  const keep = (t: number) => (size.get(find(m.indices[3 * t])) ?? 0) >= minFraction * largest;
  const remap = new Int32Array(nv).fill(-1);
  const pos: number[] = [], col: number[] = [], idx: number[] = [];
  for (let t = 0; t < nt; t++) {
    if (!keep(t)) continue;
    for (let k = 0; k < 3; k++) {
      const v = m.indices[3 * t + k];
      if (remap[v] < 0) {
        remap[v] = pos.length / 3;
        pos.push(m.positions[3 * v], m.positions[3 * v + 1], m.positions[3 * v + 2]);
        col.push(m.colors[3 * v], m.colors[3 * v + 1], m.colors[3 * v + 2]);
      }
      idx.push(remap[v]);
    }
  }
  return { positions: new Float32Array(pos), colors: new Float32Array(col), indices: new Uint32Array(idx) };
}

/** Area-weighted vertex normals. */
export function vertexNormals(m: Mesh): Float32Array {
  const P = m.positions, I = m.indices;
  const n = new Float32Array(P.length);
  for (let t = 0; t < I.length; t += 3) {
    const a = 3 * I[t], b = 3 * I[t + 1], c = 3 * I[t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const o of [a, b, c]) (n[o] += nx), (n[o + 1] += ny), (n[o + 2] += nz);
  }
  for (let i = 0; i < n.length; i += 3) {
    const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1;
    n[i] /= l;
    n[i + 1] /= l;
    n[i + 2] /= l;
  }
  return n;
}
