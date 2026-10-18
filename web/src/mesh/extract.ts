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
/** Index in DIRS of the direction dx + 2·dy + 4·dz. */
const DIR_OF = (() => {
  const t = new Int8Array(8).fill(-1);
  DIRS.forEach(([x, y, z], i) => (t[x + 2 * y + 4 * z] = i));
  return t;
})();
// Flat tables for the hot loops (no nested arrays).
const CX = Int32Array.from(CORNER, (c) => c[0]), CY = Int32Array.from(CORNER, (c) => c[1]), CZ = Int32Array.from(CORNER, (c) => c[2]);
const TET = Int8Array.from(TETS.flat());

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
  const { tsdfw, color, voxel } = vol;
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
  // Scratch for emitTet (it runs millions of times: no allocations).
  const ins = new Int32Array(4), outs = new Int32Array(4);
  /** Up to two triangles, each as three lattice edges (corner pairs). */
  const E = new Int32Array(12);

  const vertex = (a: number, b: number) => {
    // ordered endpoints: componentwise a ≤ b
    let ia = a, ib = b;
    if (gxs[a] > gxs[b] || gys[a] > gys[b] || gzs[a] > gzs[b]) (ia = b), (ib = a);
    const k = lin(gxs[ia], gys[ia], gzs[ia]) * 7 + DIR_OF[gxs[ib] - gxs[ia] + 2 * (gys[ib] - gys[ia]) + 4 * (gzs[ib] - gzs[ia])];
    const found = edges.get(k);
    if (found !== undefined) return found;
    const t = s[ia] / (s[ia] - s[ib]);
    const v = voxel;
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
      const n = index.get(bkey(bx + CX[k], by + CY[k], bz + CZ[k]));
      neighbors[k] = n === undefined ? -1 : n;
    }
    for (let lz = 0; lz < B; lz++) {
      for (let ly = 0; ly < B; ly++) {
        for (let lx = 0; lx < B; lx++) {
          let ok = true, neg = 0;
          for (let k = 0; k < 8 && ok; k++) {
            let x = lx + CX[k], y = ly + CY[k], z = lz + CZ[k];
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
            const sw = tsdfw[vi];
            w[k] = F16[sw >>> 16];
            if (w[k] < minWeight) ok = false;
            s[k] = F16[sw & 0xffff] || 1e-6; // exact zeros count as outside
            if (s[k] < 0) neg++;
            local[k] = vi;
          }
          if (!ok || neg === 0 || neg === 8) continue;
          for (let k = 0; k < 8; k++) {
            gxs[k] = (bx * B) + lx + CX[k];
            gys[k] = (by * B) + ly + CY[k];
            gzs[k] = (bz * B) + lz + CZ[k];
            const c = color[local[k]];
            rgb[3 * k] = (c & 255) / 255;
            rgb[3 * k + 1] = ((c >>> 8) & 255) / 255;
            rgb[3 * k + 2] = ((c >>> 16) & 255) / 255;
          }
          for (let q = 0; q < 6; q++) emitTet(q);
        }
      }
    }
  }

  function emitTet(q: number) {
    let ni = 0, no = 0;
    for (let c = 4 * q; c < 4 * q + 4; c++) {
      const k = TET[c];
      if (s[k] < 0) ins[ni++] = k;
      else outs[no++] = k;
    }
    if (ni === 0 || no === 0) return;
    let ntri = 1;
    if (ni === 1 || no === 1) {
      const lone = ni === 1 ? ins[0] : outs[0], others = ni === 1 ? outs : ins;
      for (let e = 0; e < 3; e++) (E[2 * e] = lone), (E[2 * e + 1] = others[e]);
    } else {
      // quad: i0-o0, i0-o1, i1-o1, i1-o0 (a closed loop around the tetrahedron)
      const i0 = ins[0], i1 = ins[1], o0 = outs[0], o1 = outs[1];
      (E[0] = i0), (E[1] = o0), (E[2] = i0), (E[3] = o1), (E[4] = i1), (E[5] = o1);
      (E[6] = i0), (E[7] = o0), (E[8] = i1), (E[9] = o1), (E[10] = i1), (E[11] = o0);
      ntri = 2;
    }
    // Outward = from the inside corners towards the outside ones. The winding
    // is decided on the lattice-edge midpoints, which never degenerate (the
    // interpolated vertices can, when the surface passes through a corner).
    let gx = 0, gy = 0, gz = 0;
    for (let c = 0; c < no; c++) {
      const k = outs[c];
      (gx += gxs[k] / no), (gy += gys[k] / no), (gz += gzs[k] / no);
    }
    for (let c = 0; c < ni; c++) {
      const k = ins[c];
      (gx -= gxs[k] / ni), (gy -= gys[k] / ni), (gz -= gzs[k] / ni);
    }
    for (let tri = 0; tri < ntri; tri++) {
      const e = 6 * tri;
      // edge midpoints (×2)
      const ax = gxs[E[e]] + gxs[E[e + 1]], ay = gys[E[e]] + gys[E[e + 1]], az = gzs[E[e]] + gzs[E[e + 1]];
      const ux = gxs[E[e + 2]] + gxs[E[e + 3]] - ax, uy = gys[E[e + 2]] + gys[E[e + 3]] - ay, uz = gzs[E[e + 2]] + gzs[E[e + 3]] - az;
      const vx = gxs[E[e + 4]] + gxs[E[e + 5]] - ax, vy = gys[E[e + 4]] + gys[E[e + 5]] - ay, vz = gzs[E[e + 4]] + gzs[E[e + 5]] - az;
      const flip = (uy * vz - uz * vy) * gx + (uz * vx - ux * vz) * gy + (ux * vy - uy * vx) * gz < 0;
      const a = vertex(E[e], E[e + 1]), b = vertex(E[e + 2], E[e + 3]), c = vertex(E[e + 4], E[e + 5]);
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
