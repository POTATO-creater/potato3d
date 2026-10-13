// SOG v2 (PlayCanvas) writer: an uncompressed ZIP with `meta.json` and
// lossless WebP textures. Port of src-tauri/src/export/sog.rs (which came from
// WebGaussian's sog-encode.ts, following splat-transform's `write-sog.ts`):
// Morton order, dynamic-programming 1D quantizer (256-entry codebooks),
// log-transformed 16-bit positions, "smallest three" quaternions and opacity
// in the `sh0` alpha. SH bands are stored as in splat-transform: a palette of
// up to 64k vectors (two-level k-means), quantized into a 256-entry codebook,
// plus a 16-bit label per Gaussian.

import type { Msg } from "../../../src/lib/api";
import { restLen, shCoeffs, type Splats } from "./splats.ts";
import { zipStore } from "./zip.ts";

const KMEANS_ITERATIONS = 8;
const COARSE_CLUSTERS = 256;
const COARSE_SAMPLE = 65_536;

/** RGBA → lossless WebP (straight alpha). */
export type WebpEncoder = (rgba: Uint8Array, width: number, height: number) => Uint8Array;

export function writeSog(s: Splats, webp: WebpEncoder, generator: string, progress: (f: number, msg: Msg) => void): Uint8Array {
  const n = s.count;
  if (n === 0) throw new Error("no Gaussians to export");
  const width = Math.max(Math.ceil(Math.sqrt(n) / 4) * 4, 4);
  const height = Math.max(Math.ceil(n / width / 4) * 4, 4);
  if (width > 16383 || height > 16383) throw new Error(`too many Gaussians for SOG (${n})`);
  const texels = width * height;

  progress(0, { key: "export.sorting" });
  const order = mortonOrder(s.pos);

  // ---- quantize scales and colors (256-entry codebooks)
  progress(0.1, { key: "export.quantizing" });
  const column = (src: Float32Array, stride: number, a: number) => {
    const c = new Float32Array(src.length / stride);
    for (let i = 0; i < c.length; i++) c[i] = src[i * stride + a];
    return c;
  };
  const scalesQ = quantize1d([0, 1, 2].map((a) => column(s.scale, 3, a)), 256);
  const sh0Q = quantize1d([0, 1, 2].map((a) => column(s.dc, 3, a)), 256);

  // ---- SH palette
  let sh: ShPalette | null = null;
  if (s.shDegree > 0) {
    progress(0.25, { key: "export.clustering" });
    sh = encodeSh(s);
  }

  progress(0.6, { key: "export.textures" });
  // ---- means: log, 16 bits split over two textures
  const logT = (v: number) => Math.sign(v) * Math.log(Math.abs(v) + 1);
  const mins = [Infinity, Infinity, Infinity], maxs = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++) {
      const v = logT(s.pos[i * 3 + a]);
      if (v < mins[a]) mins[a] = v;
      if (v > maxs[a]) maxs[a] = v;
    }
  const meansL = new Uint8Array(texels * 4);
  const meansU = new Uint8Array(texels * 4);
  const quats = new Uint8Array(texels * 4);
  const scales = new Uint8Array(texels * 4);
  const sh0 = new Uint8Array(texels * 4);
  const shLabels = sh ? new Uint8Array(texels * 4) : null;
  for (let t = 0; t < n; t++) {
    const g = order[t], o = t * 4;
    for (let a = 0; a < 3; a++) {
      const range = maxs[a] - mins[a];
      const v = Math.trunc((65535 * (logT(s.pos[g * 3 + a]) - mins[a])) / (range === 0 ? 1 : range));
      meansL[o + a] = v & 0xff;
      meansU[o + a] = (v >> 8) & 0xff;
    }
    meansL[o + 3] = meansU[o + 3] = 0xff;
    packQuat(s.rot, g * 4, quats, o);
    for (let a = 0; a < 3; a++) {
      scales[o + a] = scalesQ.labels[a][g];
      sh0[o + a] = sh0Q.labels[a][g];
    }
    scales[o + 3] = 0xff;
    sh0[o + 3] = Math.trunc(Math.min(Math.max((1 / (1 + Math.exp(-s.opacity[g]))) * 255, 0), 255));
    if (sh && shLabels) {
      const label = sh.labels[g];
      shLabels[o] = label & 0xff;
      shLabels[o + 1] = label >> 8;
      shLabels[o + 3] = 0xff;
    }
  }

  const textures: [string, Uint8Array, number, number][] = [
    ["means_l.webp", meansL, width, height],
    ["means_u.webp", meansU, width, height],
    ["quats.webp", quats, width, height],
    ["scales.webp", scales, width, height],
    ["sh0.webp", sh0, width, height],
  ];
  const meta: Record<string, unknown> = {
    version: 2,
    asset: { generator },
    count: n,
    means: { mins, maxs, files: ["means_l.webp", "means_u.webp"] },
    scales: { codebook: Array.from(scalesQ.centroids), files: ["scales.webp"] },
    quats: { files: ["quats.webp"] },
    sh0: { codebook: Array.from(sh0Q.centroids), files: ["sh0.webp"] },
  };
  if (sh && shLabels) {
    const coeffs = shCoeffs(s.shDegree);
    textures.push(["shN_centroids.webp", sh.centroidsRgba, 64 * coeffs, Math.ceil(sh.count / 64)]);
    textures.push(["shN_labels.webp", shLabels, width, height]);
    meta.shN = {
      count: sh.count,
      bands: s.shDegree,
      codebook: Array.from(sh.codebook),
      files: ["shN_centroids.webp", "shN_labels.webp"],
    };
  }

  const files: [string, Uint8Array][] = [];
  for (let k = 0; k < textures.length; k++) {
    const [name, rgba, w, h] = textures[k];
    progress(0.7 + (0.25 * k) / textures.length, { key: "export.compressing" });
    files.push([name, webp(rgba, w, h)]);
  }
  progress(0.95, { key: "export.writingSog" });
  files.push(["meta.json", new TextEncoder().encode(JSON.stringify(meta))]);
  return zipStore(files);
}

/** Normalized quaternion in 4 bytes: the three smallest components in RGB and
 *  the index of the largest (dropped) one in A (252 + i). */
export function packQuat(rot: ArrayLike<number>, ro: number, out: Uint8Array, o: number) {
  const q = [rot[ro], rot[ro + 1], rot[ro + 2], rot[ro + 3]];
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  let max = 0;
  for (let i = 0; i < 4; i++) {
    q[i] /= l;
    if (Math.abs(q[i]) > Math.abs(q[max])) max = i;
  }
  const sign = (q[max] < 0 ? -1 : 1) * Math.SQRT2;
  const idx = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]][max];
  for (let k = 0; k < 3; k++) out[o + k] = Math.trunc(Math.min(Math.max(255 * (q[idx[k]] * sign * 0.5 + 0.5), 0), 255));
  out[o + 3] = 252 + max;
}

// ---------------------------------------------------------------------------
// Morton order (spatial locality → WebP compresses better).
// ---------------------------------------------------------------------------

function mortonOrder(pos: Float32Array): Uint32Array {
  const idx = new Uint32Array(pos.length / 3).map((_, i) => i);
  sortMorton(pos, idx);
  return idx;
}

function part1by2(v: number) {
  v &= 0x3ff;
  v = (v ^ (v << 16)) & 0xff0000ff;
  v = (v ^ (v << 8)) & 0x0300f00f;
  v = (v ^ (v << 4)) & 0x030c30c3;
  v = (v ^ (v << 2)) & 0x09249249;
  return v >>> 0;
}

/** 10-bit codes per axis; groups sharing a code with more than 256
 *  elements are refined recursively. */
function sortMorton(pos: Float32Array, range: Uint32Array) {
  if (!range.length) return;
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (const i of range)
    for (let a = 0; a < 3; a++) {
      const v = pos[i * 3 + a];
      if (v < mn[a]) mn[a] = v;
      if (v > mx[a]) mx[a] = v;
    }
  const len = [0, 1, 2].map((a) => mx[a] - mn[a]);
  if (len.some((l) => !Number.isFinite(l)) || len.every((l) => l === 0)) return;
  // In f32, as in Rust: points on a cell boundary land in the same cell.
  const f = Math.fround;
  const mul = len.map((l) => (l === 0 ? 0 : f(1024 / f(l))));
  const codes = new Uint32Array(range.length);
  for (let k = 0; k < range.length; k++) {
    const i = range[k];
    const c = [0, 1, 2].map((a) => Math.trunc(Math.min(f(f(pos[i * 3 + a] - mn[a]) * mul[a]), 1023)));
    codes[k] = ((part1by2(c[2]) << 2) + (part1by2(c[1]) << 1) + part1by2(c[0])) >>> 0;
  }
  const order = new Uint32Array(range.length).map((_, k) => k);
  order.sort((a, b) => codes[a] - codes[b]);
  const src = range.slice();
  for (let k = 0; k < range.length; k++) range[k] = src[order[k]];
  for (let start = 0; start < order.length; ) {
    let end = start + 1;
    while (end < order.length && codes[order[end]] === codes[order[start]]) end++;
    if (end - start > 256) sortMorton(pos, range.subarray(start, end));
    start = end;
  }
}

// ---------------------------------------------------------------------------
// Optimal 1D quantizer (dynamic programming over a histogram), port of
// `quantize1dColumns` (k = 256, alpha = 0.5). The columns share one codebook.
// ---------------------------------------------------------------------------

export interface Quantized {
  centroids: Float32Array;
  labels: Uint8Array[];
}

export function quantize1d(columns: Float32Array[], k: number): Quantized {
  const rows = columns[0]?.length ?? 0;
  const n = rows * columns.length;
  const empty = () => columns.map(() => new Uint8Array(rows));
  if (n === 0) return { centroids: new Float32Array(k), labels: empty() };
  const sorted = new Float32Array(n);
  columns.forEach((c, i) => sorted.set(c, i * rows));
  sorted.sort();
  const vmin = sorted[0], vmax = sorted[n - 1];
  if (vmax - vmin < 1e-20) return { centroids: new Float32Array(k).fill(vmin), labels: empty() };

  const h = Math.min(n, 1024);
  const range = vmax - vmin;
  const iqr = sorted[Math.floor((n * 3) / 4)] - sorted[Math.floor(n / 4)];
  const beta = Math.min(Math.max(1 - iqr / range, 0.5), 0.999);
  const counts = new Float64Array(h), sums = new Float64Array(h);
  for (let i = 0; i < n; i++) {
    const v = sorted[i];
    const bin = Math.min(Math.floor(h * (beta * (i / n) + (1 - beta) * ((v - vmin) / range))), h - 1);
    counts[bin]++;
    sums[bin] += v;
  }
  const centers = new Float64Array(h).map((_, i) => (counts[i] > 0 ? sums[i] / counts[i] : vmin + ((i + 0.5) / h) * range));
  const pw = new Float64Array(h + 1), pwx = new Float64Array(h + 1), pwxx = new Float64Array(h + 1);
  for (let i = 0; i < h; i++) {
    const w = counts[i] > 0 ? Math.sqrt(counts[i]) : 0;
    pw[i + 1] = pw[i] + w;
    pwx[i + 1] = pwx[i] + w * centers[i];
    pwxx[i + 1] = pwxx[i] + w * centers[i] * centers[i];
  }
  const cost = (a: number, b: number) => {
    const w = pw[b + 1] - pw[a];
    if (w <= 0) return 0;
    const wx = pwx[b + 1] - pwx[a];
    return pwxx[b + 1] - pwxx[a] - (wx * wx) / w;
  };
  const mean = (a: number, b: number) => {
    const w = pw[b + 1] - pw[a];
    return w <= 0 ? (centers[a] + centers[b]) * 0.5 : (pwx[b + 1] - pwx[a]) / w;
  };

  const effK = Math.min(k, counts.reduce((s, c) => s + (c > 0 ? 1 : 0), 0));
  const INF = 1e30;
  let prev = new Float64Array(h).map((_, j) => cost(0, j));
  let curr = new Float64Array(h);
  const split: Int32Array[] = [];
  split[1] = new Int32Array(h).fill(-1);
  for (let m = 2; m <= effK; m++) {
    curr.fill(INF);
    const sm = new Int32Array(h);
    for (let j = m - 1; j < h; j++) {
      let best = INF, bestS = m - 2;
      for (let s = m - 2; s < j; s++) {
        const c = prev[s] + cost(s + 1, j);
        if (c < best) {
          best = c;
          bestS = s;
        }
      }
      curr[j] = best;
      sm[j] = bestS;
    }
    split[m] = sm;
    [prev, curr] = [curr, prev];
  }

  const cents = new Float32Array(effK);
  let j = h - 1;
  for (let m = effK; m >= 1; m--) {
    const s = m > 1 ? split[m][j] : -1;
    cents[m - 1] = mean(s + 1, j);
    j = s;
  }
  cents.sort();
  const centroids = new Float32Array(k).fill(cents[effK - 1]);
  centroids.set(cents);

  const label = (v: number) => {
    let lo = 0, hi = k - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      // midpoint in f32, as in Rust
      if (v < Math.fround(centroids[mid] + centroids[mid + 1]) * 0.5) hi = mid;
      else lo = mid + 1;
    }
    return lo;
  };
  return { centroids, labels: columns.map((c) => Uint8Array.from(c, label)) };
}

// ---------------------------------------------------------------------------
// SH bands: palette via two-level k-means.
// ---------------------------------------------------------------------------

interface ShPalette {
  count: number;
  labels: Uint16Array;
  codebook: Float32Array;
  /** `shN_centroids` texture: 64 centroids per row, `coeffs` texels each. */
  centroidsRgba: Uint8Array;
}

function encodeSh(s: Splats): ShPalette {
  const n = s.count, d = restLen(s), coeffs = shCoeffs(s.shDegree);
  // Same palette size as splat-transform.
  const palette = Math.min(Math.max(Math.trunc(Math.min(2 ** Math.floor(Math.log2(n / 1024)), 64) * 1024), 1), Math.min(n, 65_536));
  const { centroids, labels } = kmeansTwoLevel(s.rest, n, d, palette);
  const count = centroids.length / d;

  // Single codebook across all palette columns.
  const cols = Array.from({ length: d }, (_, j) => {
    const c = new Float32Array(count);
    for (let i = 0; i < count; i++) c[i] = centroids[i * d + j];
    return c;
  });
  const q = quantize1d(cols, 256);
  const rows = Math.ceil(count / 64);
  const rgba = new Uint8Array(64 * coeffs * rows * 4);
  for (let i = 0; i < count; i++)
    for (let j = 0; j < coeffs; j++) {
      const o = (i * coeffs + j) * 4;
      rgba[o] = q.labels[j][i];
      rgba[o + 1] = q.labels[coeffs + j][i];
      rgba[o + 2] = q.labels[2 * coeffs + j][i];
      rgba[o + 3] = 0xff;
    }
  return { count, labels, codebook: q.centroids, centroidsRgba: rgba };
}

/** Nearest centroid to `data[p·d …]`. */
function nearest(data: Float32Array, p: number, cent: Float32Array, d: number): number {
  let best = 0, bestD = Infinity;
  const base = p * d;
  for (let c = 0, cb = 0; cb < cent.length; c++, cb += d) {
    let dist = 0;
    for (let k = 0; k < d; k++) {
      const t = data[base + k] - cent[cb + k];
      dist += t * t;
      if (dist >= bestD) break;
    }
    if (dist < bestD) {
      bestD = dist;
      best = c;
    }
  }
  return best;
}

/** Lloyd over the `members` points of `data`, with `k` initial centroids
 *  evenly spaced in the list. Returns centroids and each member's local label. */
function lloyd(data: Float32Array, d: number, members: Uint32Array, k: number, iterations: number) {
  k = Math.max(Math.min(k, members.length), 1);
  const cent = new Float32Array(k * d);
  for (let c = 0; c < k; c++) {
    const m = members[Math.floor((c * members.length) / k)];
    cent.set(data.subarray(m * d, (m + 1) * d), c * d);
  }
  const labels = new Uint32Array(members.length);
  const sums = new Float64Array(k * d), counts = new Uint32Array(k);
  for (let it = 0; it < iterations; it++) {
    let changed = 0;
    for (let i = 0; i < members.length; i++) {
      const c = nearest(data, members[i], cent, d);
      if (labels[i] !== c) changed++;
      labels[i] = c;
    }
    if (it > 0 && changed === 0) break;
    sums.fill(0);
    counts.fill(0);
    for (let i = 0; i < members.length; i++) {
      const l = labels[i], m = members[i];
      counts[l]++;
      for (let j = 0; j < d; j++) sums[l * d + j] += data[m * d + j];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) for (let j = 0; j < d; j++) cent[c * d + j] = sums[c * d + j] / counts[c];
      else {
        // Empty centroid: reseed it at some point of the group.
        const m = members[(c * 7919 + it * 104_729) % members.length];
        cent.set(data.subarray(m * d, (m + 1) * d), c * d);
      }
    }
  }
  return { cent, labels };
}

/** Hierarchical k-means: coarse clusters on a sample, every point assigned
 *  to them, then each cluster subdivided with centroids proportional to its
 *  population (sum = `k`). */
export function kmeansTwoLevel(data: Float32Array, n: number, d: number, k: number) {
  const step = Math.max(Math.floor(n / COARSE_SAMPLE), 1);
  const sample = Uint32Array.from({ length: Math.ceil(n / step) }, (_, i) => i * step);
  const { cent: coarse } = lloyd(data, d, sample, Math.min(COARSE_CLUSTERS, k), KMEANS_ITERATIONS);
  const coarseK = coarse.length / d;

  const buckets: number[][] = Array.from({ length: coarseK }, () => []);
  for (let i = 0; i < n; i++) buckets[nearest(data, i, coarse, d)].push(i);
  const groups = buckets.filter((g) => g.length).map((g) => Uint32Array.from(g));

  // Distribute the k centroids proportionally (at least 1 per group).
  const alloc = groups.map((g) => Math.max(Math.floor((g.length * k) / n), 1));
  let total = alloc.reduce((s, a) => s + a, 0);
  const bySize = groups.map((_, g) => g).sort((a, b) => groups[b].length - groups[a].length);
  let cursor = 0;
  while (total < k && bySize.length) {
    const g = bySize[cursor % bySize.length];
    if (alloc[g] < groups[g].length) {
      alloc[g]++;
      total++;
    }
    cursor++;
    if (cursor > bySize.length * 4 && total < k) break;
  }
  while (total > k) {
    const g = bySize[cursor % bySize.length];
    if (alloc[g] > 1) {
      alloc[g]--;
      total--;
    }
    cursor++;
  }

  const parts: Float32Array[] = [];
  const labels = new Uint16Array(n);
  let base = 0;
  groups.forEach((members, g) => {
    const { cent, labels: local } = lloyd(data, d, members, alloc[g], KMEANS_ITERATIONS);
    for (let i = 0; i < members.length; i++) labels[members[i]] = base + local[i];
    base += cent.length / d;
    parts.push(cent);
  });
  const centroids = new Float32Array(base * d);
  let o = 0;
  for (const p of parts) {
    centroids.set(p, o);
    o += p.length;
  }
  return { centroids, labels };
}
