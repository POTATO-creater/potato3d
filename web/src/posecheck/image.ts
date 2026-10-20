// Image primitives for the pose check: distance transforms, binary
// morphology, gradients and sampling on row-major w×h arrays. Binary masks
// are Uint8Array with 0/1.

const INF = 1e20;

/** 1-D squared distance transform of `f` (Felzenszwalb & Huttenlocher), into `d`. */
function dt1(f: Float64Array, n: number, d: Float64Array, v: Int32Array, z: Float64Array) {
  let k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = INF;
  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = INF;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) ** 2 + f[v[k]];
  }
}

/** Euclidean distance from each pixel to the nearest pixel where `target`
 *  is set (0 on those pixels). With no target pixel at all, every distance is
 *  huge (~1e10). */
export function distanceTo(target: Uint8Array, w: number, h: number): Float32Array {
  const n = Math.max(w, h);
  const f = new Float64Array(n), d = new Float64Array(n), z = new Float64Array(n + 1);
  const v = new Int32Array(n);
  const g = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) g[i] = target[i] ? 0 : INF;
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = g[y * w + x];
    dt1(f, h, d, v, z);
    for (let y = 0; y < h; y++) g[y * w + x] = d[y];
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) f[x] = g[row + x];
    dt1(f, w, d, v, z);
    for (let x = 0; x < w; x++) out[row + x] = Math.sqrt(d[x]);
  }
  return out;
}

/** Signed distance to the mask's boundary in pixels: > 0 outside, < 0 inside
 *  (±1 on the pixels next to it, so it crosses 0 halfway between them). */
export function signedDistance(mask: Uint8Array, w: number, h: number): Float32Array {
  const outside = distanceTo(mask, w, h);
  const inv = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) inv[i] = mask[i] ? 0 : 1;
  const inside = distanceTo(inv, w, h);
  for (let i = 0; i < w * h; i++) outside[i] -= inside[i];
  return outside;
}

/** Offsets of a disk of radius r (x² + y² ≤ r² + 0.5); r = 1 is the 4-neighbour cross. */
export function disk(r: number): [number, number][] {
  const o: [number, number][] = [];
  for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) if (x * x + y * y <= r * r + 0.5) o.push([x, y]);
  return o;
}

/** Binary erosion by `shape` (offsets); pixels outside the image count as 0. */
export function erodeBy(m: Uint8Array, w: number, h: number, shape: [number, number][]): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (!m[y * w + x]) continue;
      let keep = 1;
      for (const [dx, dy] of shape) {
        const sx = x + dx, sy = y + dy;
        if (sx < 0 || sy < 0 || sx >= w || sy >= h || !m[sy * w + sx]) {
          keep = 0;
          break;
        }
      }
      out[y * w + x] = keep;
    }
  return out;
}

/** Binary dilation by `shape` (symmetric offsets). */
export function dilateBy(m: Uint8Array, w: number, h: number, shape: [number, number][]): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (!m[y * w + x]) continue;
      for (const [dx, dy] of shape) {
        const sx = x + dx, sy = y + dy;
        if (sx >= 0 && sy >= 0 && sx < w && sy < h) out[sy * w + sx] = 1;
      }
    }
  return out;
}

/** Erosion repeated `n` times by the 4-neighbour cross. */
export function erodeCross(m: Uint8Array, w: number, h: number, n = 1): Uint8Array {
  const cross = disk(1);
  let out = m;
  for (let i = 0; i < n; i++) out = erodeBy(out, w, h, cross);
  return out === m ? m.slice() : out;
}

/** Morphological opening by a disk of radius r: removes blobs and strands
 *  thinner than the disk. */
export function openDisk(m: Uint8Array, w: number, h: number, r: number): Uint8Array {
  const s = disk(r);
  return dilateBy(erodeBy(m, w, h, s), w, h, s);
}

/** Sobel gradient divided by 8 (per pixel), borders reflected. */
export function sobel(img: Float32Array, w: number, h: number): { gx: Float32Array; gy: Float32Array } {
  const gx = new Float32Array(w * h), gy = new Float32Array(w * h);
  const at = (x: number, y: number) => img[Math.min(Math.max(y, 0), h - 1) * w + Math.min(Math.max(x, 0), w - 1)];
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const a = at(x - 1, y - 1), b = at(x, y - 1), c = at(x + 1, y - 1);
      const d = at(x - 1, y), f = at(x + 1, y);
      const g = at(x - 1, y + 1), k = at(x, y + 1), l = at(x + 1, y + 1);
      gx[y * w + x] = (c + 2 * f + l - a - 2 * d - g) / 8;
      gy[y * w + x] = (g + 2 * k + l - a - 2 * b - c) / 8;
    }
  return { gx, gy };
}

/** Bilinear sample at continuous array coordinates (x, y) = pixel indices,
 *  clamped to the border. */
export function bilinear(img: Float32Array, w: number, h: number, x: number, y: number): number {
  x = Math.min(Math.max(x, 0), w - 1);
  y = Math.min(Math.max(y, 0), h - 1);
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const fx = x - x0, fy = y - y0;
  const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
  const i00 = img[y0 * w + x0], i10 = img[y0 * w + x1], i01 = img[y1 * w + x0], i11 = img[y1 * w + x1];
  return (i00 * (1 - fx) + i10 * fx) * (1 - fy) + (i01 * (1 - fx) + i11 * fx) * fy;
}

/** 3×3 binomial blur ([1 2 1]/4 each way), borders clamped. */
export function blur(img: Float32Array, w: number, h: number): Float32Array {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const r = y * w;
      tmp[r + x] = (img[r + Math.max(x - 1, 0)] + 2 * img[r + x] + img[r + Math.min(x + 1, w - 1)]) / 4;
    }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      out[y * w + x] = (tmp[Math.max(y - 1, 0) * w + x] + 2 * tmp[y * w + x] + tmp[Math.min(y + 1, h - 1) * w + x]) / 4;
  return out;
}

/** Half size (2×2 mean; an odd last row/column is dropped). */
export function halve(img: Float32Array, w: number, h: number): { img: Float32Array; w: number; h: number } {
  const w2 = Math.max(1, w >> 1), h2 = Math.max(1, h >> 1);
  const out = new Float32Array(w2 * h2);
  for (let y = 0; y < h2; y++)
    for (let x = 0; x < w2; x++) {
      const a = 2 * y * w + 2 * x;
      const b = Math.min(2 * y + 1, h - 1) * w + 2 * x;
      const dx = 2 * x + 1 < w ? 1 : 0;
      out[y * w2 + x] = (img[a] + img[a + dx] + img[b] + img[b + dx]) / 4;
    }
  return { img: out, w: w2, h: h2 };
}

/** The mask with its holes filled: background pixels not connected (4-way)
 *  to the image border become subject. */
export function fillHoles(m: Uint8Array, w: number, h: number): Uint8Array {
  const outside = new Uint8Array(w * h);
  const stack: number[] = [];
  const push = (i: number) => {
    if (!m[i] && !outside[i]) {
      outside[i] = 1;
      stack.push(i);
    }
  };
  for (let x = 0; x < w; x++) {
    push(x);
    push((h - 1) * w + x);
  }
  for (let y = 0; y < h; y++) {
    push(y * w);
    push(y * w + w - 1);
  }
  while (stack.length) {
    const i = stack.pop()!;
    const x = i % w, y = (i - x) / w;
    if (x > 0) push(i - 1);
    if (x < w - 1) push(i + 1);
    if (y > 0) push(i - w);
    if (y < h - 1) push(i + w);
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = outside[i] ? 0 : 1;
  return out;
}
