// Chroma key and despill (mod chroma and
// Despill), same formulas and constants. Masks: 255 = subject.

export type RGB = [number, number, number];

export interface ChromaOptions {
  autoColor: boolean;
  color: RGB;
  /** 0–100 */
  tolerance: number;
  /** 0–100 */
  softness: number;
  contiguous: boolean;
}

export const DEFAULT_CHROMA: ChromaOptions = {
  autoColor: true,
  color: [0, 255, 0],
  tolerance: 12,
  softness: 6,
  contiguous: true,
};

/** Per-channel (upper) median of the border pixels, with step (w+h)/400. */
export function borderKey(rgba: Uint8ClampedArray, w: number, h: number): RGB {
  const step = Math.max(Math.floor((w + h) / 400), 1);
  const ch: number[][] = [[], [], []];
  const push = (x: number, y: number) => {
    const o = (y * w + x) * 4;
    for (let c = 0; c < 3; c++) ch[c].push(rgba[o + c]);
  };
  for (let x = 0; x < w; x += step) {
    push(x, 0);
    push(x, h - 1);
  }
  for (let y = 0; y < h; y += step) {
    push(0, y);
    push(w - 1, y);
  }
  return ch.map((v) => v.sort((a, b) => a - b)[v.length >> 1]) as RGB;
}

/**
 * Distance to the background color, accepting the color up to 1.6× brighter
 * (the screen center is usually better lit than the edge) and never darker (a
 * shaded, greenish object doesn't become background).
 */
export function keyDistance(r: number, g: number, b: number, k: RGB) {
  const kk = Math.max(k[0] * k[0] + k[1] * k[1] + k[2] * k[2], 1);
  const s = Math.min(1.6, Math.max(1, (r * k[0] + g * k[1] + b * k[2]) / kk));
  return Math.hypot(r - s * k[0], g - s * k[1], b - s * k[2]);
}

export function chromaMask(rgba: Uint8ClampedArray, w: number, h: number, key: RGB, o: ChromaOptions): Uint8Array {
  const n = w * h;
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = keyDistance(rgba[4 * i], rgba[4 * i + 1], rgba[4 * i + 2], key);
  const tol = Math.min(100, Math.max(0, o.tolerance)) * 4.42;
  const soft = Math.max(Math.min(100, Math.max(0, o.softness)) * 4.42, 1);
  const lim = tol + soft;

  // only what connects to the border counts as background (holes in the subject stay)
  let reached: Uint8Array | null = null;
  if (o.contiguous) {
    reached = new Uint8Array(n);
    const stack = new Int32Array(n);
    let sp = 0;
    const seed = (i: number) => {
      if (!reached![i] && d[i] <= lim) {
        reached![i] = 1;
        stack[sp++] = i;
      }
    };
    for (let x = 0; x < w; x++) {
      seed(x);
      seed((h - 1) * w + x);
    }
    for (let y = 0; y < h; y++) {
      seed(y * w);
      seed(y * w + w - 1);
    }
    while (sp > 0) {
      const i = stack[--sp];
      const x = i % w;
      if (x > 0) seed(i - 1);
      if (x < w - 1) seed(i + 1);
      if (i >= w) seed(i - w);
      if (i < n - w) seed(i + w);
    }
  }
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if ((reached && !reached[i]) || d[i] >= lim) mask[i] = 255;
    else if (d[i] <= tol) mask[i] = 0;
    else mask[i] = Math.round((255 * (d[i] - tol)) / soft);
  }
  return mask;
}

/** Background color from the masks: median of pixels with mask < 26, every 3rd one. */
export function backgroundColor(samples: { rgba: Uint8ClampedArray; mask: Uint8Array }[]): RGB | null {
  const hist = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
  let n = 0;
  for (const { rgba, mask } of samples)
    for (let p = 0; p < mask.length; p += 3) {
      if (mask[p] >= 26) continue;
      for (let c = 0; c < 3; c++) hist[c][rgba[4 * p + c]]++;
      n++;
    }
  if (n < 1000) return null;
  return hist.map((h) => {
    let cum = 0;
    for (let v = 0; v < 256; v++) {
      cum += h[v];
      if (cum * 2 >= n) return v;
    }
    return 255;
  }) as RGB;
}

export interface Despill {
  channel: number;
  others: [number, number];
  strength: number;
}

/** Only well-saturated green or blue: dominant channel − max of the others > 50. */
export function despillFor(key: RGB, strength: number): Despill | null {
  let channel = 0;
  for (let c = 1; c < 3; c++) if (key[c] >= key[channel]) channel = c; // tie: the last one wins
  const others = [0, 1, 2].filter((c) => c !== channel) as [number, number];
  const other = Math.max(key[others[0]], key[others[1]]);
  if (channel === 0 || key[channel] - other <= 50 || strength <= 0) return null;
  return { channel, others, strength: Math.min(1, Math.max(0, strength / 100)) };
}

const LUMA = [0.2126, 0.7152, 0.0722];

/** Over the whole subject (mask > 0): clamps the screen channel to the max of
 *  the other two and adds the removed luminance back as gray. */
export function applyDespill(rgba: Uint8ClampedArray, mask: Uint8Array, d: Despill) {
  const c = d.channel, [a, b] = d.others;
  for (let p = 0; p < mask.length; p++) {
    if (mask[p] === 0) continue;
    const o = 4 * p;
    const limit = Math.max(rgba[o + a], rgba[o + b]);
    const spill = (rgba[o + c] - limit) * d.strength;
    if (spill <= 0) continue;
    const gray = spill * LUMA[c];
    for (let i = 0; i < 3; i++) rgba[o + i] = Math.round(Math.min(255, Math.max(0, rgba[o + i] - (i === c ? spill : 0) + gray)));
  }
}
