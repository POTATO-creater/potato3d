// Edge decontamination of the training images.
//
// A soft mask edge keeps the photo's color, a mix of subject and background
// (I = a·F + (1 − a)·B). Brush premultiplies by the alpha, so the background
// leaks into the training target and the splat learns a light fringe around
// hair and outlines. Recovers F = (I − (1 − a)·B) / a with B estimated from the
// background nearby; where a is too small, F is the solid subject color nearby.
// The mixing ratio a is the training alpha (eroded mask), not the raw matte:
// mattes leak past the real outline, see decontam.rs.

/** Grid cell, in pixels. */
const BLOCK = 4;
/** Box blur radius on the grid (cells) and number of passes: ~±24 px support. */
const BLUR_RADIUS = 3;
const BLUR_PASSES = 2;
/** Mask values (0–255) counted as clear background, and the solid subject: only
 *  a full mask, near-solid edge pixels are still mixed with the background. */
const BACKGROUND_MAX = 5;
const SOLID = 255;
/** Below this alpha the unmixing is unstable: use the solid color nearby. */
const UNMIX_MIN = 0.25;

/** Normalized average color of the pixels selected by `pick`, as a blurred grid
 *  (rgb per cell; NaN where no selected pixel is nearby). */
function field(rgba: Uint8ClampedArray, mask: Uint8Array, w: number, h: number, pick: (m: number) => boolean) {
  const gw = Math.ceil(w / BLOCK), gh = Math.ceil(h / BLOCK);
  let sum = new Float32Array(gw * gh * 4); // rgb·weight, weight
  for (let y = 0; y < h; y++) {
    const row = Math.floor(y / BLOCK) * gw;
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      if (!pick(mask[p])) continue;
      const s = (row + Math.floor(x / BLOCK)) * 4;
      sum[s] += rgba[4 * p];
      sum[s + 1] += rgba[4 * p + 1];
      sum[s + 2] += rgba[4 * p + 2];
      sum[s + 3] += 1;
    }
  }
  for (let i = 0; i < BLUR_PASSES; i++) {
    sum = boxBlur(sum, gw, gh, 1, 0);
    sum = boxBlur(sum, gw, gh, 0, 1);
  }
  const color = new Float32Array(gw * gh * 3);
  for (let c = 0; c < gw * gh; c++) {
    const wgt = sum[4 * c + 3];
    for (let k = 0; k < 3; k++) color[3 * c + k] = wgt > 1e-3 ? sum[4 * c + k] / wgt : NaN;
  }
  return {
    at(x: number, y: number) {
      const c = Math.min(Math.floor(y / BLOCK), gh - 1) * gw + Math.min(Math.floor(x / BLOCK), gw - 1);
      return 3 * c;
    },
    color,
  };
}

/** Box blur along (dx, dy) with radius BLUR_RADIUS, edges clamped. */
function boxBlur(src: Float32Array, w: number, h: number, dx: number, dy: number) {
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = 4 * (y * w + x);
      for (let k = -BLUR_RADIUS; k <= BLUR_RADIUS; k++) {
        const sx = Math.min(Math.max(x + k * dx, 0), w - 1);
        const sy = Math.min(Math.max(y + k * dy, 0), h - 1);
        const s = 4 * (sy * w + sx);
        out[o] += src[s];
        out[o + 1] += src[s + 1];
        out[o + 2] += src[s + 2];
        out[o + 3] += src[s + 3];
      }
    }
  }
  return out;
}

/** Replaces the color of the pixels whose training alpha is partial
 *  (0 < alpha < 255) with the subject color unmixed from the background.
 *  `mask` is the raw matte (finds clear background and solid subject); `alpha`
 *  the training alpha (the eroded mask), also the mixing ratio. Pixels with
 *  alpha 0 or 255 are untouched. */
export function decontaminate(rgba: Uint8ClampedArray, mask: Uint8Array, alpha: Uint8Array, w: number, h: number) {
  const bg = field(rgba, mask, w, h, (m) => m <= BACKGROUND_MAX);
  const solid = field(rgba, mask, w, h, (m) => m === SOLID);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      const a = alpha[p];
      if (a === 0 || a === 255) continue;
      const m = a / 255;
      const bi = bg.at(x, y), si = solid.at(x, y);
      const hasBg = !Number.isNaN(bg.color[bi]), hasSolid = !Number.isNaN(solid.color[si]);
      if (!hasBg && !(hasSolid && m < UNMIX_MIN)) continue;
      const mm = Math.max(m, UNMIX_MIN);
      // Below UNMIX_MIN the solve amplifies noise: take the solid neighbor.
      const useSolid = hasSolid && m < UNMIX_MIN;
      for (let c = 0; c < 3; c++) {
        rgba[4 * p + c] = Math.round(
          useSolid ? solid.color[si + c] : (rgba[4 * p + c] - (1 - mm) * bg.color[bi + c]) / mm,
        ); // Uint8ClampedArray clamps
      }
    }
  }
}
