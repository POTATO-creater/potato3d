// Erosion of the training alpha: port of `erode` in src-tauri/src/masks.rs.

/** Radius used in the training dataset (see ALPHA_ERODE in masks.rs). */
export const ALPHA_ERODE = 1;

/** Minimum over a (2r+1)² window, separable, with the image border replicated. */
export function erode(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
  if (r <= 0) return mask.slice();
  const pass = (src: Uint8Array, dx: number, dy: number) => {
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let m = 255;
        for (let k = -r; k <= r; k++) {
          const sx = Math.min(Math.max(x + k * dx, 0), w - 1);
          const sy = Math.min(Math.max(y + k * dy, 0), h - 1);
          m = Math.min(m, src[sy * w + sx]);
        }
        out[y * w + x] = m;
      }
    }
    return out;
  };
  return pass(pass(mask, 1, 0), 0, 1);
}
