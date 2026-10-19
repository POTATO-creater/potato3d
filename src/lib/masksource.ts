// Mask video or image sequence (method "external"): which mask frame goes
// with each project frame, and the small numeric rules around it. Pure, so
// both the job and the preview use the same mapping.

import type { MaskSource } from "./api";

/** How far (in source frames) a mask frame may sit from the frame it masks.
 *  Video models like LTX return up to 7 frames fewer than they got (8n + 1),
 *  so an attached mask may end a little early. */
export const MAX_GAP = 8;

/** Frame rate cap of the video sent to a mask workflow: a 60 fps source goes
 *  at 30 (every other frame); video models gain nothing from more. */
export const WORKFLOW_MAX_FPS = 30;

/** Frame rate of the video made from images (they have none). */
export const IMAGES_FPS = 24;

/** Long side cap of the video sent to a mask workflow. */
export const WORKFLOW_MAX_SIDE = 1920;

/** Mask frame for a source position (seconds into the video, or the image
 *  index) and its distance from it in source frames. */
export function maskFrameFor(src: Pick<MaskSource, "count" | "positions" | "fps">, position: number): { index: number; gap: number } {
  const perFrame = src.fps ?? 1;
  if (!src.positions) {
    const k = Math.round(position * perFrame);
    const index = Math.min(Math.max(k, 0), src.count - 1);
    return { index, gap: Math.abs(k - index) };
  }
  const p = src.positions;
  const n = Math.min(p.length, src.count);
  // Positions increase: binary search for the nearest.
  let lo = 0, hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (p[mid] < position) lo = mid + 1;
    else hi = mid;
  }
  const index = lo > 0 && Math.abs(p[lo - 1] - position) <= Math.abs(p[lo] - position) ? lo - 1 : lo;
  return { index, gap: Math.abs(p[index] - position) * perFrame };
}

/** Frames to send so a video model keeps them all: the next 8n + 1 (LTX
 *  drops whatever is past it). The extra ones repeat the last frame. */
export const paddedFrames = (n: number) => (n <= 1 ? 1 : Math.ceil((n - 1) / 8) * 8 + 1);

/** Keep one frame in `step` so the workflow gets at most WORKFLOW_MAX_FPS. */
export const fpsStep = (fps: number) => Math.max(1, Math.round(fps / WORKFLOW_MAX_FPS));

/** Even size with the long side at most `maxSide` (H.264 needs even sides). */
export function workflowSize(width: number, height: number, maxSide = WORKFLOW_MAX_SIDE): [number, number] {
  const k = Math.min(1, maxSide / Math.max(width, height));
  return [Math.max(2, Math.round((width * k) / 2) * 2), Math.max(2, Math.round((height * k) / 2) * 2)];
}

/** Black and white points of a decoded mask: video compression leaves the
 *  background a few levels above 0 (faint haze in training) and the subject
 *  a few below 255. */
const BLACK = 8, WHITE = 247;
export function maskLevel(v: number): number {
  return v <= BLACK ? 0 : v >= WHITE ? 255 : Math.round(((v - BLACK) * 255) / (WHITE - BLACK));
}

/** Mask (255 = subject) from decoded RGBA: the mean of the channels, since
 *  a mask video is gray and an image may be any color depth. */
export function rgbaToMask(rgba: Uint8ClampedArray | Uint8Array): Uint8Array {
  const m = new Uint8Array(rgba.length >> 2);
  for (let i = 0; i < m.length; i++) m[i] = maskLevel((rgba[4 * i] + rgba[4 * i + 1] + rgba[4 * i + 2]) / 3);
  return m;
}
