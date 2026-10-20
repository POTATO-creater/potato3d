// Pose consistency score: each frame against the silhouette of a model that
// never saw it (the other fold). Where the model sticks out of the frame's
// mask, that frame's pose disagrees with the others' (or the video changed
// shape): a misplaced contour is what turns into a doubled edge in training.

import { openDisk } from "./image.ts";

export interface FrameScore {
  /** Pixels where the model sticks out of the frame's mask, after opening. */
  out: number;
  /** Pixels where the frame's mask sticks out of the model, after opening. */
  in: number;
  /** Pixels in the frame's mask. */
  subject: number;
}

/** XOR of the render's silhouette (alpha > 0.5) and the frame's mask, split
 *  into "out" and "in", each opened with a disk of radius `radius` (drops
 *  thin strands and anti-aliasing). */
export function xorScore(alpha: Float32Array, mask: Uint8Array, w: number, h: number, radius: number): FrameScore {
  const out = new Uint8Array(w * h), inn = new Uint8Array(w * h);
  let subject = 0;
  for (let i = 0; i < w * h; i++) {
    const r = alpha[i] > 0.5, g = mask[i] !== 0;
    if (g) subject++;
    if (r && !g) out[i] = 1;
    else if (g && !r) inn[i] = 1;
  }
  const count = (m: Uint8Array) => {
    let n = 0;
    for (let i = 0; i < m.length; i++) n += m[i];
    return n;
  };
  return { out: count(openDisk(out, w, h, radius)), in: count(openDisk(inn, w, h, radius)), subject };
}

/** "Out" area over the subject's (0 for an empty mask). With a short screen,
 *  "in" is mostly thin detail the model has not learned yet, so
 *  only "out" goes into the score. */
export const frameValue = (s: FrameScore) => (s.subject > 0 ? s.out / s.subject : 0);

/** Mean of the frames' values: the orbit's score. */
export function orbitScore(values: readonly number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

/** Mean of each run of `size` consecutive frames (the last run may be shorter). */
export function segments(values: readonly number[], size: number): { start: number; end: number; value: number }[] {
  const out = [];
  for (let i = 0; i < values.length; i += size) {
    const run = values.slice(i, i + size);
    out.push({ start: i, end: i + run.length - 1, value: orbitScore(run) });
  }
  return out;
}

/** Opening radius at an analysis size (longer side): ~2 px of a 1280 px
 *  frame, at least 1. */
export const openingRadius = (maxSide: number) => Math.max(1, Math.round(maxSide / 640));
