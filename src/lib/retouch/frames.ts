// Which restored frames go into the fine-tune. Neighbouring frames of a clip are
// nearly identical, and too many restored frames relative to the photos pull the
// splat toward details the video model reinvented: the center, which the photos
// cover well, loses sharpness (scripts/restore-test: 1 in every 3 frames
// recovered 2 dB on the photos and kept the ghost removed).

/** Default: restored frames up to 25% of the photo count. */
export const DEFAULT_RESTORED_RATIO = 0.25;

/** Total restored frames allowed for `photos` photos. */
export const restoredLimit = (photos: number, ratio: number) => Math.max(1, Math.round(photos * ratio));

/** Indices picked from each clip (`counts` = frames per clip), at most `max`
 *  in total. Each clip's quota is proportional to its length (at least 1), and
 *  frames are evenly spaced at the middle of each segment, skipping the ends
 *  (which sit almost at a real photo's pose). */
export function pickRestoredFrames(counts: number[], max: number): number[][] {
  const total = counts.reduce((a, b) => a + b, 0);
  if (max >= total) return counts.map((n) => Array.from({ length: n }, (_, i) => i));
  const quota = counts.map((n) => (n ? Math.max(1, Math.round((max * n) / total)) : 0));
  // Trim so the sum matches `max` (taking from the largest clips).
  let excess = quota.reduce((a, b) => a + b, 0) - Math.max(max, counts.filter(Boolean).length);
  while (excess > 0) {
    const i = quota.indexOf(Math.max(...quota));
    if (quota[i] <= 1) break;
    quota[i]--;
    excess--;
  }
  return counts.map((n, c) => {
    const k = Math.min(quota[c], n);
    return Array.from({ length: k }, (_, j) => Math.min(n - 1, Math.floor(((j + 0.5) * n) / k)));
  });
}
