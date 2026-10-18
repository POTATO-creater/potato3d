// Brush training config (`TrainStreamConfig` fields, flattened JSON,
// kebab-case) for the app's TrainOptions.

import type { TrainOptions } from "../../../src/lib/api";

/** Background for masked training: a uniform color drawn every step (0.5 ± 0.5).
 *  On Brush's black ± 0.1, dark semi-transparent layers outside the silhouette
 *  cost almost nothing and become a ghost silhouette around the subject; with a
 *  varying background, any spurious transparency shows up in the loss. */
export const MASKED_BACKGROUND = { color: [0.5, 0.5, 0.5], noise: 0.5 };

/** When depth distortion ("Reduce haze", web/brush/geom.patch) runs during
 *  masked training: from 25% of the run, every 2 steps, at full resolution,
 *  rendered only inside the mask's box. Every step removed slightly more haze
 *  for about twice the extra training time; half-resolution renders saved less
 *  time than skipping steps. */
export const DISTORTION_SCHEDULE = { start: 0.25, every: 2, downscale: 1 };

/** The fields Gausify Orbit controls; the rest keep Brush's defaults. `masked`:
 *  the images carry the mask alpha. */
export function trainConfig(o: TrainOptions, masked: boolean): Record<string, unknown> {
  const set: Record<string, unknown> = {
    "total-train-iters": o.totalSteps,
    "sh-degree": Math.min(o.shDegree, 3),
    "max-splats": o.maxSplats,
    "max-resolution": o.maxResolution,
    "ssim-weight": Math.max(o.ssimWeight, 0),
    "eval-split-every": o.evalSplitEvery || null,
  };
  // Only with evaluation on: Brush computes iter % eval-every
  // even without eval views, and 0 there crashes training (panic in WASM).
  if (o.evalSplitEvery) set["eval-every"] = o.exportEvery;
  if (o.contribPruneMinViews) set["contrib-prune-min-views"] = o.contribPruneMinViews;
  if (o.contribPruneRatio != null) set["contrib-prune-ratio"] = o.contribPruneRatio;
  if (o.contribPruneStart != null) set["contrib-prune-start"] = o.contribPruneStart;
  if (o.opacSolidWeight) set["opac-solid-weight"] = o.opacSolidWeight;
  if (o.opacBinaryWeight) set["opac-binary-weight"] = o.opacBinaryWeight;
  if (o.volumeRegWeight) set["volume-reg-weight"] = o.volumeRegWeight;
  if (o.lrMean != null) set["lr-mean"] = o.lrMean;
  if (o.growthStopIter != null) set["growth-stop-iter"] = o.growthStopIter;
  if (masked) {
    set["background-color"] = MASKED_BACKGROUND.color;
    set["background-noise-strength"] = MASKED_BACKGROUND.noise;
    // Without masks it stays at Brush's 0: there, fog sheets next to the
    // cameras dominate the term and training collapses.
    const w = o.depthDistortionWeight ?? 0;
    if (w > 0) {
      set["depth-distortion-weight"] = w;
      set["geom-reg-start"] = DISTORTION_SCHEDULE.start;
      set["geom-reg-every"] = DISTORTION_SCHEDULE.every;
      set["geom-reg-downscale"] = DISTORTION_SCHEDULE.downscale;
      set["geom-reg-crop"] = true;
    }
  }
  return set;
}
