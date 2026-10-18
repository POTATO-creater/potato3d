import type { Settings } from "./api";

export type PresetId = "preview" | "balanced" | "quality";

/** Each preset's name and description live in the catalog (`t.presets`). */
export interface Preset {
  id: PresetId;
  settings: Settings;
}

// Calibrated on Apple Silicon with 16 GB of unified memory (Brush on WebGPU;
// the first numbers below come from the former native build).
//
// SSIM was left out of Preview and Balanced when, with Brush 0.3 (2026-10-04,
// a real AI-generated video, 107 frames 960×1280, M4), it made a step ~3×
// dearer and lost to more steps without it in the same time:
//   no SSIM    5k → 29.7 dB (~3 min) · 15k → 32.6 dB (8.6 min)
//   with SSIM  7k → 31.8 dB (9.3 min) · 20k → 33.2 dB (~28 min)
// With the pinned Brush (2026-10-18, synthetic orbit of 480 views at
// 1280×960) it costs far less (+12% in the browser) and still gains detail:
//   3k steps   no SSIM 78 s → 40.0 dB · with SSIM 83 s → 41.8 dB
//   10k steps  no SSIM 279 s → 43.1 dB · with SSIM 308 s → 43.9 dB
//
// opacBinaryWeight is left out: on the same video, with
// masks, 0.01 took semi-transparent Gaussians from 71% to 14% and removed
// the haze under the chin, but hardened the edges; the baseline looked better.
//
// "Reduce haze" (off in every preset) turns on depth distortion at
// REDUCE_HAZE_WEIGHT, every 2 steps from 25% of training, masked videos only:
// less haze around the subject in AI videos for ~25% more training time. On a clean synthetic orbit it cost 0.3–0.76 dB,
// so it stays opt-in. Weight 100
// hardened the edges; without masks any weight collapses training. See
// docs/web.md, "Reduce haze".
export const REDUCE_HAZE_WEIGHT = 10;

export const PRESETS: Preset[] = [
  {
    id: "preview",
    settings: {
      frames: { targetFrames: 90, maxSide: 1280, oversample: 3 },
      poses: { matchWindow: 8, closedLoop: true },
      train: {
        totalSteps: 5000,
        shDegree: 1,
        maxSplats: 1_000_000,
        maxResolution: 1280,
        exportEvery: 1000,
        evalSplitEvery: 16,
        ssimWeight: 0,
      },
    },
  },
  {
    id: "balanced",
    settings: {
      frames: { targetFrames: 160, maxSide: 1600, oversample: 3 },
      poses: { matchWindow: 20, closedLoop: true },
      train: {
        totalSteps: 15000,
        shDegree: 1,
        maxSplats: 2_000_000,
        maxResolution: 1600,
        exportEvery: 2500,
        evalSplitEvery: 16,
        ssimWeight: 0,
      },
    },
  },
  {
    id: "quality",
    settings: {
      frames: { targetFrames: 220, maxSide: 1920, oversample: 4 },
      poses: { matchWindow: 20, closedLoop: true },
      train: {
        totalSteps: 30000,
        shDegree: 2,
        maxSplats: 3_000_000,
        maxResolution: 1920,
        exportEvery: 5000,
        evalSplitEvery: 16,
        ssimWeight: 0.2,
      },
    },
  },
];

export const presetById = (id: PresetId) => PRESETS.find((p) => p.id === id)!;
