import type { Settings } from "./api";

export type PresetId = "preview" | "balanced" | "quality";

/** Each preset's name and description live in the catalog (`t.presets`). */
export interface Preset {
  id: PresetId;
  settings: Settings;
}

// Calibrated for Apple Silicon with 16 GB of unified memory (COLMAP on CPU,
// Brush via Metal). On NVIDIA GPUs SSIM should cost much less, but that has
// not been measured yet.
//
// Training measured on a real AI-generated video (107 frames 960×1280, M4):
// without SSIM a step is ~3× cheaper and, in the same time, gives higher PSNR.
//   no SSIM    5k → 29.7 dB (~3 min) · 15k → 32.6 dB (8.6 min)
//   with SSIM  7k → 31.8 dB (9.3 min) · 20k → 33.2 dB (~28 min)
//
// opacBinaryWeight (web Brush only) is left out: on the same video, with
// masks, 0.01 took semi-transparent Gaussians from 71% to 14% and removed
// the haze under the chin, but hardened the edges; the baseline looked better.
export const PRESETS: Preset[] = [
  {
    id: "preview",
    settings: {
      frames: { targetFrames: 90, maxSide: 1280, oversample: 3 },
      poses: { features: "sift", cameraModel: "SIMPLE_PINHOLE", matchWindow: 8, closedLoop: true },
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
      poses: { features: "siftPlus", cameraModel: "SIMPLE_PINHOLE", matchWindow: 20, closedLoop: true },
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
      poses: { features: "siftPlus", cameraModel: "SIMPLE_PINHOLE", matchWindow: 20, closedLoop: true },
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
