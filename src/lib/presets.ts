import type { Settings } from "./api";

export type PresetId = "preview" | "balanced" | "quality";

/** Nome e descrição de cada preset ficam no catálogo (`t.presets`). */
export interface Preset {
  id: PresetId;
  settings: Settings;
}

// Calibrado para Apple Silicon com 16 GB de memória unificada (COLMAP na CPU,
// Brush via Metal). Em GPUs NVIDIA o SSIM deve pesar bem menos, mas isso
// ainda não foi medido.
//
// Treino medido num vídeo real de IA (107 frames 960×1280, M4): sem SSIM o
// passo é ~3× mais barato e, no mesmo tempo, rende PSNR maior.
//   sem SSIM  5k → 29,7 dB (~3 min) · 15k → 32,6 dB (8,6 min)
//   com SSIM  7k → 31,8 dB (9,3 min) · 20k → 33,2 dB (~28 min)
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
