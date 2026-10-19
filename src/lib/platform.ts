import type { MaskOptions, PoseEstimator, Stage } from "./api";
import type { Messages } from "../i18n";

/** Short label of each stage's tool. */
export function stageToolLabels(t: Messages, masks?: MaskOptions, images?: boolean, estimator?: PoseEstimator): Record<Stage, string> {
  const p = t.platform;
  return {
    frames: images ? p.yourImages : p.browserDecoder,
    masks:
      masks?.method === "chroma"
        ? p.solidColor
        : masks?.method === "sam2"
          ? "SAM 2.1 · WebGPU"
          : masks?.method === "external"
            ? p.maskSource
            : "RMBG-1.4 · WebGPU",
    poses: estimator === "colmap" ? p.webPosesColmap : p.webPoses,
    train: "Brush · WebGPU",
  };
}

/** What to do about a missing tool. */
export function installCommand(t: Messages, tool: "brush" | "model"): string {
  return tool === "brush" ? t.platform.webgpuChrome : t.platform.webgpuBrowser;
}

export const basename = (p: string) => p.split(/[\\/]/).pop() ?? p;

/** Label of the button that saves a file: projects live in browser storage,
 *  so the button downloads. */
export const revealLabel = (t: Messages, what?: string) => t.platform.download(what);

/** What each stage does, for the start screen. */
export function stageBlurbs(t: Messages): Record<Stage, string> {
  return t.platform.blurbs;
}
