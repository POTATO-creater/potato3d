import { Channel, convertFileSrc, invoke } from "@tauri-apps/api/core";

export type Stage = "frames" | "masks" | "poses" | "train";
export const STAGES: Stage[] = ["frames", "masks", "poses", "train"];

export interface Tools {
  ffmpeg: string | null;
  ffprobe: string | null;
  colmap: string | null;
  /** COLMAP com CUDA: extração, matching e bundle adjustment na GPU. */
  colmapCuda: boolean;
  brush: string | null;
  /** macOS: recortador Vision (sidecar). */
  masker: string | null;
  /** Windows: modelo BiRefNet baixado pelo instalador. */
  maskModel: string | null;
  os: "macos" | "windows" | "linux";
}

export interface VideoInfo {
  width: number;
  height: number;
  fps: number;
  duration: number;
  codec: string;
}

export interface FrameOptions {
  targetFrames: number;
  maxSide: number;
  oversample: number;
  trimStart: number;
  trimEnd: number | null;
}

export interface MaskOptions {
  enabled: boolean;
  /** Também usa as máscaras no COLMAP (ignora features do fundo). */
  forPoses: boolean;
}

export interface MaskStats {
  count: number;
  noSubject: number;
  meanCoverage: number;
  backend: string;
}

export interface PoseOptions {
  features: "sift" | "siftPlus" | "aliked";
  cameraModel: "SIMPLE_PINHOLE" | "PINHOLE";
  matchWindow: number;
  closedLoop: boolean;
}

export interface TrainOptions {
  totalSteps: number;
  shDegree: number;
  maxSplats: number;
  maxResolution: number;
  exportEvery: number;
  evalSplitEvery: number;
  /** 0 desliga a perda SSIM (~3× mais rápido no Mac). */
  ssimWeight: number;
  /** Regularização de escala; omitido = padrão do Brush. */
  scaleLossWeight?: number;
}

export interface Settings {
  frames: Omit<FrameOptions, "trimStart" | "trimEnd">;
  poses: PoseOptions;
  train: TrainOptions;
}

export interface JobConfig extends Omit<Settings, "frames"> {
  videoPath: string;
  projectDir: string;
  fromStage: Stage;
  frames: FrameOptions;
  masks: MaskOptions;
}

export interface FrameInfo {
  path: string;
  time: number;
  sharpness: number;
}

export interface ModelStats {
  registered: number;
  total: number;
  points: number;
  reprojectionError: number | null;
  mapper: string;
}

export interface ProjectState {
  videoPath: string | null;
  frames: FrameInfo[];
  masks: MaskStats | null;
  masksDir: string | null;
  poses: ModelStats | null;
  finalPly: string | null;
  lastPsnr: number | null;
}

export interface SceneFrame {
  center: [number, number, number];
  up: [number, number, number];
  radius: number;
  cameras: [number, number, number][];
  /** Índice do frame de cada câmera; saltos indicam frames não registrados. */
  frameIndices: number[];
}

export type TrainEvent =
  | { kind: "step"; iter: number; splats: number }
  | { kind: "eval"; iter: number; psnr: number; ssim: number }
  | { kind: "snapshot"; iter: number; path: string };

export type JobEvent =
  | { type: "stageStarted"; stage: Stage }
  | { type: "progress"; stage: Stage; fraction: number; message: string }
  | { type: "log"; stage: Stage; line: string }
  | { type: "framesReady"; frames: FrameInfo[] }
  | { type: "masksReady"; stats: MaskStats; dir: string }
  | { type: "stageSkipped"; stage: Stage; reason: string }
  | { type: "posesReady"; stats: ModelStats }
  | { type: "train"; event: TrainEvent }
  | { type: "stageDone"; stage: Stage; seconds: number }
  | { type: "finished"; ply: string; seconds: number }
  | { type: "failed"; stage: Stage | null; error: string }
  | { type: "cancelled" };

export interface InstallPackage {
  id: string;
  label: string;
  file: string;
  size: number;
}

export interface InstallPlan {
  packages: InstallPackage[];
  totalBytes: number;
  cuda: boolean;
  note: string | null;
}

export type InstallEvent =
  | { type: "started"; id: string; label: string; total: number }
  | { type: "progress"; id: string; done: number; total: number }
  | { type: "extracting"; id: string }
  | { type: "done"; id: string }
  | { type: "finished" }
  | { type: "failed"; error: string }
  | { type: "cancelled" };

export const api = {
  detectTools: () => invoke<Tools>("detect_tools"),
  probeVideo: (path: string) => invoke<VideoInfo>("probe_video", { path }),
  defaultProjectDir: (videoPath: string) => invoke<string>("default_project_dir", { videoPath }),
  loadProject: (projectDir: string) => invoke<ProjectState>("load_project", { projectDir }),
  sceneFrame: (projectDir: string) => invoke<SceneFrame>("scene_frame", { projectDir }),
  cancelJob: () => invoke<void>("cancel_job"),
  installPlan: () => invoke<InstallPlan>("install_plan"),
  cancelInstall: () => invoke<void>("cancel_install"),
  installTools: (onEvent: (e: InstallEvent) => void) => {
    const channel = new Channel<InstallEvent>();
    channel.onmessage = onEvent;
    return invoke<void>("install_tools", { onEvent: channel });
  },
  startJob: (config: JobConfig, onEvent: (e: JobEvent) => void) => {
    const channel = new Channel<JobEvent>();
    channel.onmessage = onEvent;
    return invoke<void>("start_job", { config, onEvent: channel });
  },
};

/** URL servível pelo protocolo `asset:` do Tauri (com cache-buster opcional). */
export function fileUrl(path: string, bust?: string | number) {
  const url = convertFileSrc(path);
  return bust === undefined ? url : `${url}?v=${bust}`;
}
