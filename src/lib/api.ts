export type Stage = "frames" | "masks" | "poses" | "train";
export const STAGES: Stage[] = ["frames", "masks", "poses", "train"];

/** Translatable text from the backend (or the reducer): a key in the `msg`
 *  catalog of src/i18n plus parameters. A parameter can be another `Msg`. */
export interface Msg {
  key: string;
  args?: Record<string, MsgArg>;
}
export type MsgArg = string | number | Msg;

export interface Tools {
  ffmpeg: string | null;
  ffprobe: string | null;
  colmap: string | null;
  /** COLMAP with CUDA: extraction, matching and bundle adjustment on the GPU. */
  colmapCuda: boolean;
  brush: string | null;
  /** macOS: Vision masker (sidecar). */
  masker: string | null;
  /** Windows: masking models downloaded by the installer. */
  birefnetModel: string | null;
  modnetModel: string | null;
  /** Windows and macOS (Apple Silicon): RMBG-1.4, downloaded on demand. */
  rmbgModel: string | null;
  /** "web": everything runs in the browser (web/); the tools above become labels. */
  os: "macos" | "windows" | "linux" | "web";
}

export interface VideoInfo {
  width: number;
  height: number;
  fps: number;
  duration: number;
  codec: string;
}

/** Already-extracted frames (loose images or a folder) instead of a video. */
export interface ImageSet {
  /** In orbit order: natural sort of the names. */
  paths: string[];
  /** Folder of the images: names the project. */
  dir: string;
  /** Of the first image, already rotated per EXIF. */
  width: number;
  height: number;
}

/** Extensions accepted as frames. */
export const IMAGE_EXT = ["jpg", "jpeg", "png", "webp"];

export interface FrameOptions {
  targetFrames: number;
  maxSide: number;
  oversample: number;
  trimStart: number;
  trimEnd: number | null;
}

/** auto: Vision (macOS) / MODNet (Windows) · people: MODNet · general: BiRefNet · rmbg: RMBG-1.4 ·
 *  sam2: SAM 2.1 with propagation across frames (web only) · chroma: solid color */
export type MaskMethod = "auto" | "people" | "general" | "rmbg" | "sam2" | "chroma";

/** Where the subject is, for SAM 2.1, in normalized (0–1) frame coordinates. */
export interface SubjectPrompt {
  /** Video time where it was marked (s); null = first frame. */
  time: number | null;
  /** With images instead of a video: the name of the marked image. */
  image?: string | null;
  /** Box [x0, y0, x1, y1]. */
  box: [number, number, number, number] | null;
  /** label 1 = subject, 0 = background. */
  points: { x: number; y: number; label: 0 | 1 }[];
}

export interface ChromaOptions {
  autoColor: boolean;
  color: [number, number, number];
  /** 0–100 */
  tolerance: number;
  /** 0–100 */
  softness: number;
  contiguous: boolean;
}

export const DEFAULT_CHROMA: ChromaOptions = {
  autoColor: true,
  color: [0, 255, 0],
  tolerance: 12,
  softness: 6,
  contiguous: true,
};

export interface MaskOptions {
  enabled: boolean;
  /** Also use the masks in COLMAP (ignores background features). */
  forPoses: boolean;
  method: MaskMethod;
  chroma: ChromaOptions;
  /** Removes green/blue background spill from the subject (any method). */
  despill: boolean;
  /** 0–100 */
  despillStrength: number;
  /** SAM 2.1: subject marked in the video; null = whatever is at the center of the first frame. */
  prompt?: SubjectPrompt | null;
}

export interface MaskStats {
  count: number;
  noSubject: number;
  meanCoverage: number;
  backend: string;
  keyColor?: [number, number, number] | null;
}

export interface PoseOptions {
  features: "sift" | "siftPlus" | "aliked";
  cameraModel: "SIMPLE_PINHOLE" | "PINHOLE";
  matchWindow: number;
  closedLoop: boolean;
  /** Cameras forced onto one plane. Missing in older settings. */
  flatOrbit?: boolean;
  /** Frames without a registered pose get one inferred from their neighbors and
   *  join training. Missing in older settings. */
  fillMissing?: boolean;
}

export interface TrainOptions {
  totalSteps: number;
  shDegree: number;
  maxSplats: number;
  maxResolution: number;
  exportEvery: number;
  evalSplitEvery: number;
  /** 0 disables the SSIM loss (~3× faster on Mac). */
  ssimWeight: number;
  /** Scale regularization; omitted = Brush default. */
  scaleLossWeight?: number;
  /** Prunes Gaussians whose contribution to the loss, summed over the views
   *  in the refine window, is negative (they hurt more than they help:
   *  ghosts and doubled edges). Minimum views to judge; omitted or 0 = off. */
  contribPruneMinViews?: number;
  /** How clear the harm must be: balance < −ratio × sum of |effects|. */
  contribPruneRatio?: number;
  /** Fraction of training before contribution pruning starts. */
  contribPruneStart?: number;
  /** DeG regularization: weight of mean(1 − α) (pushes toward opaque). */
  opacSolidWeight?: number;
  /** Weight of mean(α(1 − α)) (pushes opacity toward 0 or 1). */
  opacBinaryWeight?: number;
  /** Weight of mean Gaussian size relative to the scene (keeps them compact). */
  volumeRegWeight?: number;
  /** Initial learning rate of positions; omitted = Brush default (2e-5).
   *  Retouch fine-tuning uses much less so the splat doesn't fall apart. */
  lrMean?: number;
  /** Step at which splat growth stops; omitted = Brush default. */
  growthStopIter?: number;
}

export interface Settings {
  frames: Omit<FrameOptions, "trimStart" | "trimEnd">;
  poses: PoseOptions;
  train: TrainOptions;
}

export interface JobConfig extends Omit<Settings, "frames"> {
  /** The video, or the images folder when `images` is set. */
  videoPath: string;
  /** Already-extracted frames, in orbit order, instead of the video. */
  images: string[] | null;
  projectDir: string;
  fromStage: Stage;
  /** Last stage to run (inclusive); missing = through training. */
  toStage?: Stage;
  frames: FrameOptions;
  masks: MaskOptions;
}

export interface FrameInfo {
  path: string;
  time: number;
  sharpness: number;
  /** Source image name when frames came from images (then `time` does not
   *  apply). */
  source?: string;
  /** Rejected by hand: the file leaves the dataset (moved to `rejected/`) and
   *  is skipped by masking, poses and training. */
  rejected?: boolean;
}

export interface ModelStats {
  registered: number;
  total: number;
  points: number;
  reprojectionError: number | null;
  mapper: string;
  /** Poses inferred from neighbors (frames SfM did not register). */
  inferred?: number;
  /** Of those, the ones refined against the 3D points. */
  refined?: number;
}

export interface ProjectState {
  videoPath: string | null;
  frames: FrameInfo[];
  masks: MaskStats | null;
  masksDir: string | null;
  poses: ModelStats | null;
  finalPly: string | null;
  lastPsnr: number | null;
  lastExport?: ExportStats | null;
  /** Applied retouch: `base` is the previous splat (to undo, and to reapply
   *  without stacking fine-tunes); `clips`, the clips used. */
  retouch?: { base: string; clips: string[]; appliedAt: number } | null;
}

export interface ExportOptions {
  format: "sog" | "ply";
  /** Fraction removed by merging similar Gaussians (0–0.9). */
  decimate: number;
  /** Minimum opacity (0–1; 0 disables pruning). */
  minOpacity: number;
  /** Output SH degree; null keeps the training one. */
  shDegree: number | null;
}

export interface ExportStats {
  path: string;
  inputCount: number;
  /** After opacity pruning. */
  prunedCount: number;
  outputCount: number;
  inputBytes: number;
  outputBytes: number;
  shDegree: number;
  seconds: number;
}

/** Project package (.gorbit): a ZIP of what has been processed so far, to
 *  open on another machine (or in the other app) as a cached project. */
export const PACKAGE_EXT = "gorbit";

export interface PackRequest {
  projectDir: string;
  /** From `Host.pickPackageDest`. */
  dest: string;
  /** Finished stages to include besides frames (not those of a running job). */
  include: { masks: boolean; poses: boolean; splat: boolean };
  /** Source video or images, if still open. */
  source: { kind: "video"; path: string } | { kind: "images"; paths: string[] } | null;
  /** UI settings, returned as-is when the package is opened. */
  session: unknown;
}

export interface PackStats {
  path: string;
  bytes: number;
  files: number;
}

export interface UnpackedProject {
  projectDir: string;
  /** The restored source: the video or the images (never both). */
  video: string | null;
  images: string[] | null;
  session: unknown;
}

export interface SplatInfo {
  count: number;
  shDegree: number;
  bytes: number;
}

export interface SceneFrame {
  center: [number, number, number];
  up: [number, number, number];
  radius: number;
  cameras: [number, number, number][];
  /** Frame index of each camera; gaps mean unregistered frames. */
  frameIndices: number[];
  /** Video lens (cameras.bin): vertical field of view in radians and
   *  width/height. Missing if the model has no readable camera. */
  lens?: { fovY: number; aspect: number };
}

export type TrainEvent =
  | { kind: "step"; iter: number; splats: number }
  | { kind: "eval"; iter: number; psnr: number; ssim: number }
  | { kind: "snapshot"; iter: number; path: string };

export type JobEvent =
  | { type: "stageStarted"; stage: Stage }
  | { type: "progress"; stage: Stage; fraction: number; message: Msg }
  | { type: "log"; stage: Stage; line: string }
  | { type: "framesReady"; frames: FrameInfo[] }
  | { type: "masksReady"; stats: MaskStats; dir: string }
  | { type: "stageSkipped"; stage: Stage; reason: Msg }
  | { type: "posesReady"; stats: ModelStats }
  | { type: "train"; event: TrainEvent }
  | { type: "stageDone"; stage: Stage; seconds: number }
  /** `ply` is null when the job stopped before training. */
  | { type: "finished"; ply: string | null; seconds: number }
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

/** Pose of a clip frame, in COLMAP convention (world → camera). */
export interface RetouchPose {
  qvec: [number, number, number, number];
  tvec: [number, number, number];
  /** Real frame (dataset name) when the pose is exactly that of a photo. */
  anchor: string | null;
}

/** Retouch clip: a trajectory rendered from the splat (reference video),
 *  restored outside the app (ComfyUI) and used for incremental training. */
export interface RetouchClip {
  id: string;
  label: string;
  createdAt: number;
  /** Width and height of the reference video (multiples of 32). */
  size: [number, number];
  fps: number;
  /** PINHOLE, at clip size. */
  intrinsics: { fx: number; fy: number; cx: number; cy: number };
  poses: RetouchPose[];
  /** Paths of the saved videos (for `fileUrl`/`readBlob`). */
  reference: string | null;
  restored: string | null;
  restoredWith: { workflow: string; at: number } | null;
  /** Size of the restored video (same aspect as the reference). Fine-tuning
   *  uses frames at this size, with intrinsics scaled to match. */
  restoredSize?: [number, number] | null;
}

export interface RetouchOptions {
  /** Fine-tuning steps from the current splat. */
  steps: number;
  /** Initial learning rate of positions (Brush uses 2e-5). */
  lrMean: number;
  /** Regular training settings reused (SH degree, resolution, etc.). */
  train: TrainOptions;
  /** Masking the project trained with (frames go in with a transparent
   *  background, as in training); null = no masking. */
  masks: MaskOptions | null;
  /** Maximum restored frames used in fine-tuning (across all clips);
   *  missing = all. */
  maxRestored?: number;
}

export type RetouchEvent =
  | { kind: "progress"; fraction: number; message: Msg }
  | { kind: "train"; event: TrainEvent }
  | { kind: "log"; line: string };

/** The project's real cameras, to build trajectories from. */
export interface RetouchCameras {
  lens: { width: number; height: number; fx: number; fy: number };
  /** In name order (= frame order). */
  cameras: { name: string; qvec: [number, number, number, number]; tvec: [number, number, number] }[];
}

/** Web only, for now. */
export interface RetouchApi {
  cameras(projectDir: string): Promise<RetouchCameras>;
  listClips(projectDir: string): Promise<RetouchClip[]>;
  /** Saves the clip and the given files, keyed by path relative to its folder:
   *  `reference.mp4`, `restored.mp4`, `alpha/00000.png`… */
  saveClip(projectDir: string, clip: RetouchClip, files: Record<string, Blob>): Promise<RetouchClip>;
  deleteClip(projectDir: string, id: string): Promise<void>;
  /** Incremental training with the restored clips; replaces the project's
   *  splat and keeps the previous one for undo. */
  apply(projectDir: string, clipIds: string[], options: RetouchOptions, onEvent: (e: RetouchEvent) => void): Promise<ProjectState>;
  cancel(): Promise<void>;
  /** Reverts to the pre-retouch splat. */
  undo(projectDir: string): Promise<ProjectState>;
}

/** Backend commands: Rust via Tauri on desktop, the TypeScript + WebGPU
 *  pipeline in the web version. */
export interface Api {
  detectTools(): Promise<Tools>;
  probeVideo(path: string): Promise<VideoInfo>;
  /** Images and folders (top level only) → ordered set. */
  probeImages(paths: string[]): Promise<ImageSet>;
  defaultProjectDir(videoPath: string): Promise<string>;
  loadProject(projectDir: string): Promise<ProjectState>;
  sceneFrame(projectDir: string): Promise<SceneFrame>;
  cancelJob(): Promise<void>;
  splatInfo(path: string): Promise<SplatInfo>;
  /** Rejects (or restores) frames by file name. Invalidates poses; restoring
   *  a frame with no mask also invalidates the masks. */
  setFramesRejected(projectDir: string, names: string[], rejected: boolean): Promise<ProjectState>;
  exportSplat(
    projectDir: string,
    options: ExportOptions,
    onProgress: (p: { fraction: number; message: Msg }) => void,
  ): Promise<ExportStats>;
  /** Can run during a job: `include` says what is already done. */
  packProject(request: PackRequest, onProgress: (fraction: number) => void): Promise<PackStats>;
  /** Extracts a package into a new project (never overwrites an existing one). */
  unpackProject(path: string, onProgress: (fraction: number) => void): Promise<UnpackedProject>;
  /** `rmbg`: include the RMBG-1.4 model (only downloaded when chosen). */
  installPlan(rmbg: boolean): Promise<InstallPlan>;
  cancelInstall(): Promise<void>;
  installTools(rmbg: boolean, onEvent: (e: InstallEvent) => void): Promise<void>;
  /** Returns immediately; progress arrives via `onEvent`. */
  startJob(config: JobConfig, onEvent: (e: JobEvent) => void): Promise<void>;
  /** Retouch with ComfyUI; missing where not yet supported (desktop). */
  retouch?: RetouchApi;
}

/** Mask preview on one frame, using the same models as the pipeline. */
export interface MaskPreviewer {
  /** Are the method's models already downloaded (the preview never downloads unasked)? */
  ready(method: MaskMethod): Promise<boolean>;
  /** Mask (255 = subject) at frame size; downloads the models if needed.
   *  `key` identifies the frame, to reuse whatever does not depend on the prompt. */
  mask(
    key: string,
    frame: ImageBitmap,
    masks: MaskOptions,
    onProgress?: (done: number, total: number) => void,
  ): Promise<Uint8Array>;
  /** Frees the GPU (the real masking run will need it). */
  release(): void;
}

/** What differs between the desktop app and the web version, beyond the
 *  commands. Paths are opaque to the UI: on the web they point into OPFS. */
export interface Host {
  api: Api;
  /** Servable URL for a file (with optional cache-buster). */
  fileUrl(path: string, bust?: string | number): string;
  /** One video or several images; empty list if cancelled. */
  pickFiles(extensions: string[]): Promise<string[]>;
  /** Frames folder, for `probeImages`; empty list if cancelled. May throw
   *  when the folder has no images. */
  pickFramesFolder(): Promise<string[]>;
  pickFolder(defaultPath?: string): Promise<string | null>;
  /** Where to save the project package; null if cancelled. */
  pickPackageDest(projectDir: string): Promise<string | null>;
  /** Shows the file in the file manager (on the web, downloads it). */
  reveal(path: string): Promise<void>;
  /** Files and folders dropped on the window; returns an unsubscribe function. */
  onDrop(handler: (paths: string[]) => void): () => void;
  /** File contents (on desktop, via the `asset:` protocol). */
  readBlob(path: string): Promise<Blob>;
  /** Share-link server (worker/index.ts): the page's own origin on the web,
   *  the published address on desktop. */
  shareOrigin: string;
  /** Web only: notifies when a new app version has been downloaded (service
   *  worker); returns an unsubscribe function. */
  onUpdate?(handler: () => void): () => void;
  /** Activates the new version and reloads the page. */
  applyUpdate?(): void;
  /** Web only, for now: mask preview in the panel. */
  maskPreview?: MaskPreviewer;
}

// Set once by the entry point (src/main.tsx or web/src/main.tsx), before the
// first render.
export let host: Host;
export let api: Api;

export function setHost(h: Host) {
  host = h;
  api = h.api;
}

export const fileUrl = (path: string, bust?: string | number) => host.fileUrl(path, bust);
