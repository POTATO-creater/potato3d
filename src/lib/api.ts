export type Stage = "frames" | "masks" | "poses" | "train";
export const STAGES: Stage[] = ["frames", "masks", "poses", "train"];

/** Texto traduzível vindo do backend (ou do reducer): chave do catálogo `msg`
 *  de src/i18n + parâmetros. Um parâmetro pode ser outra `Msg`. */
export interface Msg {
  key: string;
  args?: Record<string, MsgArg>;
}
export type MsgArg = string | number | Msg;

export interface Tools {
  ffmpeg: string | null;
  ffprobe: string | null;
  colmap: string | null;
  /** COLMAP com CUDA: extração, matching e bundle adjustment na GPU. */
  colmapCuda: boolean;
  brush: string | null;
  /** macOS: recortador Vision (sidecar). */
  masker: string | null;
  /** Windows: modelos de recorte baixados pelo instalador. */
  birefnetModel: string | null;
  modnetModel: string | null;
  /** Windows e macOS (Apple Silicon): RMBG-1.4, baixado sob demanda. */
  rmbgModel: string | null;
  /** "web": tudo roda no navegador (web/); as ferramentas acima viram rótulos. */
  os: "macos" | "windows" | "linux" | "web";
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

/** auto: Vision (macOS) / MODNet (Windows) · people: MODNet · general: BiRefNet · rmbg: RMBG-1.4 · chroma: cor sólida */
export type MaskMethod = "auto" | "people" | "general" | "rmbg" | "chroma";

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
  /** Também usa as máscaras no COLMAP (ignora features do fundo). */
  forPoses: boolean;
  method: MaskMethod;
  chroma: ChromaOptions;
  /** Tira do sujeito o reflexo de um fundo verde ou azul (qualquer método). */
  despill: boolean;
  /** 0–100 */
  despillStrength: number;
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
  /** Câmeras forçadas no mesmo plano. Ausente em configurações antigas. */
  flatOrbit?: boolean;
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
  lastExport?: ExportStats | null;
}

export interface ExportOptions {
  format: "sog" | "ply";
  /** Fração removida por fusão de gaussianas parecidas (0–0,9). */
  decimate: number;
  /** Opacidade mínima (0–1; 0 desliga a poda). */
  minOpacity: number;
  /** Grau SH na saída; null mantém o do treino. */
  shDegree: number | null;
}

export interface ExportStats {
  path: string;
  inputCount: number;
  /** Depois da poda por opacidade. */
  prunedCount: number;
  outputCount: number;
  inputBytes: number;
  outputBytes: number;
  shDegree: number;
  seconds: number;
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
  /** Índice do frame de cada câmera; saltos indicam frames não registrados. */
  frameIndices: number[];
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

/** Comandos do backend: o Rust via Tauri no desktop, o pipeline em
 *  TypeScript + WebGPU na versão web. */
export interface Api {
  detectTools(): Promise<Tools>;
  probeVideo(path: string): Promise<VideoInfo>;
  defaultProjectDir(videoPath: string): Promise<string>;
  loadProject(projectDir: string): Promise<ProjectState>;
  sceneFrame(projectDir: string): Promise<SceneFrame>;
  cancelJob(): Promise<void>;
  splatInfo(path: string): Promise<SplatInfo>;
  exportSplat(
    projectDir: string,
    options: ExportOptions,
    onProgress: (p: { fraction: number; message: Msg }) => void,
  ): Promise<ExportStats>;
  /** `rmbg`: incluir o modelo RMBG-1.4 (só baixado quando escolhido). */
  installPlan(rmbg: boolean): Promise<InstallPlan>;
  cancelInstall(): Promise<void>;
  installTools(rmbg: boolean, onEvent: (e: InstallEvent) => void): Promise<void>;
  /** Volta logo; o andamento chega por `onEvent`. */
  startJob(config: JobConfig, onEvent: (e: JobEvent) => void): Promise<void>;
}

/** O que muda entre o app desktop e a versão web, além dos comandos. Os
 *  caminhos são opacos para a interface: no web apontam para o OPFS. */
export interface Host {
  api: Api;
  /** URL servível de um arquivo (com cache-buster opcional). */
  fileUrl(path: string, bust?: string | number): string;
  pickVideo(extensions: string[]): Promise<string | null>;
  pickFolder(defaultPath?: string): Promise<string | null>;
  /** Mostra o arquivo no gerenciador de arquivos (no web, baixa). */
  reveal(path: string): Promise<void>;
  /** Arquivos soltos na janela; devolve a função que para de ouvir. */
  onDrop(handler: (paths: string[]) => void): () => void;
  /** Só no web: avisa quando uma versão nova do app foi baixada (service
   *  worker); devolve a função que para de ouvir. */
  onUpdate?(handler: () => void): () => void;
  /** Põe a versão nova no ar e recarrega a página. */
  applyUpdate?(): void;
}

// Definidos uma vez pelo ponto de entrada (src/main.tsx ou web/src/main.tsx),
// antes do primeiro render.
export let host: Host;
export let api: Api;

export function setHost(h: Host) {
  host = h;
  api = h.api;
}

export const fileUrl = (path: string, bust?: string | number) => host.fileUrl(path, bust);
