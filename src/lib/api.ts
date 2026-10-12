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

/** Frames já extraídos (imagens soltas ou uma pasta), no lugar do vídeo. */
export interface ImageSet {
  /** Na ordem da órbita: ordem natural dos nomes. */
  paths: string[];
  /** Pasta das imagens: dá nome ao projeto. */
  dir: string;
  /** Da primeira imagem, já girada conforme o EXIF. */
  width: number;
  height: number;
}

/** Extensões aceitas como frames. */
export const IMAGE_EXT = ["jpg", "jpeg", "png", "webp"];

export interface FrameOptions {
  targetFrames: number;
  maxSide: number;
  oversample: number;
  trimStart: number;
  trimEnd: number | null;
}

/** auto: Vision (macOS) / MODNet (Windows) · people: MODNet · general: BiRefNet · rmbg: RMBG-1.4 ·
 *  sam2: SAM 2.1 com propagação entre frames (só no web) · chroma: cor sólida */
export type MaskMethod = "auto" | "people" | "general" | "rmbg" | "sam2" | "chroma";

/** Onde está o sujeito, para o SAM 2.1, em coordenadas normalizadas (0–1) do quadro. */
export interface SubjectPrompt {
  /** Instante do vídeo em que foi marcado (s); null = primeiro frame. */
  time: number | null;
  /** Com imagens no lugar do vídeo: o nome da imagem marcada. */
  image?: string | null;
  /** Caixa [x0, y0, x1, y1]. */
  box: [number, number, number, number] | null;
  /** label 1 = sujeito, 0 = fundo. */
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
  /** Também usa as máscaras no COLMAP (ignora features do fundo). */
  forPoses: boolean;
  method: MaskMethod;
  chroma: ChromaOptions;
  /** Tira do sujeito o reflexo de um fundo verde ou azul (qualquer método). */
  despill: boolean;
  /** 0–100 */
  despillStrength: number;
  /** SAM 2.1: sujeito marcado no vídeo; null = o que estiver no centro do primeiro frame. */
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
  /** Câmeras forçadas no mesmo plano. Ausente em configurações antigas. */
  flatOrbit?: boolean;
  /** Frames sem pose registrada ganham uma inferida pelos vizinhos e entram
   *  no treino. Ausente em configurações antigas. */
  fillMissing?: boolean;
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
  /** Poda as gaussianas cuja contribuição para a perda, somada nas vistas
   *  da janela de refinamento, é negativa (atrapalham mais que ajudam:
   *  fantasmas e contornos duplicados). Mínimo de vistas para julgar;
   *  omitido ou 0 = desligado. */
  contribPruneMinViews?: number;
  /** Quão nítido o prejuízo precisa ser: saldo < −razão × soma dos |efeitos|. */
  contribPruneRatio?: number;
  /** Fração do treino antes de começar a podar por contribuição. */
  contribPruneStart?: number;
  /** Regularização do DeG: peso da média de 1 − α (empurra para opaco). */
  opacSolidWeight?: number;
  /** Peso da média de α(1 − α) (empurra a opacidade para 0 ou 1). */
  opacBinaryWeight?: number;
  /** Peso do tamanho médio das gaussianas, relativo à cena (compactas). */
  volumeRegWeight?: number;
  /** Taxa de aprendizado inicial das posições; omitido = padrão do Brush
   *  (2e-5). O ajuste do retoque usa bem menos, para não desmontar o splat. */
  lrMean?: number;
  /** Passo em que o crescimento de splats para; omitido = padrão do Brush. */
  growthStopIter?: number;
}

export interface Settings {
  frames: Omit<FrameOptions, "trimStart" | "trimEnd">;
  poses: PoseOptions;
  train: TrainOptions;
}

export interface JobConfig extends Omit<Settings, "frames"> {
  /** O vídeo, ou a pasta das imagens quando `images` vem preenchido. */
  videoPath: string;
  /** Frames já extraídos, na ordem da órbita, em vez do vídeo. */
  images: string[] | null;
  projectDir: string;
  fromStage: Stage;
  /** Última etapa a rodar (inclusive); ausente = até o treino. */
  toStage?: Stage;
  frames: FrameOptions;
  masks: MaskOptions;
}

export interface FrameInfo {
  path: string;
  time: number;
  sharpness: number;
  /** Nome da imagem de origem quando os frames vieram de imagens (aí `time`
   *  não se aplica). */
  source?: string;
  /** Recusado à mão: o arquivo sai do dataset (fica em `rejected/`) e não
   *  entra no recorte, nas poses nem no treino. */
  rejected?: boolean;
}

export interface ModelStats {
  registered: number;
  total: number;
  points: number;
  reprojectionError: number | null;
  mapper: string;
  /** Poses inferidas pelos vizinhos (frames que o SfM não registrou). */
  inferred?: number;
  /** Dessas, as refinadas contra os pontos 3D. */
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
  /** Retoque aplicado: `base` é o splat de antes (para desfazer e para
   *  reaplicar sem acumular ajustes); `clips`, os clipes usados. */
  retouch?: { base: string; clips: string[]; appliedAt: number } | null;
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

/** Pacote do projeto (.gorbit): um ZIP com o que já foi processado, para
 *  abrir em outra máquina (ou no outro app) como um projeto em cache. */
export const PACKAGE_EXT = "gorbit";

export interface PackRequest {
  projectDir: string;
  /** De `Host.pickPackageDest`. */
  dest: string;
  /** Etapas prontas a levar além dos frames (as de um job em andamento, não). */
  include: { masks: boolean; poses: boolean; splat: boolean };
  /** Vídeo ou imagens de origem, se ainda estiverem abertos. */
  source: { kind: "video"; path: string } | { kind: "images"; paths: string[] } | null;
  /** Ajustes da interface, devolvidos como vieram ao abrir o pacote. */
  session: unknown;
}

export interface PackStats {
  path: string;
  bytes: number;
  files: number;
}

export interface UnpackedProject {
  projectDir: string;
  /** A fonte restaurada: o vídeo ou as imagens (nunca os dois). */
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
  /** Índice do frame de cada câmera; saltos indicam frames não registrados. */
  frameIndices: number[];
  /** Lente do vídeo (cameras.bin): campo de visão vertical em radianos e
   *  largura/altura. Ausente se o modelo não tiver câmera legível. */
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
  /** `ply` null quando o job parou antes do treino. */
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

/** Pose de um frame de clipe, em convenção COLMAP (mundo → câmera). */
export interface RetouchPose {
  qvec: [number, number, number, number];
  tvec: [number, number, number];
  /** Frame real (nome no dataset) quando a pose é exatamente a de uma foto. */
  anchor: string | null;
}

/** Clipe de retoque: trajetória renderizada do splat (vídeo de referência),
 *  restaurada fora do app (ComfyUI) e usada no treino incremental. */
export interface RetouchClip {
  id: string;
  label: string;
  createdAt: number;
  /** Largura e altura do vídeo de referência (múltiplos de 32). */
  size: [number, number];
  fps: number;
  /** PINHOLE, no tamanho do clipe. */
  intrinsics: { fx: number; fy: number; cx: number; cy: number };
  poses: RetouchPose[];
  /** Caminhos dos vídeos já gravados (para `fileUrl`/`readBlob`). */
  reference: string | null;
  restored: string | null;
  restoredWith: { workflow: string; at: number } | null;
}

export interface RetouchOptions {
  /** Passos do ajuste a partir do splat atual. */
  steps: number;
  /** Taxa de aprendizado das posições no início (o Brush usa 2e-5). */
  lrMean: number;
  /** Ajustes do treino normal reaproveitados (grau SH, resolução etc.). */
  train: TrainOptions;
  /** Recorte com que o projeto treinou (os frames entram com fundo
   *  transparente, como no treino); null = sem recorte. */
  masks: MaskOptions | null;
}

export type RetouchEvent =
  | { kind: "progress"; fraction: number; message: Msg }
  | { kind: "train"; event: TrainEvent }
  | { kind: "log"; line: string };

/** Câmeras reais do projeto, para montar as trajetórias. */
export interface RetouchCameras {
  lens: { width: number; height: number; fx: number; fy: number };
  /** Em ordem de nome (= ordem dos frames). */
  cameras: { name: string; qvec: [number, number, number, number]; tvec: [number, number, number] }[];
}

/** Só no web, por enquanto. */
export interface RetouchApi {
  cameras(projectDir: string): Promise<RetouchCameras>;
  listClips(projectDir: string): Promise<RetouchClip[]>;
  /** Grava o clipe e os arquivos dados, por caminho relativo à pasta dele:
   *  `reference.mp4`, `restored.mp4`, `alpha/00000.png`… */
  saveClip(projectDir: string, clip: RetouchClip, files: Record<string, Blob>): Promise<RetouchClip>;
  deleteClip(projectDir: string, id: string): Promise<void>;
  /** Treino incremental com os clipes restaurados; substitui o splat do
   *  projeto e guarda o anterior para desfazer. */
  apply(projectDir: string, clipIds: string[], options: RetouchOptions, onEvent: (e: RetouchEvent) => void): Promise<ProjectState>;
  cancel(): Promise<void>;
  /** Volta ao splat de antes do retoque. */
  undo(projectDir: string): Promise<ProjectState>;
}

/** Comandos do backend: o Rust via Tauri no desktop, o pipeline em
 *  TypeScript + WebGPU na versão web. */
export interface Api {
  detectTools(): Promise<Tools>;
  probeVideo(path: string): Promise<VideoInfo>;
  /** Imagens e pastas (só o primeiro nível) → conjunto ordenado. */
  probeImages(paths: string[]): Promise<ImageSet>;
  defaultProjectDir(videoPath: string): Promise<string>;
  loadProject(projectDir: string): Promise<ProjectState>;
  sceneFrame(projectDir: string): Promise<SceneFrame>;
  cancelJob(): Promise<void>;
  splatInfo(path: string): Promise<SplatInfo>;
  /** Recusa (ou aceita de volta) frames pelo nome do arquivo. Invalida as
   *  poses; aceitar um frame sem recorte invalida também o recorte. */
  setFramesRejected(projectDir: string, names: string[], rejected: boolean): Promise<ProjectState>;
  exportSplat(
    projectDir: string,
    options: ExportOptions,
    onProgress: (p: { fraction: number; message: Msg }) => void,
  ): Promise<ExportStats>;
  /** Pode rodar durante um job: `include` diz o que já está pronto. */
  packProject(request: PackRequest, onProgress: (fraction: number) => void): Promise<PackStats>;
  /** Extrai um pacote num projeto novo (nunca sobrescreve um existente). */
  unpackProject(path: string, onProgress: (fraction: number) => void): Promise<UnpackedProject>;
  /** `rmbg`: incluir o modelo RMBG-1.4 (só baixado quando escolhido). */
  installPlan(rmbg: boolean): Promise<InstallPlan>;
  cancelInstall(): Promise<void>;
  installTools(rmbg: boolean, onEvent: (e: InstallEvent) => void): Promise<void>;
  /** Volta logo; o andamento chega por `onEvent`. */
  startJob(config: JobConfig, onEvent: (e: JobEvent) => void): Promise<void>;
  /** Retoque com o ComfyUI; ausente onde ainda não há suporte (desktop). */
  retouch?: RetouchApi;
}

/** Prévia do recorte num quadro, com os mesmos modelos do pipeline. */
export interface MaskPreviewer {
  /** Os modelos do método já estão baixados (a prévia não baixa sem pedir)? */
  ready(method: MaskMethod): Promise<boolean>;
  /** Máscara (255 = sujeito) no tamanho do quadro; baixa os modelos se preciso.
   *  `key` identifica o quadro, para reaproveitar o que não depende do prompt. */
  mask(
    key: string,
    frame: ImageBitmap,
    masks: MaskOptions,
    onProgress?: (done: number, total: number) => void,
  ): Promise<Uint8Array>;
  /** Libera a GPU (o recorte de verdade vai usá-la). */
  release(): void;
}

/** O que muda entre o app desktop e a versão web, além dos comandos. Os
 *  caminhos são opacos para a interface: no web apontam para o OPFS. */
export interface Host {
  api: Api;
  /** URL servível de um arquivo (com cache-buster opcional). */
  fileUrl(path: string, bust?: string | number): string;
  /** Um vídeo ou várias imagens; lista vazia se cancelado. */
  pickFiles(extensions: string[]): Promise<string[]>;
  /** Pasta de frames, para `probeImages`; lista vazia se cancelado. Pode
   *  falhar quando a pasta não tem imagens. */
  pickFramesFolder(): Promise<string[]>;
  pickFolder(defaultPath?: string): Promise<string | null>;
  /** Onde salvar o pacote do projeto; null se cancelado. */
  pickPackageDest(projectDir: string): Promise<string | null>;
  /** Mostra o arquivo no gerenciador de arquivos (no web, baixa). */
  reveal(path: string): Promise<void>;
  /** Arquivos e pastas soltos na janela; devolve a função que para de ouvir. */
  onDrop(handler: (paths: string[]) => void): () => void;
  /** Conteúdo de um arquivo (no desktop, pelo protocolo `asset:`). */
  readBlob(path: string): Promise<Blob>;
  /** Servidor dos links de compartilhamento (worker/index.ts): a própria
   *  origem no web, o endereço publicado no desktop. */
  shareOrigin: string;
  /** Só no web: avisa quando uma versão nova do app foi baixada (service
   *  worker); devolve a função que para de ouvir. */
  onUpdate?(handler: () => void): () => void;
  /** Põe a versão nova no ar e recarrega a página. */
  applyUpdate?(): void;
  /** Só no web, por enquanto: prévia do recorte no painel. */
  maskPreview?: MaskPreviewer;
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
