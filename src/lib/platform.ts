import type { MaskOptions, Stage, Tools } from "./api";
import type { Messages } from "../i18n";

export type Os = Tools["os"];

/** Rótulo curto da ferramenta de cada etapa, conforme o sistema e a GPU. */
export function stageToolLabels(t: Messages, tools: Tools | null, masks?: MaskOptions, images?: boolean): Record<Stage, string> {
  const os = tools?.os ?? "macos";
  const p = t.platform;
  if (os === "web")
    return {
      frames: images ? p.yourImages : p.browserDecoder,
      masks: masks?.method === "chroma" ? p.solidColor : masks?.method === "sam2" ? "SAM 2.1 · WebGPU" : "RMBG-1.4 · WebGPU",
      poses: p.webPoses,
      train: "Brush · WebGPU",
    };
  const decode = os === "macos" ? "VideoToolbox" : p.hwDecode;
  const gpu = os === "macos" ? "Metal" : os === "windows" ? "Vulkan/DX12" : "Vulkan";
  return {
    frames: images ? p.yourImages : `ffmpeg · ${decode}`,
    masks:
      masks?.method === "chroma"
        ? p.solidColor
        : os !== "windows"
          ? "Vision · GPU"
          : masks?.method === "general"
            ? "BiRefNet · DirectML"
            : "MODNet · DirectML",
    poses: `COLMAP · ${tools?.colmapCuda ? "CUDA" : "CPU"} · ${p.globalMapper}`,
    train: `Brush · ${gpu}`,
  };
}

/** Comando de instalação sugerido para uma ferramenta que falta. */
export function installCommand(t: Messages, tool: "ffmpeg" | "colmap" | "brush" | "model", os: Os): string {
  const p = t.platform;
  if (os === "web") return tool === "brush" ? p.webgpuChrome : p.webgpuBrowser;
  if (tool === "model") return os === "windows" ? p.installTools : "macOS 14+";
  if (tool === "brush" || os === "windows") return "pnpm setup:tools";
  if (os === "macos") return `brew install ${tool}`;
  return p.packageManager(tool);
}

export const basename = (p: string) => p.split(/[\\/]/).pop() ?? p;

/** Nome do gerenciador de arquivos do sistema. */
export const fileManager = (t: Messages, os: Os | undefined) =>
  os === "windows" ? "Explorer" : os === "linux" ? t.platform.fileManager : "Finder";

/** Rótulo do botão que leva a um arquivo: no web os arquivos ficam no
 *  armazenamento do navegador, então o botão baixa. */
export const revealLabel = (t: Messages, os: Os | undefined, what?: string) =>
  os === "web" ? t.platform.download(what) : t.platform.reveal(fileManager(t, os), what);

/** O que cada etapa faz, para a tela inicial. */
export function stageBlurbs(t: Messages, os: Os | undefined): Record<Stage, string> {
  return os === "web" ? t.platform.blurbs.web : t.platform.blurbs.desktop;
}
