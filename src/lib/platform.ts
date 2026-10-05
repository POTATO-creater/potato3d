import type { Stage, Tools } from "./api";

export type Os = Tools["os"];

/** Rótulo curto da ferramenta de cada etapa, conforme o sistema e a GPU. */
export function stageToolLabels(tools: Tools | null): Record<Stage, string> {
  const os = tools?.os ?? "macos";
  const decode = os === "macos" ? "VideoToolbox" : "decodificação por hardware";
  const gpu = os === "macos" ? "Metal" : os === "windows" ? "Vulkan/DX12" : "Vulkan";
  return {
    frames: `ffmpeg · ${decode}`,
    masks: os === "macos" ? "Vision · GPU" : "BiRefNet · DirectML",
    poses: `COLMAP · ${tools?.colmapCuda ? "CUDA" : "CPU"} · mapeador global`,
    train: `Brush · ${gpu}`,
  };
}

/** Comando de instalação sugerido para uma ferramenta que falta. */
export function installCommand(tool: "ffmpeg" | "colmap" | "brush" | "model", os: Os): string {
  if (tool === "model") return os === "windows" ? "Instalar ferramentas" : "macOS 14+";
  if (tool === "brush" || os === "windows") return "pnpm setup:tools";
  if (os === "macos") return `brew install ${tool}`;
  return `instale ${tool} pelo gerenciador de pacotes`;
}

export const basename = (p: string) => p.split(/[\\/]/).pop() ?? p;

/** Nome do gerenciador de arquivos do sistema. */
export const fileManager = (os: Os | undefined) =>
  os === "windows" ? "Explorer" : os === "linux" ? "gerenciador de arquivos" : "Finder";
