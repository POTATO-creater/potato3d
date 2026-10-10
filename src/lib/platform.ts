import type { MaskOptions, Stage, Tools } from "./api";

export type Os = Tools["os"];

/** Rótulo curto da ferramenta de cada etapa, conforme o sistema e a GPU. */
export function stageToolLabels(tools: Tools | null, masks?: MaskOptions): Record<Stage, string> {
  const os = tools?.os ?? "macos";
  if (os === "web")
    return {
      frames: "decodificador do navegador",
      masks: masks?.method === "chroma" ? "cor sólida" : "RMBG-1.4 · WebGPU",
      poses: "AKAZE · WebGPU · SfM incremental",
      train: "Brush · WebGPU",
    };
  const decode = os === "macos" ? "VideoToolbox" : "decodificação por hardware";
  const gpu = os === "macos" ? "Metal" : os === "windows" ? "Vulkan/DX12" : "Vulkan";
  return {
    frames: `ffmpeg · ${decode}`,
    masks:
      masks?.method === "chroma"
        ? "cor sólida"
        : os !== "windows"
          ? "Vision · GPU"
          : masks?.method === "general"
            ? "BiRefNet · DirectML"
            : "MODNet · DirectML",
    poses: `COLMAP · ${tools?.colmapCuda ? "CUDA" : "CPU"} · mapeador global`,
    train: `Brush · ${gpu}`,
  };
}

/** Comando de instalação sugerido para uma ferramenta que falta. */
export function installCommand(tool: "ffmpeg" | "colmap" | "brush" | "model", os: Os): string {
  if (os === "web") return tool === "brush" ? "Chrome ou Edge com WebGPU" : "navegador com WebGPU";
  if (tool === "model") return os === "windows" ? "Instalar ferramentas" : "macOS 14+";
  if (tool === "brush" || os === "windows") return "pnpm setup:tools";
  if (os === "macos") return `brew install ${tool}`;
  return `instale ${tool} pelo gerenciador de pacotes`;
}

export const basename = (p: string) => p.split(/[\\/]/).pop() ?? p;

/** Nome do gerenciador de arquivos do sistema. */
export const fileManager = (os: Os | undefined) =>
  os === "windows" ? "Explorer" : os === "linux" ? "gerenciador de arquivos" : "Finder";

/** Rótulo do botão que leva a um arquivo: no web os arquivos ficam no
 *  armazenamento do navegador, então o botão baixa. */
export const revealLabel = (os: Os | undefined, what?: string) =>
  os === "web" ? `Baixar${what ? ` ${what}` : ""}` : `Mostrar${what ? ` ${what}` : ""} no ${fileManager(os)}`;

/** O que cada etapa faz, para a tela inicial. */
export function stageBlurbs(os: Os | undefined): Record<Stage, string> {
  return os === "web"
    ? {
        frames: "o navegador decodifica o vídeo e escolhe os frames mais nítidos.",
        masks: "isola o sujeito para o splat sair sem o fundo.",
        poses: "features AKAZE, matching circular na GPU (fecha a órbita) e SfM incremental.",
        train: "Brush em WebAssembly + WebGPU, com prévias ao vivo aqui.",
      }
    : {
        frames: "ffmpeg decodifica por hardware e escolhe os frames mais nítidos.",
        masks: "isola o sujeito para o splat sair sem o fundo.",
        poses: "COLMAP com matching circular (fecha a órbita) e mapeador global.",
        train: "Brush na GPU, com prévias ao vivo aqui.",
      };
}
