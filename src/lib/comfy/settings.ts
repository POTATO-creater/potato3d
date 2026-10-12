// Integração com o ComfyUI: URL e workflows, só neste navegador.

import type { ApiGraph, WorkflowIO } from "./workflow.ts";

/** "restore": recebe o vídeo renderizado do splat e devolve o restaurado.
 *  (Previsto: "orbit", imagem → vídeo orbital 360.) */
export type WorkflowKind = "restore";

export interface ComfyWorkflow {
  id: string;
  name: string;
  kind: WorkflowKind;
  graph: ApiGraph;
  io: WorkflowIO;
  importedAt: number;
}

export interface ComfySettings {
  url: string;
  workflows: ComfyWorkflow[];
}

const STORE_KEY = "splatforge.comfy.v1";

export const DEFAULT_COMFY: ComfySettings = { url: "http://127.0.0.1:8188", workflows: [] };

export function loadComfy(): ComfySettings {
  try {
    const raw = JSON.parse(localStorage.getItem(STORE_KEY) ?? "null") as Partial<ComfySettings> | null;
    if (!raw) return DEFAULT_COMFY;
    return {
      url: typeof raw.url === "string" ? raw.url : DEFAULT_COMFY.url,
      workflows: Array.isArray(raw.workflows) ? raw.workflows.filter((w) => w && w.graph && w.io) : [],
    };
  } catch {
    return DEFAULT_COMFY;
  }
}

/** Devolve false se o navegador recusou (armazenamento cheio ou bloqueado). */
export function saveComfy(s: ComfySettings): boolean {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(s));
    return true;
  } catch {
    return false;
  }
}

export const workflowsOf = (s: ComfySettings, kind: WorkflowKind) => s.workflows.filter((w) => w.kind === kind);
