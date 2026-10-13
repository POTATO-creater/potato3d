// Integração com o ComfyUI: URL e workflows, só neste navegador.

import type { ApiGraph, ParamValue, WorkflowIO, WorkflowParam } from "./workflow.ts";

/** "restore": recebe o vídeo renderizado do splat e devolve o restaurado.
 *  (Previsto: "orbit", imagem → vídeo orbital 360.) */
export type WorkflowKind = "restore";

export interface ComfyWorkflow {
  id: string;
  name: string;
  kind: WorkflowKind;
  graph: ApiGraph;
  io: WorkflowIO;
  /** Campos que o usuário expôs para editar no app antes de enviar. */
  params?: WorkflowParam[];
  importedAt: number;
}

export interface ComfySettings {
  url: string;
  workflows: ComfyWorkflow[];
}

const STORE_KEY = "splatforge.comfy.v1";
/** Evento disparado em `window` quando a configuração muda. */
export const COMFY_CHANGED = "splatforge:comfy-changed";

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
    // Quem lê a configuração na tela (o painel de retoque) se atualiza.
    window.dispatchEvent(new Event(COMFY_CHANGED));
    return true;
  } catch {
    return false;
  }
}

export const workflowsOf = (s: ComfySettings, kind: WorkflowKind) => s.workflows.filter((w) => w.kind === kind);

/** Valores dos campos personalizados, por workflow e por campo. */
type ParamValues = Record<string, Record<string, ParamValue>>;
const VALUES_KEY = "splatforge.comfyParams.v1";

export function loadParamValues(workflowId: string): Record<string, ParamValue> {
  try {
    const all = JSON.parse(localStorage.getItem(VALUES_KEY) ?? "{}") as ParamValues;
    return all && typeof all === "object" && all[workflowId] && typeof all[workflowId] === "object" ? all[workflowId] : {};
  } catch {
    return {};
  }
}

/** `undefined` num campo volta ao padrão do workflow. */
export function saveParamValues(workflowId: string, values: Record<string, ParamValue | undefined>): boolean {
  try {
    const all = (JSON.parse(localStorage.getItem(VALUES_KEY) ?? "{}") as ParamValues) || {};
    all[workflowId] = Object.fromEntries(Object.entries(values).filter(([, v]) => v !== undefined)) as Record<string, ParamValue>;
    localStorage.setItem(VALUES_KEY, JSON.stringify(all));
    return true;
  } catch {
    return false;
  }
}
