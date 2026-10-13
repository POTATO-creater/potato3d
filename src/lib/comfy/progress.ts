// Progresso de uma execução do ComfyUI a partir das mensagens do WebSocket
// (/ws?clientId=…). Função pura: o cliente só repassa as mensagens.
//
// Mensagens usadas (todas com data.prompt_id, menos status):
//   execution_start · execution_cached {nodes} · executing {node|null}
//   progress {value, max, node} · executed {node} · execution_success
//   execution_error {node_id, node_type, exception_message}
//   execution_interrupted

export interface ComfyProgress {
  promptId: string;
  /** Nós do grafo enviado; alguns nunca rodam (ramos de um switch). */
  total: number;
  done: string[];
  /** Nó rodando agora e o passo dele (amostradores mandam value/max). */
  node: string | null;
  step: { value: number; max: number } | null;
  /** Posição na fila antes de começar (status.exec_info.queue_remaining). */
  queue: number | null;
  status: "queued" | "running" | "success" | "error" | "interrupted";
  error: { node: string | null; nodeType: string | null; message: string } | null;
}

export function initialProgress(promptId: string, total: number): ComfyProgress {
  return { promptId, total, done: [], node: null, step: null, queue: null, status: "queued", error: null };
}

const finished = (p: ComfyProgress) => p.status === "success" || p.status === "error" || p.status === "interrupted";

function markDone(done: string[], ...ids: (string | null | undefined)[]) {
  const set = new Set(done);
  for (const id of ids) if (id) set.add(String(id));
  return [...set];
}

export function reduceProgress(p: ComfyProgress, msg: { type: string; data?: Record<string, unknown> }): ComfyProgress {
  const d = msg.data ?? {};
  if (msg.type === "status") {
    const info = (d.status as { exec_info?: { queue_remaining?: number } } | undefined)?.exec_info;
    return p.status === "queued" && typeof info?.queue_remaining === "number" ? { ...p, queue: info.queue_remaining } : p;
  }
  if (d.prompt_id !== p.promptId || finished(p)) return p;
  switch (msg.type) {
    case "execution_start":
      return { ...p, status: "running", queue: null };
    case "execution_cached":
      return { ...p, status: "running", done: markDone(p.done, ...((d.nodes as string[]) ?? [])) };
    case "executing": {
      const node = d.node == null ? null : String(d.node);
      // node null = fim da execução (protocolo antigo, antes do execution_success).
      if (node === null) return { ...p, done: markDone(p.done, p.node), node: null, step: null, status: "success" };
      return { ...p, status: "running", done: markDone(p.done, p.node === node ? null : p.node), node, step: null };
    }
    case "progress":
      return { ...p, status: "running", node: d.node == null ? p.node : String(d.node), step: { value: Number(d.value), max: Number(d.max) } };
    case "executed":
      return { ...p, done: markDone(p.done, d.node as string), step: d.node === p.node ? null : p.step };
    case "execution_success":
      return { ...p, done: markDone(p.done, p.node), node: null, step: null, status: "success" };
    case "execution_error":
      return {
        ...p,
        status: "error",
        error: {
          node: d.node_id == null ? null : String(d.node_id),
          nodeType: (d.node_type as string) ?? null,
          message: String(d.exception_message ?? "").trim() || "erro no ComfyUI",
        },
      };
    case "execution_interrupted":
      return { ...p, status: "interrupted" };
    default:
      return p;
  }
}

/** Fração 0–1: nós concluídos mais a parte do nó atual. Vai a 1 só no fim,
 *  porque nem todo nó do grafo roda. */
export function progressFraction(p: ComfyProgress): number {
  if (p.status === "success") return 1;
  const partial = p.step && p.step.max > 0 ? Math.min(p.step.value / p.step.max, 1) : 0;
  const running = p.node && !p.done.includes(p.node) ? partial : 0;
  return Math.min((p.done.length + running) / Math.max(p.total, 1), 0.99);
}
