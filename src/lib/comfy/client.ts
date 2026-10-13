// ComfyUI HTTP + WebSocket client, called straight from the browser. ComfyUI
// must run with --enable-cors-header (without it the browser blocks the
// responses and ComfyUI itself rejects cross-origin requests on loopback).

import type { ApiGraph } from "./workflow.ts";
import { initialProgress, reduceProgress, type ComfyProgress } from "./progress.ts";

export type ComfyErrorReason = "network" | "http" | "validation" | "execution" | "interrupted" | "noOutput";

export class ComfyError extends Error {
  readonly reason: ComfyErrorReason;
  readonly detail: string;
  constructor(reason: ComfyErrorReason, detail = "") {
    super(detail || reason);
    this.reason = reason;
    this.detail = detail;
  }
}

export interface ComfyStats {
  version: string | null;
  device: string | null;
  vramGb: number | null;
}

/** Output file as /history describes it. */
export interface ComfyFile {
  filename: string;
  subfolder: string;
  type: string;
}

/** Accepts "127.0.0.1:8188", "http://host:8188/", etc. */
export function normalizeUrl(raw: string): string {
  let url = raw.trim();
  if (!url) return "";
  if (!/^https?:\/\//i.test(url)) url = `http://${url}`;
  return url.replace(/\/+$/, "");
}

export const wsUrl = (base: string, clientId: string) =>
  `${base.replace(/^http/i, "ws")}/ws?clientId=${encodeURIComponent(clientId)}`;

/** A node's outputs in /history: videos come under "images" (SaveVideo, with
 *  animated: [true]), "videos" or "gifs" (VideoHelperSuite). */
export function outputFiles(output: Record<string, unknown> | undefined): ComfyFile[] {
  if (!output) return [];
  const out: ComfyFile[] = [];
  for (const key of ["videos", "gifs", "images"]) {
    const list = output[key];
    if (!Array.isArray(list)) continue;
    for (const f of list) {
      if (f && typeof f === "object" && typeof (f as ComfyFile).filename === "string") {
        const { filename, subfolder = "", type = "output" } = f as ComfyFile;
        out.push({ filename, subfolder, type });
      }
    }
  }
  return out;
}

const POLL_MS = 2000;

export class ComfyClient {
  readonly base: string;
  constructor(url: string) {
    this.base = normalizeUrl(url);
  }

  private async req(path: string, init?: RequestInit): Promise<Response> {
    let res: Response;
    try {
      res = await fetch(`${this.base}${path}`, init);
    } catch (e) {
      if ((e as Error).name === "AbortError") throw e;
      throw new ComfyError("network", String((e as Error).message ?? e));
    }
    if (!res.ok) throw new ComfyError("http", `${res.status} ${await res.text().catch(() => "")}`.trim());
    return res;
  }

  async stats(signal?: AbortSignal): Promise<ComfyStats> {
    const s = await (await this.req("/system_stats", { signal })).json();
    const dev = Array.isArray(s?.devices) ? s.devices[0] : null;
    return {
      version: s?.system?.comfyui_version ?? null,
      device: dev?.name ?? null,
      vramGb: typeof dev?.vram_total === "number" ? dev.vram_total / 1024 ** 3 : null,
    };
  }

  /** Uploads to ComfyUI's input/ folder; returns the name to put in the node. */
  async upload(file: Blob, name: string, signal?: AbortSignal): Promise<string> {
    const form = new FormData();
    form.append("image", file, name);
    form.append("type", "input");
    form.append("overwrite", "true");
    const r = await (await this.req("/upload/image", { method: "POST", body: form, signal })).json();
    return r.subfolder ? `${r.subfolder}/${r.name}` : r.name;
  }

  private async queue(graph: ApiGraph, clientId: string, signal?: AbortSignal): Promise<string> {
    let res: Response;
    try {
      res = await fetch(`${this.base}/prompt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: graph, client_id: clientId }),
        signal,
      });
    } catch (e) {
      if ((e as Error).name === "AbortError") throw e;
      throw new ComfyError("network", String((e as Error).message ?? e));
    }
    const body = await res.json().catch(() => null);
    if (!res.ok || !body?.prompt_id) {
      // 400 with error/node_errors: the graph failed validation (missing model, unknown node…).
      type NodeError = { class_type?: string; errors?: { message?: string; details?: string }[] };
      const nodeErrors: NodeError[] = body?.node_errors ? Object.values(body.node_errors as Record<string, NodeError>) : [];
      const first = nodeErrors[0];
      const detail = first
        ? `${first.class_type ?? ""}: ${first.errors?.map((e) => [e.message, e.details].filter(Boolean).join(" — ")).join("; ")}`
        : (body?.error?.message ?? `${res.status}`);
      throw new ComfyError("validation", detail);
    }
    return body.prompt_id as string;
  }

  async history(promptId: string, signal?: AbortSignal): Promise<Record<string, unknown> | null> {
    const h = await (await this.req(`/history/${encodeURIComponent(promptId)}`, { signal })).json();
    return (h?.[promptId] as Record<string, unknown>) ?? null;
  }

  async download(f: ComfyFile, signal?: AbortSignal): Promise<Blob> {
    const q = new URLSearchParams({ filename: f.filename, subfolder: f.subfolder, type: f.type });
    return (await this.req(`/view?${q}`, { signal })).blob();
  }

  async cancel(promptId: string) {
    await Promise.allSettled([
      this.req("/queue", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ delete: [promptId] }) }),
      this.req("/interrupt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt_id: promptId }) }),
    ]);
  }

  /** Queues the graph and follows it to the end. The WebSocket opens before
   *  POST /prompt so the first messages aren't lost; if it fails to open or
   *  drops, progress falls back to polling /history (no per-step detail).
   *  Returns the output node's files. */
  async run(
    graph: ApiGraph,
    outputNode: string,
    onProgress: (p: ComfyProgress) => void,
    signal?: AbortSignal,
  ): Promise<ComfyFile[]> {
    const clientId = crypto.randomUUID();
    const ws = await openSocket(wsUrl(this.base, clientId), signal);
    let promptId: string | null = null;
    const pending: { type: string; data?: Record<string, unknown> }[] = [];
    let state: ComfyProgress | null = null;
    let wake: (() => void) | null = null;
    const apply = (m: { type: string; data?: Record<string, unknown> }) => {
      if (!state) return void pending.push(m);
      state = reduceProgress(state, m);
      onProgress(state);
      wake?.();
    };
    if (ws) {
      ws.onmessage = (ev) => {
        if (typeof ev.data !== "string") return; // binary previews
        try {
          apply(JSON.parse(ev.data));
        } catch {
          /* non-JSON message */
        }
      };
      ws.onclose = () => wake?.();
    }
    const onAbort = () => {
      if (promptId) void this.cancel(promptId);
      wake?.();
    };
    signal?.addEventListener("abort", onAbort);
    try {
      promptId = await this.queue(graph, clientId, signal);
      state = initialProgress(promptId, Object.keys(graph).length);
      onProgress(state);
      for (const m of pending.splice(0)) apply(m);
      // Wait for completion over the WebSocket; without it (or if it drops), poll the history.
      for (;;) {
        if (signal?.aborted) throw new DOMException("cancelled", "AbortError");
        const s = state as ComfyProgress;
        if (s.status === "error") throw new ComfyError("execution", [s.error?.nodeType, s.error?.message].filter(Boolean).join(": "));
        if (s.status === "interrupted") throw new ComfyError("interrupted");
        const h = s.status === "success" || !ws || ws.readyState !== WebSocket.OPEN ? await this.history(promptId, signal) : null;
        const st = h?.status as { status_str?: string; completed?: boolean; messages?: [string, Record<string, unknown>][] } | undefined;
        if (h && st?.status_str === "error") {
          const err = st.messages?.find(([t]) => t === "execution_error")?.[1];
          throw new ComfyError("execution", String(err?.exception_message ?? "ComfyUI error"));
        }
        if (h && (st?.completed || st?.status_str === "success")) {
          if (s.status !== "success") apply({ type: "execution_success", data: { prompt_id: promptId } });
          const files = outputFiles((h.outputs as Record<string, Record<string, unknown>> | undefined)?.[outputNode]);
          if (!files.length) throw new ComfyError("noOutput", outputNode);
          return files;
        }
        await new Promise<void>((resolve) => {
          wake = resolve;
          setTimeout(resolve, ws && ws.readyState === WebSocket.OPEN ? 15000 : POLL_MS);
        });
        wake = null;
      }
    } finally {
      signal?.removeEventListener("abort", onAbort);
      ws?.close();
    }
  }
}

/** Opens the WebSocket, waiting a few seconds at most; null if it fails. */
function openSocket(url: string, signal?: AbortSignal): Promise<WebSocket | null> {
  return new Promise((resolve) => {
    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      return resolve(null);
    }
    const done = (v: WebSocket | null) => {
      clearTimeout(timer);
      ws.onopen = ws.onerror = null;
      if (!v) ws.close();
      resolve(v);
    };
    const timer = setTimeout(() => done(null), 4000);
    ws.onopen = () => done(ws);
    ws.onerror = () => done(null);
    signal?.addEventListener("abort", () => done(null), { once: true });
  });
}
