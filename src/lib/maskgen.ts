// Mask generation in ComfyUI (method "external"), per project. Kept outside
// the panel: a run takes minutes and must survive switching batch items.

import { useSyncExternalStore } from "react";
import { api, type MaskSource, type MaskSourceOf } from "./api";
import { ComfyClient } from "./comfy/client";
import { progressFraction } from "./comfy/progress";
import { loadComfy, type ComfyWorkflow } from "./comfy/settings";
import { withInput, withParams, type ParamValue } from "./comfy/workflow";

export interface MaskGen {
  /** Null once finished: then `last` says how it went. */
  fraction: number | null;
  text: string;
  abort: AbortController;
  /** Outcome of the latest run (not of a cancelled one). */
  last: { error: unknown } | { saved: MaskSource; sent: number } | null;
}

/** Text shown while generating; the panel passes its catalog. */
export interface MaskGenTexts {
  preparing: (pct: number) => string;
  uploading: (n: number) => string;
  queued: (n: number | null) => string;
  running: (node: string, step: string) => string;
  downloading: string;
  saving: string;
  noVideo: string;
}

const running = new Map<string, MaskGen>();
let snapshot: ReadonlyMap<string, MaskGen> = new Map();
const listeners = new Set<() => void>();

function set(dir: string, g: MaskGen | null) {
  if (g) running.set(dir, g);
  else running.delete(dir);
  snapshot = new Map(running);
  for (const l of listeners) l();
}

const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => void listeners.delete(l);
};

/** Generations, in progress or finished, by project folder. */
export const useMaskGens = () => useSyncExternalStore(subscribe, () => snapshot);

/** Source → input video → workflow → mask video saved in the project.
 *  Resolves with the saved source and how many frames were sent; rejects on
 *  failure (AbortError when cancelled). */
export async function generateMask(
  projectDir: string,
  source: MaskSourceOf,
  workflow: ComfyWorkflow,
  values: Record<string, ParamValue>,
  tx: MaskGenTexts,
): Promise<{ saved: MaskSource; sent: number }> {
  const ms = api.maskSource;
  if (!ms) throw new Error("mask sources are not available here");
  if (running.get(projectDir)?.fraction != null) throw new Error("a mask is already being generated for this project");
  const abort = new AbortController();
  const mine = () => running.get(projectDir)?.abort === abort;
  const progress = (fraction: number, text: string) => {
    if (mine()) set(projectDir, { fraction, text, abort, last: null });
  };
  set(projectDir, { fraction: 0, text: tx.preparing(0), abort, last: null });
  try {
    const { video, ...input } = await ms.workflowInput(source, (f) => progress(0.1 * f, tx.preparing(Math.round(f * 100))), abort.signal);
    const sent = input.positions.length;
    const client = new ComfyClient(loadComfy().url);
    progress(0.1, tx.uploading(sent));
    const name = await client.upload(video, `gorbit_mask_${Date.now()}.mp4`, abort.signal);
    const titles = Object.fromEntries(Object.entries(workflow.graph).map(([id, n]) => [id, n._meta?.title || n.class_type]));
    const files = await client.run(
      withParams(withInput(workflow.graph, workflow.io, name), workflow.params ?? [], values),
      workflow.io.output.node,
      (p) =>
        progress(
          0.15 + 0.8 * progressFraction(p),
          p.status === "queued" ? tx.queued(p.queue) : tx.running(p.node ? (titles[p.node] ?? p.node) : "", p.step ? `${p.step.value}/${p.step.max}` : ""),
        ),
      abort.signal,
    );
    const out = files.find((f) => /\.(mp4|webm|mov|mkv|m4v)$/i.test(f.filename));
    if (!out) throw new Error(tx.noVideo);
    progress(0.96, tx.downloading);
    const blob = await client.download(out, abort.signal);
    progress(0.99, tx.saving);
    const saved = await ms.saveGenerated(projectDir, blob, input, workflow.name);
    if (mine()) set(projectDir, { fraction: null, text: "", abort, last: { saved, sent } });
    return { saved, sent };
  } catch (e) {
    if (mine()) set(projectDir, abort.signal.aborted ? null : { fraction: null, text: "", abort, last: { error: e } });
    throw e;
  }
}

/** Forgets the outcome shown for a project (a new attachment replaced it). */
export function clearMaskGen(projectDir: string) {
  if (running.get(projectDir)?.fraction === null) set(projectDir, null);
}

export function cancelMaskGen(projectDir: string) {
  running.get(projectDir)?.abort.abort();
}
