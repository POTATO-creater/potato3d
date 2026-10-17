// Batch: several videos or frame folders open at once, each with its own
// project, trim, masking and job state. Quality settings are shared; the
// backends run one job at a time, so the batch is processed as a queue.

import type { ExportStats, ImageSet, MaskOptions, Msg, SceneFrame, Stage, VideoInfo } from "./api";
import { basename } from "./platform";
import { initialJob, jobReducer, type JobAction, type JobState } from "./job";

/** Progress or result of the last package save/open. */
export type PackageStatus =
  | { kind: "saving" | "opening"; fraction: number }
  | { kind: "saved"; name: string; bytes: number }
  | { kind: "error"; message: string };

export interface BatchItem {
  id: string;
  /** Shown until there is a source or project (a package being opened). */
  label?: string;
  /** Null when the video could not be probed or a package came without its source. */
  videoPath: string | null;
  video: VideoInfo | null;
  /** Already-extracted frames, instead of a video. */
  imageSet: ImageSet | null;
  sourceError: string | null;
  trim: { start: number; end: number | null };
  projectDir: string | null;
  /** Per item: each subject may need its own method or SAM 2.1 marking. */
  masks: MaskOptions;
  fromStage: Stage;
  job: JobState;
  frame: SceneFrame | null;
  lastExport: ExportStats | null;
  /** Thumbnail cache-buster. */
  framesVersion: number;
  pack: PackageStatus | null;
}

export interface Workspace {
  items: BatchItem[];
  active: string | null;
}

export const emptyWorkspace: Workspace = { items: [], active: null };

let nextId = 0;
export function newItem(masks: MaskOptions, patch: Partial<BatchItem> = {}): BatchItem {
  return {
    id: `item-${++nextId}`,
    videoPath: null,
    video: null,
    imageSet: null,
    sourceError: null,
    trim: { start: 0, end: null },
    projectDir: null,
    // A SAM 2.1 marking belongs to one video.
    masks: { ...masks, prompt: null },
    fromStage: "frames",
    job: initialJob,
    frame: null,
    lastExport: null,
    framesVersion: 0,
    pack: null,
    ...patch,
  };
}

export type WorkspaceAction =
  | { type: "add"; item: BatchItem; select: boolean }
  | { type: "remove"; id: string }
  | { type: "select"; id: string }
  | { type: "patch"; id: string; patch: Partial<BatchItem> | ((item: BatchItem) => Partial<BatchItem>) }
  | { type: "job"; id: string; action: JobAction };

export function workspaceReducer(s: Workspace, a: WorkspaceAction): Workspace {
  const update = (id: string, f: (it: BatchItem) => BatchItem): Workspace => ({
    ...s,
    items: s.items.map((it) => (it.id === id ? f(it) : it)),
  });
  switch (a.type) {
    case "add":
      return { items: [...s.items, a.item], active: a.select || !s.active ? a.item.id : s.active };
    case "remove": {
      const i = s.items.findIndex((it) => it.id === a.id);
      if (i < 0) return s;
      const items = s.items.filter((it) => it.id !== a.id);
      // The neighbor takes the removed item's place.
      const active = s.active === a.id ? (items[Math.min(i, items.length - 1)]?.id ?? null) : s.active;
      return { items, active };
    }
    case "select":
      return s.items.some((it) => it.id === a.id) ? { ...s, active: a.id } : s;
    case "patch":
      return update(a.id, (it) => ({ ...it, ...(typeof a.patch === "function" ? a.patch(it) : a.patch) }));
    case "job":
      return update(a.id, (it) => ({ ...it, job: jobReducer(it.job, a.action) }));
  }
}

/** Name shown in the selector: the video, the frames folder or the project. */
export function itemName(it: BatchItem): string {
  const p = it.videoPath ?? it.imageSet?.dir ?? it.projectDir;
  return p ? basename(p.replace(/[\\/]+$/, "")) : (it.label ?? "…");
}

/** "Export all": every trained item exported in turn with the same options. */
export interface BulkExport {
  running: boolean;
  /** Item being exported (0-based) out of `total`. */
  index: number;
  total: number;
  name: string;
  progress: { fraction: number; message: Msg } | null;
  results: { id: string; name: string; stats?: ExportStats; error?: string }[];
  cancelled: boolean;
}

export type ItemStatus =
  | { kind: "running"; stage: Stage; fraction: number }
  | { kind: "queued" | "done" | "failed" | "pending" };

export function itemStatus(it: BatchItem, queued: boolean): ItemStatus {
  const j = it.job;
  if (j.running) {
    const stage = (["frames", "masks", "poses", "train"] as Stage[]).find((st) => j.stages[st].status === "running");
    return { kind: "running", stage: stage ?? "frames", fraction: stage ? j.stages[stage].fraction : 0 };
  }
  if (queued) return { kind: "queued" };
  if (j.error) return { kind: "failed" };
  if (j.finalPly) return { kind: "done" };
  return { kind: "pending" };
}
