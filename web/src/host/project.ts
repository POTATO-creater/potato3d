// Web projects: same layout and project.json as the desktop
// (src-tauri/src/pipeline.rs: Layout and ProjectState), inside OPFS.

import type { FrameInfo, ProjectState } from "../../../src/lib/api";
import { sharpness } from "../pipeline/frames";
import { modelStats } from "../scene";
import {
  basename,
  dirAt,
  exists,
  listFiles,
  PROJECTS,
  publish,
  publishDir,
  readFile,
  remove,
  stem,
  writeFile,
} from "./files";

export const IMAGE_EXT = /\.(jpe?g|png|webp)$/i;
const MODEL_FILES = ["cameras.bin", "images.bin", "points3D.bin"];

export const layout = (root: string) => ({
  root,
  state: `${root}/project.json`,
  dataset: `${root}/dataset`,
  images: `${root}/dataset/images`,
  /** Frames rejected by hand, kept outside the dataset. */
  rejected: `${root}/rejected`,
  model: `${root}/dataset/sparse/0`,
  masks: `${root}/masks`,
  alpha: `${root}/dataset_alpha`,
  training: `${root}/training`,
  finalPly: `${root}/splat.ply`,
  exports: `${root}/export`,
  /** Retouch clips, one per folder (retouch/<id>/clip.json…). */
  retouch: `${root}/retouch`,
  /** Incremental-training dataset, outside dataset/ (Brush reads everything in there). */
  retouchDataset: `${root}/dataset_retouch`,
  /** Pre-retouch splat: base for fine-tuning and for undo. */
  retouchBase: `${root}/splat_before_retouch.ply`,
});

export const emptyState = (): ProjectState => ({
  videoPath: null,
  frames: [],
  masks: null,
  masksDir: null,
  poses: null,
  finalPly: null,
  lastPsnr: null,
  lastExport: null,
  retouch: null,
});

/** project.json plus the last-save timestamp (for pruning). */
type Stored = ProjectState & { updatedAt?: number };

export async function readState(root: string): Promise<Stored> {
  try {
    return { ...emptyState(), ...JSON.parse(await (await readFile(layout(root).state)).text()) };
  } catch {
    return emptyState();
  }
}

export async function saveState(root: string, state: ProjectState) {
  await writeFile(layout(root).state, JSON.stringify({ ...state, updatedAt: Date.now() }, null, 2));
}

/** Saved state, with the files the UI shows already published. */
export async function openProject(root: string): Promise<ProjectState> {
  const l = layout(root);
  const s = await readState(root);
  await publishDir(l.images, IMAGE_EXT);
  await publishDir(l.rejected, IMAGE_EXT);
  if (s.masks) await publishDir(l.masks, /\.png$/i);
  if (s.retouch) {
    try {
      publish(s.retouch.base, await readFile(s.retouch.base));
    } catch {
      s.retouch = null;
    }
  }
  for (const p of [s.finalPly, s.lastExport?.path])
    if (p) {
      try {
        publish(p, await readFile(p));
      } catch {
        if (p === s.finalPly) s.finalPly = null;
        else s.lastExport = null;
      }
    }
  return s;
}

/** Rejects (moves to `rejected/`) or restores frames, like
 *  `set_frames_rejected` in src-tauri/src/pipeline.rs. */
export async function setFramesRejected(root: string, names: string[], rejected: boolean): Promise<ProjectState> {
  const l = layout(root);
  const state = await readState(root);
  const want = new Set(names);
  let changed = false;
  let missingMask = false;
  const frames = [...state.frames];
  const save = async () => {
    if (!changed) return state;
    const next: ProjectState = { ...state, frames, poses: null, ...(missingMask ? { masks: null, masksDir: null } : {}) };
    await saveState(root, next);
    return next;
  };
  try {
    for (const [i, f] of frames.entries()) {
      const name = basename(f.path);
      if (!!f.rejected === rejected || !want.has(name)) continue;
      const path = `${rejected ? l.rejected : l.images}/${name}`;
      publish(path, await writeFile(path, await readFile(f.path)));
      await remove(f.path);
      if (!rejected && !(await exists(`${l.masks}/${stem(name)}.png`))) missingMask = true;
      const { rejected: _, ...rest } = f;
      frames[i] = rejected ? { ...rest, path, rejected: true } : { ...rest, path };
      changed = true;
    }
  } catch (e) {
    // Save whatever already moved before rethrowing.
    await save();
    throw e;
  }
  return save();
}

/** Keeps only the most recent projects: each takes hundreds of MB of the
 *  browser's quota. */
export async function pruneProjects(keep: string, max = 3) {
  const dir = await dirAt(PROJECTS, true);
  const all: { path: string; at: number }[] = [];
  for await (const [name, h] of dir.entries()) {
    const path = `${PROJECTS}/${name}`;
    if (h.kind === "directory" && path !== keep) all.push({ path, at: (await readState(path)).updatedAt ?? 0 });
  }
  all.sort((a, b) => b.at - a.at);
  for (const p of all.slice(max - 1)) await remove(p.path);
}

/** Frames of a dataset that didn't come from a video: sharpness measured as in
 *  extraction, timestamp unknown. */
async function framesOf(images: string): Promise<FrameInfo[]> {
  const frames: FrameInfo[] = [];
  for (const name of await listFiles(images, IMAGE_EXT)) {
    const bmp = await createImageBitmap(await readFile(`${images}/${name}`));
    const s = Math.min(1, 640 / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * s)), h = Math.max(1, Math.round(bmp.height * s));
    const ctx = new OffscreenCanvas(w, h).getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(bmp, 0, 0, w, h);
    bmp.close();
    frames.push({ path: `${images}/${name}`, time: 0, sharpness: sharpness(ctx.getImageData(0, 0, w, h).data, w, h) });
  }
  return frames;
}

/** Project state built from an already-copied COLMAP dataset. */
async function datasetState(root: string, mapper: string, saved?: ProjectState): Promise<ProjectState> {
  const l = layout(root);
  const names = new Set(await listFiles(l.images, IMAGE_EXT));
  // The desktop's project.json provides each frame's timestamp and sharpness.
  const known = saved?.frames.filter((f) => names.has(basename(f.path)));
  const frames =
    known && known.length === names.size
      ? known.map((f) => ({ ...f, path: `${l.images}/${basename(f.path)}` }))
      : await framesOf(l.images);
  const bin = async (n: string) => (await exists(`${l.model}/${n}`) ? (await readFile(`${l.model}/${n}`)).arrayBuffer() : null);
  const images = await bin("images.bin");
  // There may be extra masks from rejected frames (which aren't imported).
  const cut = new Set(await listFiles(l.masks, /\.png$/i));
  const masks = saved?.masks && [...names].every((n) => cut.has(`${stem(n)}.png`)) ? saved.masks : null;
  return {
    ...emptyState(),
    videoPath: saved?.videoPath ?? null,
    frames,
    masks,
    masksDir: masks ? l.masks : null,
    poses: images ? (saved?.poses ?? modelStats(images, await bin("points3D.bin"), frames.length, mapper)) : null,
  };
}

/** Copies a folder picked from disk into the project: a Gausify Orbit project
 *  (dataset/ + masks/ + project.json) or a COLMAP dataset
 *  (images/ + sparse/0 or sparse/). */
export async function importFolder(src: FileSystemDirectoryHandle, root: string): Promise<ProjectState> {
  const sub = async (d: FileSystemDirectoryHandle, path: string[]) => {
    for (const p of path) d = await d.getDirectoryHandle(p);
    return d;
  };
  const tryDir = (path: string[]) => sub(src, path).catch(() => null);
  let saved: ProjectState | undefined;
  let dataset = src;
  const desktop = await tryDir(["dataset"]);
  if (desktop) {
    dataset = desktop;
    try {
      saved = JSON.parse(await (await (await src.getFileHandle("project.json")).getFile()).text());
    } catch {
      /* project folder without project.json */
    }
  }
  const images = await sub(dataset, ["images"]).catch(() => {
    throw new Error(`"${src.name}" has no images/ and is not a Gausify Orbit project folder.`);
  });
  const model = (await sub(dataset, ["sparse", "0"]).catch(() => null)) ?? (await sub(dataset, ["sparse"]).catch(() => null));

  const l = layout(root);
  await remove(root);
  const copy = async (from: FileSystemDirectoryHandle, to: string, pattern: RegExp) => {
    for await (const [name, h] of from.entries())
      if (h.kind === "file" && pattern.test(name)) await writeFile(`${to}/${name}`, await (h as FileSystemFileHandle).getFile());
  };
  await copy(images, l.images, IMAGE_EXT);
  if (model) await copy(model, l.model, new RegExp(`^(${MODEL_FILES.join("|").replace(/\./g, "\\.")})$`));
  const masks = desktop && (await tryDir(["masks"]));
  if (masks && saved?.masks) await copy(masks, l.masks, /\.png$/i);
  const state = await datasetState(root, "imported", saved);
  await saveState(root, state);
  return state;
}

/** COLMAP dataset served over HTTP (`<base>/images/…`, `<base>/sparse/0/*.bin`),
 *  for automated tests. No directory listing: image names come from
 *  images.bin. */
export async function importUrl(base: string, root: string): Promise<ProjectState> {
  base = base.replace(/\/+$/, "");
  const get = async (rel: string) => {
    const r = await fetch(`${base}/${rel}`);
    if (!r.ok) throw new Error(`${r.status} downloading ${base}/${rel}`);
    return r.blob();
  };
  const l = layout(root);
  await remove(root);
  const model = await Promise.all(MODEL_FILES.map((f) => get(`sparse/0/${f}`)));
  for (let i = 0; i < MODEL_FILES.length; i++) await writeFile(`${l.model}/${MODEL_FILES[i]}`, model[i]);
  const queue = imageNames(await model[1].arrayBuffer());
  const worker = async () => {
    for (let n = queue.shift(); n !== undefined; n = queue.shift())
      await writeFile(`${l.images}/${basename(n)}`, await get(`images/${n.split("/").map(encodeURIComponent).join("/")}`));
  };
  await Promise.all(Array.from({ length: 6 }, worker));
  const state = await datasetState(root, `imported from ${basename(base)}`);
  await saveState(root, state);
  return state;
}

function imageNames(buf: ArrayBuffer): string[] {
  const v = new DataView(buf);
  const bytes = new Uint8Array(buf);
  let o = 8;
  const names: string[] = [];
  for (let i = Number(v.getBigUint64(0, true)); i > 0; i--) {
    o += 4 + 7 * 8 + 4; // id, qvec, tvec, camera_id
    const end = bytes.indexOf(0, o);
    names.push(new TextDecoder().decode(bytes.subarray(o, end)));
    o = end + 1;
    o += 8 + Number(v.getBigUint64(o, true)) * 24;
  }
  return names;
}

export const projectFor = (videoPath: string) => `${PROJECTS}/${stem(videoPath) || "video"}`;
