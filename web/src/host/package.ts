// Project package (.gorbit) on the web: the same format as
// src-tauri/src/package.rs (a ZIP with the project folder layout, relative
// paths in project.json and a manifest.json), read and written with ./zip.ts.
// The source is restored only into the tab's memory, like a video dropped on the window.

import { PACKAGE_EXT, type PackRequest, type PackStats, type ProjectState, type UnpackedProject } from "../../../src/lib/api";
import {
  addImages,
  addVideo,
  basename,
  download,
  exists,
  groupName,
  inputImage,
  listFiles,
  packageFile,
  PROJECTS,
  readFile,
  remove,
  stem,
  video,
  writeFile,
} from "./files";
import { layout, readState, saveState } from "./project";
import { entryBlob, readZip, ZipWriter } from "./zip";

const FORMAT = "gausify-orbit-project";
const VERSION = 1;
const MODEL_FILES = ["cameras.bin", "images.bin", "points3D.bin"];
/** What a package may contain; anything else is ignored (future versions). */
const ALLOWED = ["dataset/images/", "dataset/sparse/0/", "rejected/", "masks/", "export/", "splat.ply"];

type PackedSource = { kind: "video"; file: string } | { kind: "images"; files: string[] };

interface Manifest {
  format: string;
  version: number;
  app: string;
  createdAt: number;
  source?: PackedSource | null;
  session?: unknown;
}

/** Destinations chosen in `pickPackageDest`: a file on disk or, without the
 *  File System Access API, a download at the end. */
const dests = new Map<string, FileSystemFileHandle | null>();

export async function pickPackageDest(projectDir: string): Promise<string | null> {
  const name = `${basename(projectDir)}.${PACKAGE_EXT}`;
  if ("showSaveFilePicker" in window) {
    try {
      const h = await window.showSaveFilePicker({
        id: "splatforge-package",
        suggestedName: name,
        types: [{ description: "Gausify Orbit", accept: { "application/zip": [`.${PACKAGE_EXT}`] } }],
      });
      const path = `/saves/${h.name}`;
      dests.set(path, h);
      return path;
    } catch (e) {
      if ((e as DOMException).name === "AbortError") return null;
      // Blocked (e.g. in an iframe): fall back to a download.
    }
  }
  const path = `/downloads/${name}`;
  dests.set(path, null);
  return path;
}

const compress = (name: string) => /\.(json|bin|ply)$/i.test(name);

const VIDEO_TYPES: Record<string, string> = { mp4: "video/mp4", m4v: "video/mp4", mov: "video/quicktime", webm: "video/webm", mkv: "video/x-matroska" };

export async function packProject(req: PackRequest, onProgress: (f: number) => void): Promise<PackStats> {
  if (!dests.has(req.dest)) throw new Error("unknown package destination");
  const handle = dests.get(req.dest) ?? null;
  dests.delete(req.dest);
  const l = layout(req.projectDir);
  const { updatedAt: _, ...saved } = await readState(l.root);
  const state: ProjectState = { ...saved };
  if (!state.frames.length) throw new Error("the project has no frames to save yet");
  const entries: { name: string; blob: Blob }[] = [];

  state.frames = [];
  for (const f of saved.frames) {
    const rel = `${f.rejected ? "rejected" : "dataset/images"}/${basename(f.path)}`;
    entries.push({ name: rel, blob: await readFile(f.path) });
    state.frames.push({ ...f, path: rel });
  }

  if (req.include.masks && state.masks && (await exists(l.masks))) {
    for (const n of await listFiles(l.masks, /\.png$/i)) entries.push({ name: `masks/${n}`, blob: await readFile(`${l.masks}/${n}`) });
    state.masksDir = "masks";
  } else {
    state.masks = null;
    state.masksDir = null;
  }

  const poses = req.include.poses && !!state.poses && (await exists(`${l.model}/images.bin`));
  if (poses) {
    for (const f of MODEL_FILES)
      if (await exists(`${l.model}/${f}`)) entries.push({ name: `dataset/sparse/0/${f}`, blob: await readFile(`${l.model}/${f}`) });
  } else state.poses = null;

  // The splat depends on the poses it was trained with.
  if (poses && req.include.splat && state.finalPly && (await exists(l.finalPly))) {
    entries.push({ name: "splat.ply", blob: await readFile(l.finalPly) });
    state.finalPly = "splat.ply";
    const last = state.lastExport;
    state.lastExport = null;
    if (last) {
      const rel = `export/${basename(last.path)}`;
      const blob = await readFile(last.path).catch(() => null);
      if (blob) {
        entries.push({ name: rel, blob });
        state.lastExport = { ...last, path: rel };
      }
    }
  } else {
    state.finalPly = null;
    state.lastPsnr = null;
    state.lastExport = null;
  }

  let source: PackedSource | null = null;
  const src = req.source;
  if (src?.kind === "video") {
    // The video only exists in the tab's memory; without it, the package has no source.
    const file = (() => {
      try {
        return video(src.path).file;
      } catch {
        return null;
      }
    })();
    if (file) {
      const rel = `source/${file.name}`;
      entries.push({ name: rel, blob: file });
      source = { kind: "video", file: rel };
      state.videoPath = rel;
    }
  } else if (src?.kind === "images") {
    const files: string[] = [];
    const seen = new Set<string>();
    for (const [i, p] of src.paths.entries()) {
      let file: File;
      try {
        file = inputImage(p);
      } catch {
        continue;
      }
      let name = file.name;
      if (seen.has(name)) name = `${String(i).padStart(5, "0")}_${name}`;
      seen.add(name);
      const rel = `source/images/${name}`;
      entries.push({ name: rel, blob: file });
      files.push(rel);
    }
    if (files.length) {
      source = { kind: "images", files };
      state.videoPath = "source/images";
    }
  }

  const manifest: Manifest = {
    format: FORMAT,
    version: VERSION,
    app: __APP_VERSION__,
    createdAt: Date.now(),
    source,
    session: req.session ?? null,
  };

  const writable = handle ? await handle.createWritable() : null;
  const parts: (Blob | Uint8Array<ArrayBuffer>)[] = [];
  const zip = new ZipWriter(writable ? (p) => writable.write(p) : (p) => void parts.push(p));
  const total = Math.max(1, entries.reduce((s, e) => s + e.blob.size, 0));
  let done = 0;
  let bytes: number;
  try {
    const json = (v: unknown) => new Blob([JSON.stringify(v, null, 2)]);
    await zip.add("manifest.json", json(manifest), true);
    await zip.add("project.json", json(state), true);
    onProgress(0);
    let last = 0;
    for (const e of entries) {
      await zip.add(e.name, e.blob, compress(e.name), (n) => {
        const now = performance.now();
        if (now - last > 66) {
          last = now;
          onProgress(((done + n) / total) * 0.99);
        }
      });
      done += e.blob.size;
    }
    bytes = await zip.finish();
    if (writable) await writable.close();
  } catch (e) {
    await writable?.abort().catch(() => {});
    throw e;
  }
  const name = basename(req.dest);
  if (!writable) download(new Blob(parts, { type: "application/zip" }), name);
  onProgress(1);
  return { path: name, bytes, files: entries.length + 2 };
}

/** A free project folder: `<name>`, `<name> (2)`… */
async function freeRoot(name: string): Promise<string> {
  const base = `${PROJECTS}/${name.replace(/[\\/]/g, "_") || "project"}`;
  if (!(await exists(base))) return base;
  for (let i = 2; ; i++) if (!(await exists(`${base} (${i})`))) return `${base} (${i})`;
}

function safeRel(rel: string): string {
  const parts = rel.split("/");
  if (parts.some((p) => !p || p === "." || p === ".." || /[\\:]/.test(p))) throw new Error(`invalid path in package: ${rel}`);
  return rel;
}

export async function unpackProject(path: string, onProgress: (f: number) => void): Promise<UnpackedProject> {
  const file = packageFile(path);
  const notPackage = () => new Error("the file is not a Gausify Orbit package");
  const entries = await readZip(file).catch(() => {
    throw notPackage();
  });
  const json = async <T,>(name: string): Promise<T> => {
    const e = entries.get(name);
    if (!e) throw notPackage();
    return JSON.parse(await (await entryBlob(file, e)).text());
  };
  const manifest = await json<Manifest>("manifest.json");
  if (manifest.format !== FORMAT) throw notPackage();
  if (manifest.version > VERSION)
    throw new Error(`the package was saved by a newer version of Gausify Orbit (${manifest.app}); reload the page to update`);
  const state: ProjectState = { ...(await json<ProjectState>("project.json")) };

  const root = await freeRoot(stem(file.name));
  const l = layout(root);
  const wanted = [...entries.values()].filter(
    (e) => !e.name.endsWith("/") && ALLOWED.some((a) => (a.endsWith("/") ? e.name.startsWith(a) : e.name === a)),
  );
  const total = Math.max(1, wanted.reduce((s, e) => s + e.size, 0));
  let done = 0;
  onProgress(0);
  try {
    for (const e of wanted) {
      await writeFile(`${root}/${safeRel(e.name)}`, await entryBlob(file, e));
      done += e.size;
      onProgress((done / total) * 0.99);
    }

    // Relative package paths → this project's paths.
    const abs = (rel: string) => `${root}/${safeRel(rel)}`;
    state.frames = state.frames.map((f) => ({ ...f, path: abs(f.path) }));
    if (state.masks && (await exists(l.masks))) state.masksDir = l.masks;
    else (state.masks = null), (state.masksDir = null);
    if (state.poses && !(await exists(`${l.model}/images.bin`))) state.poses = null;
    if (state.poses && state.finalPly && (await exists(l.finalPly))) {
      state.finalPly = l.finalPly;
      const last = state.lastExport;
      state.lastExport = last && (await exists(abs(last.path))) ? { ...last, path: abs(last.path) } : null;
    } else {
      state.finalPly = null;
      state.lastPsnr = null;
      state.lastExport = null;
    }

    // The source stays in the tab's memory, as if it had been dropped on the window.
    const asFile = async (rel: string, type = "") => {
      const e = entries.get(safeRel(rel));
      if (!e) throw new Error(`package is missing ${rel}`);
      return new File([await entryBlob(file, e)], basename(rel), { type });
    };
    let videoPath: string | null = null;
    let images: string[] | null = null;
    const src = manifest.source;
    if (src?.kind === "video") {
      const ext = src.file.split(".").pop()?.toLowerCase() ?? "";
      videoPath = addVideo(await asFile(src.file, VIDEO_TYPES[ext]));
      state.videoPath = videoPath;
    } else if (src?.kind === "images" && src.files.length) {
      const files = await Promise.all(src.files.map((f) => asFile(f)));
      images = addImages(files, groupName(files.map((f) => f.name)));
      state.videoPath = images[0].replace(/\/[^/]*$/, "");
    }
    await saveState(root, state);
    onProgress(1);
    return { projectDir: root, video: videoPath, images, session: manifest.session ?? null };
  } catch (e) {
    await remove(root);
    throw e;
  }
}
