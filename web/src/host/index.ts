// Web host: the same commands as the Rust backend, implemented in the browser
// (WebGPU, WASM, OPFS). The UI (src/) is the same as the desktop's.

import { IMAGE_EXT, PACKAGE_EXT, type Api, type Host, type InstallEvent, type JobEvent, type Tools } from "../../../src/lib/api";
import { brushRuntime } from "../brush";
import { probeVideo } from "../pipeline/frames";
import { cachedModel, downloadModel, MODELS } from "../pipeline/segment";
import { readFrame, readLens } from "../scene";
import { exportSplat, splatInfo } from "./export";
import {
  addImages,
  addPackage,
  addVideo,
  basename,
  download,
  groupName,
  inputImage,
  readFile,
  readPublished,
  publish,
  urlOf,
  video,
  writeFile,
} from "./files";
import { applyUpdate, onUpdate } from "./update";
import { runJob } from "./job";
import { maskPreview, previewIdle } from "./preview";
import { packProject, pickPackageDest, unpackProject } from "./package";
import { retouchApi } from "./retouch";
import { importFolder, importUrl, keepProjects, layout, openProject, projectFor, readState, saveState, setFramesRejected } from "./project";

const params = new URLSearchParams(location.search);
let job: AbortController | null = null;
let install: AbortController | null = null;

// Projects and the masking model live in OPFS: request persistent storage,
// which the browser won't evict under low disk space (it may silently refuse;
// then the usual eviction rules apply).
void navigator.storage?.persist?.().catch(() => false);

const api: Api = {
  async detectTools(): Promise<Tools> {
    // Brush (WASM + WebGPU) starts here: fails early and leaves the GPU ready.
    const gpu = await brushRuntime().then(
      (r) => r.gpu,
      (e) => (console.error("Brush/WebGPU unavailable:", e), null),
    );
    const here = "browser";
    return {
      ffmpeg: here,
      ffprobe: here,
      colmap: here,
      colmapCuda: false,
      brush: gpu ? [gpu.vendor, gpu.architecture].filter(Boolean).join(" · ") || "WebGPU" : null,
      masker: null,
      birefnetModel: null,
      modnetModel: null,
      rmbgModel: (await cachedModel(MODELS.rmbg)) ? MODELS.rmbg.file : null,
      os: "web",
    };
  },

  async probeVideo(path) {
    const v = video(path);
    return (v.info ??= await probeVideo(v.file));
  },

  async probeImages(paths) {
    // Folders arrive already expanded into files (addImages).
    const files = paths
      .filter((p) => isImage(basename(p)))
      .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }));
    if (!files.length) throw new Error(`no images (${IMAGE_EXT.join(", ")}) found`);
    // In EXIF orientation, like the extraction.
    const bmp = await createImageBitmap(inputImage(files[0]));
    const set = { paths: files, dir: files[0].replace(/\/[^/]*$/, ""), width: bmp.width, height: bmp.height };
    bmp.close();
    return set;
  },

  defaultProjectDir: async (videoPath) => projectFor(videoPath),

  async loadProject(dir) {
    // `?dataset=<url>` (tests): the video's project starts from that dataset.
    const dataset = params.get("dataset");
    if (dataset) {
      params.delete("dataset");
      await importUrl(dataset, dir);
    }
    return openProject(dir);
  },

  setFramesRejected,

  async sceneFrame(dir) {
    const model = layout(dir).model;
    const images = await (await readFile(`${model}/images.bin`)).arrayBuffer();
    const points = await readFile(`${model}/points3D.bin`).then((f) => f.arrayBuffer(), () => null);
    const frame = readFrame(images, points);
    if (!frame) throw new Error("no poses");
    const cameras = await readFile(`${model}/cameras.bin`).then((f) => f.arrayBuffer(), () => null);
    const lens = cameras ? readLens(cameras) : undefined;
    return lens ? { ...frame, lens } : frame;
  },

  async startJob(config, onEvent) {
    if (job && !job.signal.aborted) throw new Error("a job is already running");
    const ctl = new AbortController();
    job = ctl;
    // The mask preview releases the GPU first: the job opens its own sessions.
    maskPreview.release();
    // As on the desktop: returns right away and progress arrives via events.
    void previewIdle()
      .then(() => runJob(config, (e: JobEvent) => onEvent(e), ctl.signal))
      .finally(() => {
        if (job === ctl) job = null;
      });
  },

  async cancelJob() {
    job?.abort();
  },

  splatInfo: async (path) => splatInfo(await readPublished(path)),

  async exportSplat(dir, options, onProgress, signal) {
    const t0 = performance.now();
    const l = layout(dir);
    if (options.format === "glb") {
      const { exportMesh } = await import("../mesh/job");
      const { name, bytes, stats } = await exportMesh(
        dir,
        { triangles: options.meshTriangles ?? 50_000, texture: options.meshTexture ?? true },
        onProgress,
        signal,
      );
      const path = `${l.exports}/${name}`;
      publish(path, await writeFile(path, bytes as BufferSource));
      const result = {
        path,
        inputCount: stats.splats,
        prunedCount: stats.splats,
        outputCount: stats.triangles,
        inputBytes: (await readFile(l.finalPly)).size,
        outputBytes: bytes.byteLength,
        shDegree: 0,
        seconds: (performance.now() - t0) / 1000,
        mesh: { triangles: stats.triangles, vertices: stats.vertices, ...(stats.texture ? { texture: stats.texture.size } : {}) },
      };
      await saveState(dir, { ...(await readState(dir)), lastExport: result });
      return result;
    }
    const { name, bytes, stats } = await exportSplat(await readFile(l.finalPly), options, onProgress);
    const path = `${l.exports}/${name}`;
    publish(path, await writeFile(path, bytes as BufferSource));
    const result = { ...stats, path, seconds: (performance.now() - t0) / 1000 };
    await saveState(dir, { ...(await readState(dir)), lastExport: result });
    return result;
  },

  packProject,
  unpackProject,

  async installPlan(rmbg) {
    const need = rmbg && !(await cachedModel(MODELS.rmbg));
    const spec = MODELS.rmbg;
    return {
      packages: need ? [{ id: "rmbg", label: `Masking ${spec.label}`, file: spec.file, size: spec.bytes }] : [],
      totalBytes: need ? spec.bytes : 0,
      cuda: false,
      note: null,
    };
  },

  async installTools(rmbg, onEvent: (e: InstallEvent) => void) {
    if (!rmbg) return onEvent({ type: "finished" });
    const ctl = new AbortController();
    install = ctl;
    const total = MODELS.rmbg.bytes;
    onEvent({ type: "started", id: "rmbg", label: `Masking ${MODELS.rmbg.label}`, total });
    try {
      await downloadModel(MODELS.rmbg, (done) => onEvent({ type: "progress", id: "rmbg", done, total }), ctl.signal);
      onEvent({ type: "done", id: "rmbg" });
      onEvent({ type: "finished" });
    } catch (e) {
      onEvent(ctl.signal.aborted ? { type: "cancelled" } : { type: "failed", error: String(e) });
    } finally {
      install = null;
    }
  },

  async cancelInstall() {
    install?.abort();
  },

  retouch: retouchApi,
  keepProjects,
};

const dropHandlers = new Set<(paths: string[]) => void>();
const dropped = (paths: string[]) => dropHandlers.forEach((h) => h(paths));

const ext = (name: string) => name.split(".").pop()?.toLowerCase() ?? "";
// `._foo.jpg`: macOS metadata on exFAT/FAT drives, not images.
const isImage = (name: string) => !name.startsWith(".") && IMAGE_EXT.includes(ext(name));
const isPackage = (name: string) => ext(name) === PACKAGE_EXT;

/** Picked or dropped files → paths: videos and packages individually, images
 *  grouped (those from a folder under the folder's name). */
function register(loose: File[], folders: { name: string; files: File[] }[] = []): string[] {
  const images = loose.filter((f) => isImage(f.name));
  const paths = loose.filter((f) => isPackage(f.name)).map(addPackage);
  paths.push(...loose.filter((f) => !isImage(f.name) && !isPackage(f.name)).map(addVideo));
  if (images.length) paths.push(...addImages(images, groupName(images.map((f) => f.name))));
  for (const d of folders) paths.push(...addImages(d.files.filter((f) => isImage(f.name)), d.name));
  return paths;
}

async function readEntries(dir: FileSystemDirectoryEntry): Promise<FileSystemEntry[]> {
  const reader = dir.createReader();
  const entries: FileSystemEntry[] = [];
  // readEntries returns batches (~100) until it comes back empty.
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
    if (!batch.length) break;
    entries.push(...batch);
  }
  return entries;
}

const imageFiles = (entries: FileSystemEntry[]) =>
  Promise.all(
    entries
      .filter((e): e is FileSystemFileEntry => e.isFile && isImage(e.name))
      .map((e) => new Promise<File>((res, rej) => e.file(res, rej))),
  );

const byName = <T extends { name: string }>(a: T, b: T) => a.name.localeCompare(b.name, undefined, { numeric: true });

/** Images at the top of a folder dropped onto the window; a folder of frame
 *  folders gives one set per subfolder (each becomes a batch item). */
async function readFolder(dir: FileSystemDirectoryEntry): Promise<{ name: string; files: File[] }[]> {
  const entries = await readEntries(dir);
  const top = await imageFiles(entries);
  if (top.length) return [{ name: dir.name, files: top }];
  const out: { name: string; files: File[] }[] = [];
  const subs = entries.filter((e): e is FileSystemDirectoryEntry => e.isDirectory).sort(byName);
  for (const sub of subs) {
    const files = await imageFiles(await readEntries(sub));
    if (files.length) out.push({ name: sub.name, files });
  }
  return out;
}

addEventListener("dragover", (e) => {
  if (e.dataTransfer?.types.includes("Files")) e.preventDefault();
});
addEventListener("drop", (e) => {
  const items = [...(e.dataTransfer?.items ?? [])].filter((i) => i.kind === "file");
  if (!items.length) return;
  e.preventDefault();
  // Entries and files can only be read during the event.
  const taken = items.map((i) => ({ entry: i.webkitGetAsEntry?.() ?? null, file: i.getAsFile() }));
  const loose = taken.filter((t) => !t.entry?.isDirectory && t.file).map((t) => t.file!);
  const dirs = taken.map((t) => t.entry).filter((d): d is FileSystemDirectoryEntry => !!d?.isDirectory);
  Promise.all(dirs.map(readFolder)).then(
    (folders) => dropped(register(loose, folders.flat())),
    (err) => console.error("could not read the dropped folder:", err),
  );
});

// `?video=<url>`: opens a video served over HTTP as if it had been dropped on
// the window (tests).
let urlVideo: Promise<string> | null = null;
function openUrlVideo() {
  const url = params.get("video");
  if (!url || urlVideo) return;
  urlVideo = fetch(url)
    .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(`${r.status} downloading ${url}`))))
    .then((b) => addVideo(new File([b], basename(url) || "video.mp4", { type: b.type })));
  urlVideo.then((p) => dropped([p]), (e) => console.error(e));
}

function pickFiles(accept: string, folder = false): Promise<File[]> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.multiple = true;
    input.webkitdirectory = folder;
    input.onchange = () => resolve([...(input.files ?? [])]);
    input.oncancel = () => resolve([]);
    input.click();
  });
}

export const webHost: Host = {
  api,
  // URLs change on every publish, so no cache-buster is needed.
  fileUrl: (path) => urlOf(path),
  async pickFiles(extensions) {
    // Each video and package becomes a batch item; loose images, one item.
    return register(await pickFiles(["video/*", "image/*", ...extensions.map((e) => `.${e}`)].join(",")));
  },
  async pickFramesFolder() {
    const files = await pickFiles("", true);
    if (!files.length) return [];
    const at = (depth: number) => files.filter((f) => f.webkitRelativePath.split("/").length === depth && isImage(f.name));
    // Top level of the folder only: "folder/file".
    const top = at(2);
    if (top.length) return register([], [{ name: top[0].webkitRelativePath.split("/")[0], files: top }]);
    // A folder of frame folders ("folder/take/file"): one set per subfolder.
    const subs = new Map<string, File[]>();
    for (const f of at(3)) {
      const name = f.webkitRelativePath.split("/")[1];
      subs.set(name, [...(subs.get(name) ?? []), f]);
    }
    if (!subs.size) throw new Error(`no images (${IMAGE_EXT.join(", ")}) at the top level of the folder or its subfolders`);
    return register([], [...subs].map(([name, files]) => ({ name, files })).sort(byName));
  },
  /** "Import…": copies a Gausify Orbit project folder or a COLMAP dataset into
   *  a project in the browser. */
  async pickFolder() {
    if (!("showDirectoryPicker" in window)) {
      alert("Importing folders requires Chrome or Edge.");
      return null;
    }
    let src: FileSystemDirectoryHandle;
    try {
      src = await window.showDirectoryPicker({ id: "splatforge-import", mode: "read" });
    } catch {
      return null; // cancelled
    }
    const root = projectFor(src.name);
    await importFolder(src, root);
    return root;
  },
  pickPackageDest,
  async reveal(path) {
    download(await readPublished(path), basename(path));
  },
  readBlob: readPublished,
  // `VITE_SHARE_ORIGIN` points `pnpm web:dev` at a `wrangler dev`.
  shareOrigin: import.meta.env.VITE_SHARE_ORIGIN ?? location.origin,
  onDrop(handler) {
    dropHandlers.add(handler);
    openUrlVideo();
    return () => dropHandlers.delete(handler);
  },
  onUpdate,
  applyUpdate,
  maskPreview,
};
