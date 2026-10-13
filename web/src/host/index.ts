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
import { importFolder, importUrl, layout, openProject, projectFor, readState, saveState, setFramesRejected } from "./project";

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

  async exportSplat(dir, options, onProgress) {
    const t0 = performance.now();
    const l = layout(dir);
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

/** Top-level files of a folder dropped onto the window. */
async function readFolder(dir: FileSystemDirectoryEntry): Promise<File[]> {
  const reader = dir.createReader();
  const entries: FileSystemEntry[] = [];
  // readEntries returns batches (~100) until it comes back empty.
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
    if (!batch.length) break;
    entries.push(...batch);
  }
  return Promise.all(
    entries
      .filter((e): e is FileSystemFileEntry => e.isFile && isImage(e.name))
      .map((e) => new Promise<File>((res, rej) => e.file(res, rej))),
  );
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
  Promise.all(dirs.map(async (d) => ({ name: d.name, files: await readFolder(d) }))).then(
    (folders) => dropped(register(loose, folders)),
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
    const files = await pickFiles(["video/*", "image/*", ...extensions.map((e) => `.${e}`)].join(","));
    // One video (or package) at a time: among several files, it takes priority.
    const single = files.find((f) => !isImage(f.name));
    return single ? register([single]) : register(files);
  },
  async pickFramesFolder() {
    const files = await pickFiles("", true);
    // Top level of the folder only: "folder/file".
    if (!files.length) return [];
    const top = files.filter((f) => f.webkitRelativePath.split("/").length === 2 && isImage(f.name));
    if (!top.length) throw new Error(`no images (${IMAGE_EXT.join(", ")}) at the top level of the folder`);
    return register([], [{ name: top[0].webkitRelativePath.split("/")[0], files: top }]);
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
