// Host da versão web: os mesmos comandos do backend em Rust, implementados
// no navegador (WebGPU, WASM, OPFS). A interface (src/) é a mesma do desktop.

import type { Api, Host, InstallEvent, JobEvent, Tools } from "../../../src/lib/api";
import { brushRuntime } from "../brush";
import { probeVideo } from "../pipeline/frames";
import { cachedModel, downloadModel, MODELS } from "../pipeline/segment";
import { readFrame, readLens } from "../scene";
import { exportSplat, splatInfo } from "./export";
import { addVideo, basename, readFile, readPublished, publish, urlOf, video, writeFile } from "./files";
import { applyUpdate, onUpdate } from "./update";
import { runJob } from "./job";
import { importFolder, importUrl, layout, openProject, projectFor, readState, saveState } from "./project";

const params = new URLSearchParams(location.search);
let job: AbortController | null = null;
let install: AbortController | null = null;

// Projetos e o modelo de recorte ficam no OPFS: pede armazenamento
// persistente, que o navegador não apaga sob pouco espaço em disco (pode
// recusar em silêncio; aí vale a regra comum).
void navigator.storage?.persist?.().catch(() => false);

const api: Api = {
  async detectTools(): Promise<Tools> {
    // O Brush (WASM + WebGPU) sobe aqui: falha cedo e já deixa a GPU pronta.
    const gpu = await brushRuntime().then(
      (r) => r.gpu,
      (e) => (console.error("Brush/WebGPU indisponível:", e), null),
    );
    const here = "navegador";
    return {
      ffmpeg: here,
      ffprobe: here,
      colmap: here,
      colmapCuda: false,
      brush: gpu ? [gpu.vendor, gpu.architecture].filter(Boolean).join(" · ") || "WebGPU" : null,
      masker: null,
      birefnetModel: null,
      modnetModel: null,
      rmbgModel: (await cachedModel("rmbg")) ? MODELS.rmbg.file : null,
      os: "web",
    };
  },

  async probeVideo(path) {
    const v = video(path);
    return (v.info ??= await probeVideo(v.file));
  },

  defaultProjectDir: async (videoPath) => projectFor(videoPath),

  async loadProject(dir) {
    // `?dataset=<url>` (testes): o projeto do vídeo começa com esse dataset.
    const dataset = params.get("dataset");
    if (dataset) {
      params.delete("dataset");
      await importUrl(dataset, dir);
    }
    return openProject(dir);
  },

  async sceneFrame(dir) {
    const model = layout(dir).model;
    const images = await (await readFile(`${model}/images.bin`)).arrayBuffer();
    const points = await readFile(`${model}/points3D.bin`).then((f) => f.arrayBuffer(), () => null);
    const frame = readFrame(images, points);
    if (!frame) throw new Error("sem poses");
    const cameras = await readFile(`${model}/cameras.bin`).then((f) => f.arrayBuffer(), () => null);
    const lens = cameras ? readLens(cameras) : undefined;
    return lens ? { ...frame, lens } : frame;
  },

  async startJob(config, onEvent) {
    if (job && !job.signal.aborted) throw new Error("já existe um processamento em andamento");
    const ctl = new AbortController();
    job = ctl;
    // Como no desktop: volta logo e o andamento chega pelos eventos.
    void runJob(config, (e: JobEvent) => onEvent(e), ctl.signal).finally(() => {
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

  async installPlan(rmbg) {
    const need = rmbg && !(await cachedModel("rmbg"));
    const spec = MODELS.rmbg;
    return {
      packages: need ? [{ id: "rmbg", label: `Recorte ${spec.label}`, file: spec.file, size: spec.bytes }] : [],
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
    onEvent({ type: "started", id: "rmbg", label: `Recorte ${MODELS.rmbg.label}`, total });
    try {
      await downloadModel("rmbg", (done) => onEvent({ type: "progress", id: "rmbg", done, total }), ctl.signal);
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
};

const dropHandlers = new Set<(paths: string[]) => void>();
const dropped = (paths: string[]) => dropHandlers.forEach((h) => h(paths));

addEventListener("dragover", (e) => {
  if (e.dataTransfer?.types.includes("Files")) e.preventDefault();
});
addEventListener("drop", (e) => {
  const files = [...(e.dataTransfer?.files ?? [])];
  if (!files.length) return;
  e.preventDefault();
  dropped(files.map(addVideo));
});

// `?video=<url>`: abre um vídeo servido por HTTP, como se tivesse sido solto
// na janela (testes).
let urlVideo: Promise<string> | null = null;
function openUrlVideo() {
  const url = params.get("video");
  if (!url || urlVideo) return;
  urlVideo = fetch(url)
    .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(`${r.status} ao baixar ${url}`))))
    .then((b) => addVideo(new File([b], basename(url) || "video.mp4", { type: b.type })));
  urlVideo.then((p) => dropped([p]), (e) => console.error(e));
}

function pickFile(accept: string): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.onchange = () => resolve(input.files?.[0] ?? null);
    input.oncancel = () => resolve(null);
    input.click();
  });
}

function download(blob: Blob, name: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

export const webHost: Host = {
  api,
  // As URLs mudam a cada publicação, então o cache-buster não é preciso.
  fileUrl: (path) => urlOf(path),
  async pickVideo(extensions) {
    const file = await pickFile(["video/*", ...extensions.map((e) => `.${e}`)].join(","));
    return file ? addVideo(file) : null;
  },
  /** "Importar…": copia uma pasta de projeto do Gausify Orbit ou um dataset
   *  COLMAP para um projeto no navegador. */
  async pickFolder() {
    if (!("showDirectoryPicker" in window)) {
      alert("Importar pastas exige o Chrome ou o Edge.");
      return null;
    }
    let src: FileSystemDirectoryHandle;
    try {
      src = await window.showDirectoryPicker({ id: "splatforge-import", mode: "read" });
    } catch {
      return null; // cancelado
    }
    const root = projectFor(src.name);
    await importFolder(src, root);
    return root;
  },
  async reveal(path) {
    download(await readPublished(path), basename(path));
  },
  readBlob: readPublished,
  // `VITE_SHARE_ORIGIN` aponta o `pnpm web:dev` para um `wrangler dev`.
  shareOrigin: import.meta.env.VITE_SHARE_ORIGIN ?? location.origin,
  onDrop(handler) {
    dropHandlers.add(handler);
    openUrlVideo();
    return () => dropHandlers.delete(handler);
  },
  onUpdate,
  applyUpdate,
};
