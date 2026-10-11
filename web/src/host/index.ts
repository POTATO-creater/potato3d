// Host da versão web: os mesmos comandos do backend em Rust, implementados
// no navegador (WebGPU, WASM, OPFS). A interface (src/) é a mesma do desktop.

import { IMAGE_EXT, type Api, type Host, type InstallEvent, type JobEvent, type Tools } from "../../../src/lib/api";
import { brushRuntime } from "../brush";
import { probeVideo } from "../pipeline/frames";
import { cachedModel, downloadModel, MODELS } from "../pipeline/segment";
import { readFrame, readLens } from "../scene";
import { exportSplat, splatInfo } from "./export";
import {
  addImages,
  addVideo,
  basename,
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
      rmbgModel: (await cachedModel(MODELS.rmbg)) ? MODELS.rmbg.file : null,
      os: "web",
    };
  },

  async probeVideo(path) {
    const v = video(path);
    return (v.info ??= await probeVideo(v.file));
  },

  async probeImages(paths) {
    // As pastas já chegam abertas em arquivos (addImages).
    const files = paths
      .filter((p) => isImage(basename(p)))
      .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }));
    if (!files.length) throw new Error(`nenhuma imagem (${IMAGE_EXT.join(", ")}) encontrada`);
    // Na orientação do EXIF, como a extração.
    const bmp = await createImageBitmap(inputImage(files[0]));
    const set = { paths: files, dir: files[0].replace(/\/[^/]*$/, ""), width: bmp.width, height: bmp.height };
    bmp.close();
    return set;
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
    const need = rmbg && !(await cachedModel(MODELS.rmbg));
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
};

const dropHandlers = new Set<(paths: string[]) => void>();
const dropped = (paths: string[]) => dropHandlers.forEach((h) => h(paths));

const ext = (name: string) => name.split(".").pop()?.toLowerCase() ?? "";
// `._foo.jpg`: metadados do macOS em discos exFAT/FAT, não imagens.
const isImage = (name: string) => !name.startsWith(".") && IMAGE_EXT.includes(ext(name));

/** Arquivos escolhidos ou soltos → caminhos: vídeos à parte, imagens
 *  agrupadas (as de uma pasta sob o nome dela). */
function register(loose: File[], folders: { name: string; files: File[] }[] = []): string[] {
  const images = loose.filter((f) => isImage(f.name));
  const paths = loose.filter((f) => !isImage(f.name)).map(addVideo);
  if (images.length) paths.push(...addImages(images, groupName(images.map((f) => f.name))));
  for (const d of folders) paths.push(...addImages(d.files.filter((f) => isImage(f.name)), d.name));
  return paths;
}

/** Arquivos do primeiro nível de uma pasta solta na janela. */
async function readFolder(dir: FileSystemDirectoryEntry): Promise<File[]> {
  const reader = dir.createReader();
  const entries: FileSystemEntry[] = [];
  // readEntries devolve em lotes (~100) até vir vazio.
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
  // Entradas e arquivos só podem ser lidos durante o evento.
  const taken = items.map((i) => ({ entry: i.webkitGetAsEntry?.() ?? null, file: i.getAsFile() }));
  const loose = taken.filter((t) => !t.entry?.isDirectory && t.file).map((t) => t.file!);
  const dirs = taken.map((t) => t.entry).filter((d): d is FileSystemDirectoryEntry => !!d?.isDirectory);
  Promise.all(dirs.map(async (d) => ({ name: d.name, files: await readFolder(d) }))).then(
    (folders) => dropped(register(loose, folders)),
    (err) => console.error("não foi possível ler a pasta solta:", err),
  );
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
  async pickFiles(extensions) {
    const files = await pickFiles(["video/*", "image/*", ...extensions.map((e) => `.${e}`)].join(","));
    // Um vídeo de cada vez: entre vários arquivos, o vídeo tem prioridade.
    const video = files.find((f) => !isImage(f.name));
    return video ? register([video]) : register(files);
  },
  async pickFramesFolder() {
    const files = await pickFiles("", true);
    // Só o primeiro nível da pasta: "pasta/arquivo".
    if (!files.length) return [];
    const top = files.filter((f) => f.webkitRelativePath.split("/").length === 2 && isImage(f.name));
    if (!top.length) throw new Error(`nenhuma imagem (${IMAGE_EXT.join(", ")}) no primeiro nível da pasta`);
    return register([], [{ name: top[0].webkitRelativePath.split("/")[0], files: top }]);
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
