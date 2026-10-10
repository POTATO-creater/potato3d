// Projetos da versão web: mesmo layout e mesmo project.json do desktop
// (src-tauri/src/pipeline.rs: Layout e ProjectState), dentro do OPFS.

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
  model: `${root}/dataset/sparse/0`,
  masks: `${root}/masks`,
  alpha: `${root}/dataset_alpha`,
  training: `${root}/training`,
  finalPly: `${root}/splat.ply`,
  exports: `${root}/export`,
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
});

/** project.json mais o instante da última gravação (para a limpeza). */
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

/** Estado salvo, com os arquivos que a interface mostra já publicados. */
export async function openProject(root: string): Promise<ProjectState> {
  const l = layout(root);
  const s = await readState(root);
  await publishDir(l.images, IMAGE_EXT);
  if (s.masks) await publishDir(l.masks, /\.png$/i);
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

/** Mantém só os projetos mais recentes: cada um ocupa centenas de MB na
 *  cota do navegador. */
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

/** Frames de um dataset que não veio de um vídeo: nitidez medida como na
 *  extração, instante desconhecido. */
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

/** Estado de um projeto montado a partir de um dataset COLMAP já copiado. */
async function datasetState(root: string, mapper: string, saved?: ProjectState): Promise<ProjectState> {
  const l = layout(root);
  const names = new Set(await listFiles(l.images, IMAGE_EXT));
  // Do project.json do desktop vêm o instante e a nitidez de cada frame.
  const known = saved?.frames.filter((f) => names.has(basename(f.path)));
  const frames =
    known && known.length === names.size
      ? known.map((f) => ({ ...f, path: `${l.images}/${basename(f.path)}` }))
      : await framesOf(l.images);
  const bin = async (n: string) => (await exists(`${l.model}/${n}`) ? (await readFile(`${l.model}/${n}`)).arrayBuffer() : null);
  const images = await bin("images.bin");
  const masks = saved?.masks && (await listFiles(l.masks, /\.png$/i)).length === names.size ? saved.masks : null;
  return {
    ...emptyState(),
    videoPath: saved?.videoPath ?? null,
    frames,
    masks,
    masksDir: masks ? l.masks : null,
    poses: images ? (saved?.poses ?? modelStats(images, await bin("points3D.bin"), frames.length, mapper)) : null,
  };
}

/** Copia uma pasta escolhida no disco para o projeto: um projeto do
 *  Gausify Orbit (dataset/ + masks/ + project.json) ou um dataset COLMAP
 *  (images/ + sparse/0 ou sparse/). */
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
      /* pasta de projeto sem project.json */
    }
  }
  const images = await sub(dataset, ["images"]).catch(() => {
    throw new Error(`"${src.name}" não tem images/ nem é uma pasta de projeto do Gausify Orbit.`);
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
  const state = await datasetState(root, "importado", saved);
  await saveState(root, state);
  return state;
}

/** Dataset COLMAP servido por HTTP (`<base>/images/…`, `<base>/sparse/0/*.bin`),
 *  para os testes automatizados. Sem listagem de diretório: os nomes das
 *  imagens vêm do images.bin. */
export async function importUrl(base: string, root: string): Promise<ProjectState> {
  base = base.replace(/\/+$/, "");
  const get = async (rel: string) => {
    const r = await fetch(`${base}/${rel}`);
    if (!r.ok) throw new Error(`${r.status} ao baixar ${base}/${rel}`);
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
  const state = await datasetState(root, `importado de ${basename(base)}`);
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
