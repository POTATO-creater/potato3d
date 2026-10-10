// Vídeo → dataset de treino, inteiro no navegador, no mesmo layout do app
// desktop (dentro do OPFS):
//   <projeto>/dataset/images/00000.jpg …   frames
//   <projeto>/dataset/sparse/0             poses
//   <projeto>/masks/00000.png …            recorte (255 = sujeito)
//   <projeto>/dataset_alpha/               PNG RGBA (alfa direto) + poses com
//                                          os nomes .png, para o Brush treinar
//                                          com o fundo transparente

import { colmapFiles, writeModel } from "../sfm/colmap";
import { runSfm, type SfmOptions, type SfmRun } from "../sfm";
import { freshDir } from "../opfs";
import { applyDespill, backgroundColor, borderKey, type ChromaOptions, chromaMask, DEFAULT_CHROMA, despillFor, type RGB } from "./chroma";
import { extractFrames, type FrameInfo, type FrameOptions, type VideoInfo } from "./frames";
import { encodePng } from "./png";
import { MODELS, Segmenter } from "./segment";

export type MaskMethod = "rmbg" | "chroma";

export interface MaskSettings {
  enabled: boolean;
  method: MaskMethod;
  chroma: ChromaOptions;
  despill: boolean;
  /** 0–100 */
  despillStrength: number;
  /** Também usa o recorte nas poses (features só no sujeito). */
  forPoses: boolean;
}

export const DEFAULT_MASKS: MaskSettings = {
  enabled: false,
  method: "rmbg",
  chroma: DEFAULT_CHROMA,
  despill: true,
  despillStrength: 100,
  forPoses: false,
};

export interface MaskStats {
  count: number;
  meanCoverage: number;
  backend: string;
  keyColor: RGB | null;
}

export interface PipelineResult {
  project: FileSystemDirectoryHandle;
  /** Pasta para o treino: dataset_alpha com recorte, dataset sem. */
  train: FileSystemDirectoryHandle;
  frames: FrameInfo[];
  masks: MaskStats | null;
  sfm: SfmRun;
  timings: Record<string, number>;
}

type Progress = (frac: number, msg: string) => void;

const stem = (name: string) => name.replace(/\.[^.]+$/, "");

async function write(dir: FileSystemDirectoryHandle, name: string, data: Blob) {
  const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
  await w.write(data);
  await w.close();
}

async function rgbaOf(file: Blob): Promise<{ rgba: Uint8ClampedArray; w: number; h: number; bmp: ImageBitmap }> {
  const bmp = await createImageBitmap(file);
  const c = new OffscreenCanvas(bmp.width, bmp.height).getContext("2d", { willReadFrequently: true })!;
  c.drawImage(bmp, 0, 0);
  return { rgba: c.getImageData(0, 0, bmp.width, bmp.height).data, w: bmp.width, h: bmp.height, bmp };
}

async function readMask(file: Blob, w: number, h: number): Promise<Uint8Array> {
  const { rgba, bmp, w: mw, h: mh } = await rgbaOf(file);
  bmp.close();
  if (mw !== w || mh !== h) throw new Error("máscara com tamanho diferente do frame");
  const m = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) m[i] = rgba[4 * i];
  return m;
}

async function listImages(dir: FileSystemDirectoryHandle) {
  const names: string[] = [];
  for await (const [n, h] of dir.entries()) if (h.kind === "file" && /\.(jpe?g|png)$/i.test(n)) names.push(n);
  return names.sort();
}

async function makeMasks(
  images: FileSystemDirectoryHandle,
  out: FileSystemDirectoryHandle,
  s: MaskSettings,
  log: (s: string) => void,
  progress: Progress,
): Promise<MaskStats> {
  const names = await listImages(images);
  let seg: Segmenter | null = null;
  if (s.method !== "chroma") {
    const spec = MODELS[s.method];
    seg = await Segmenter.create(s.method, (d, n) =>
      progress(0, `Baixando ${spec.label}: ${(d / 1e6).toFixed(0)}/${(n / 1e6).toFixed(0)} MB`),
    );
    log(`Recorte: ${seg.backend} (${spec.license})`);
  }
  const keys: RGB[] = [];
  let coverage = 0;
  try {
    for (let i = 0; i < names.length; i++) {
      const file = await (await images.getFileHandle(names[i])).getFile();
      const { rgba, w, h, bmp } = await rgbaOf(file);
      let mask: Uint8Array;
      if (seg) {
        mask = await seg.mask(bmp, w, h);
      } else {
        const key = s.chroma.autoColor ? borderKey(rgba, w, h) : s.chroma.color;
        keys.push(key);
        mask = chromaMask(rgba, w, h, key, s.chroma);
      }
      bmp.close();
      let sum = 0;
      for (const v of mask) sum += v;
      coverage += sum / (255 * mask.length);
      await write(out, `${stem(names[i])}.png`, await encodePng(mask, w, h, 1));
      progress((i + 1) / names.length, `Recorte ${i + 1}/${names.length}`);
    }
  } finally {
    seg?.release();
  }
  const keyColor = keys.length
    ? ([0, 1, 2].map((c) => keys.map((k) => k[c]).sort((a, b) => a - b)[keys.length >> 1]) as RGB)
    : null;
  if (keyColor && s.chroma.autoColor) log(`Cor do fundo detectada: rgb(${keyColor.join(", ")})`);
  return {
    count: names.length,
    meanCoverage: coverage / Math.max(names.length, 1),
    backend: seg?.backend ?? "Cor sólida",
    keyColor,
  };
}

/** PNG RGBA com A = máscara (alfa direto) e despill no sujeito. */
async function buildAlpha(
  images: FileSystemDirectoryHandle,
  masks: FileSystemDirectoryHandle,
  out: FileSystemDirectoryHandle,
  s: MaskSettings,
  log: (s: string) => void,
  progress: Progress,
) {
  const names = await listImages(images);
  const load = async (n: string) => {
    const { rgba, w, h, bmp } = await rgbaOf(await (await images.getFileHandle(n)).getFile());
    bmp.close();
    const mask = await readMask(await (await masks.getFileHandle(`${stem(n)}.png`)).getFile(), w, h);
    return { rgba, mask, w, h };
  };
  let despill = null;
  if (s.despill) {
    const step = Math.max(Math.ceil(names.length / 24), 1);
    const samples = [];
    for (let i = 0; i < names.length; i += step) samples.push(await load(names[i]));
    const key = backgroundColor(samples);
    despill = key ? despillFor(key, s.despillStrength) : null;
    log(
      despill
        ? `Despill: fundo rgb(${key!.join(", ")}), canal ${"RGB"[despill.channel]}, ${s.despillStrength}%`
        : "Despill: o fundo não é verde nem azul; cores mantidas",
    );
  }
  for (let i = 0; i < names.length; i++) {
    const { rgba, mask, w, h } = await load(names[i]);
    if (despill) applyDespill(rgba, mask, despill);
    for (let p = 0; p < mask.length; p++) rgba[4 * p + 3] = mask[p];
    await write(out, `${stem(names[i])}.png`, await encodePng(rgba, w, h, 4));
    progress((i + 1) / names.length, `Imagens com transparência ${i + 1}/${names.length}`);
  }
}

export async function runPipeline(
  video: File,
  info: VideoInfo,
  frameOpts: FrameOptions,
  maskOpts: MaskSettings,
  sfmOpts: SfmOptions,
  log: (s: string) => void,
  progress: Progress,
): Promise<PipelineResult> {
  const timings: Record<string, number> = {};
  const stage = (lo: number, hi: number): Progress => (f, m) => progress(lo + (hi - lo) * f, m);
  const project = await freshDir(stem(video.name) || "video", undefined, "projects");
  const dataset = await project.getDirectoryHandle("dataset", { create: true });
  const images = await dataset.getDirectoryHandle("images", { create: true });
  const withMasks = maskOpts.enabled;

  let t = performance.now();
  const frames = await extractFrames(video, info, frameOpts, images, stage(0, 0.35));
  timings.frames = performance.now() - t;
  log(`Frames: ${frames.length} em ${(timings.frames / 1000).toFixed(1)} s`);

  let masks: MaskStats | null = null;
  let masksDir: FileSystemDirectoryHandle | undefined;
  if (withMasks) {
    t = performance.now();
    masksDir = await project.getDirectoryHandle("masks", { create: true });
    masks = await makeMasks(images, masksDir, maskOpts, log, stage(0.35, 0.6));
    timings.masks = performance.now() - t;
    log(`Recorte: ${masks.count} máscaras, cobertura média ${(masks.meanCoverage * 100).toFixed(0)}% em ${(timings.masks / 1000).toFixed(1)} s`);
  }

  t = performance.now();
  const sfm = await runSfm(
    dataset,
    { ...sfmOpts, masksDir: withMasks && maskOpts.forPoses ? masksDir : undefined },
    log,
    stage(withMasks ? 0.6 : 0.35, withMasks ? 0.9 : 1),
  );
  await writeModel(dataset, sfm.files);
  timings.poses = performance.now() - t;

  let train = dataset;
  if (withMasks && masksDir) {
    t = performance.now();
    train = await project.getDirectoryHandle("dataset_alpha", { create: true });
    const alphaImages = await train.getDirectoryHandle("images", { create: true });
    await buildAlpha(images, masksDir, alphaImages, maskOpts, log, stage(0.9, 1));
    // o Brush acha as imagens pelo nome gravado no images.bin
    await writeModel(train, colmapFiles(sfm.model, sfm.names.map((n) => `${stem(n)}.png`), sfm.xy));
    timings.alpha = performance.now() - t;
  }
  return { project, train, frames, masks, sfm, timings };
}
