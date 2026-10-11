// Recorte (máscaras 255 = sujeito) e dataset com fundo transparente, como
// src-tauri/src/masks.rs. No web: RMBG-1.4, SAM 2.1 (só aqui) e cor sólida.

import type { MaskOptions, MaskStats, Msg, SubjectPrompt } from "../../../src/lib/api";
import { applyDespill, backgroundColor, borderKey, chromaMask, despillFor, type RGB } from "./chroma";
import { encodePng } from "./png";
import { SAM2_LICENSE, Sam2Tracker } from "./sam2";
import { logitsToMask, promptFrame, promptPixels } from "./sam2-math";
import { MODELS, Segmenter } from "./segment";

type Progress = (frac: number, msg: Msg) => void;

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

export async function listImages(dir: FileSystemDirectoryHandle) {
  const names: string[] = [];
  for await (const [n, h] of dir.entries()) if (h.kind === "file" && /\.(jpe?g|png)$/i.test(n)) names.push(n);
  return names.sort();
}

export async function makeMasks(
  images: FileSystemDirectoryHandle,
  out: FileSystemDirectoryHandle,
  s: MaskOptions,
  log: (s: string) => void,
  progress: Progress,
  /** Instante de cada frame no vídeo (por nome), para achar o frame marcado. */
  times?: Map<string, number>,
): Promise<MaskStats> {
  const names = await listImages(images);
  if (!names.length) throw new Error("não há frames para recortar");
  if (s.method === "sam2") return sam2Masks(images, out, names, s.prompt ?? null, log, progress, times);
  let seg: Segmenter | null = null;
  if (s.method !== "chroma") {
    const spec = MODELS.rmbg;
    seg = await Segmenter.create("rmbg", (d, n) =>
      progress(0, { key: "masks.downloading", args: { model: spec.label, done: Math.round(d / 1e6), total: Math.round(n / 1e6) } }),
    );
    log(`${seg.backend} (${spec.license})`);
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
      progress((i + 1) / names.length, { key: "masks.cutting", args: { i: i + 1, n: names.length } });
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
    noSubject: 0,
    meanCoverage: coverage / names.length,
    backend: seg?.backend ?? "Cor sólida",
    keyColor,
  };
}

/** SAM 2.1: segmenta o frame marcado e propaga para a frente e depois para trás. */
async function sam2Masks(
  images: FileSystemDirectoryHandle,
  out: FileSystemDirectoryHandle,
  names: string[],
  prompt: SubjectPrompt | null,
  log: (s: string) => void,
  progress: Progress,
  times?: Map<string, number>,
): Promise<MaskStats> {
  const tracker = await Sam2Tracker.create(names.length, (d, n) =>
    progress(0, { key: "masks.downloading", args: { model: "SAM 2.1", done: Math.round(d / 1e6), total: Math.round(n / 1e6) } }),
  );
  const start = promptFrame(names, prompt, times);
  log(`${tracker.backend} (${SAM2_LICENSE})`);
  log(
    prompt && (prompt.box || prompt.points.length)
      ? `SAM 2.1: sujeito marcado, propagando a partir de ${names[start]}`
      : `SAM 2.1: sem marcação, segue o objeto no centro de ${names[start]}`,
  );
  const order = [...names.keys()].slice(start).concat([...names.keys()].slice(0, start).reverse());
  let coverage = 0, noSubject = 0;
  const t0 = performance.now();
  try {
    for (const [k, i] of order.entries()) {
      const bmp = await createImageBitmap(await (await images.getFileHandle(names[i])).getFile());
      const { width: w, height: h } = bmp;
      const f = await tracker.frame(i, bmp, w, h, i === start ? promptPixels(prompt, w, h) : undefined, i < start);
      bmp.close();
      const mask = logitsToMask(f.logits, w, h);
      if (f.objectScore <= 0) noSubject++;
      let sum = 0;
      for (const v of mask) sum += v;
      coverage += sum / (255 * mask.length);
      await write(out, `${stem(names[i])}.png`, await encodePng(mask, w, h, 1));
      progress((k + 1) / names.length, { key: "masks.cutting", args: { i: k + 1, n: names.length } });
    }
  } finally {
    tracker.release();
  }
  log(`SAM 2.1: ${((performance.now() - t0) / names.length / 1000).toFixed(2)} s/frame`);
  return { count: names.length, noSubject, meanCoverage: coverage / names.length, backend: tracker.backend, keyColor: null };
}

/** PNG RGBA com A = máscara (alfa direto) e despill no sujeito. */
export async function buildAlpha(
  images: FileSystemDirectoryHandle,
  masks: FileSystemDirectoryHandle,
  out: FileSystemDirectoryHandle,
  s: MaskOptions,
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
        ? `despill: reflexo de #${key!.map((v) => v.toString(16).padStart(2, "0")).join("")} removido do sujeito`
        : "despill: o fundo não é verde nem azul; cores mantidas",
    );
  }
  for (let i = 0; i < names.length; i++) {
    const { rgba, mask, w, h } = await load(names[i]);
    if (despill) applyDespill(rgba, mask, despill);
    for (let p = 0; p < mask.length; p++) rgba[4 * p + 3] = mask[p];
    await write(out, `${stem(names[i])}.png`, await encodePng(rgba, w, h, 4));
    progress((i + 1) / names.length, { key: "masks.alphaFrames", args: { i: i + 1, n: names.length } });
  }
}
