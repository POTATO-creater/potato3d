// Retoque na versão web: clipes (trajetória renderizada do splat e o vídeo
// restaurado pelo ComfyUI) guardados no OPFS e o treino incremental.

import type { Msg, ProjectState, RetouchApi, RetouchClip, RetouchEvent } from "../../../src/lib/api";
import { appendCameras, appendImages, type NewImage } from "../../../src/lib/retouch/colmap";
import { pickRestoredFrames } from "../../../src/lib/retouch/frames";
import { videoFrames } from "../../../src/lib/retouch/video";
import { train } from "../brush";
import { buildAlpha } from "../pipeline/masks";
import { readIntrinsics, readPoses } from "../scene";
import { renameImages } from "../sfm/colmap";
import { dirAt, exists, listFiles, publish, readFile, remove, stem, unpublishUnder, writeFile } from "./files";
import { IMAGE_EXT, layout, readState, saveState } from "./project";

const clipDir = (root: string, id: string) => `${layout(root).retouch}/${id}`;

/** Ids viram nomes de pasta: só letras, dígitos e hífen. */
const safeId = (id: string) => {
  if (!/^[\w-]+$/.test(id)) throw new Error(`id de clipe inválido: ${id}`);
  return id;
};

async function readClip(root: string, id: string): Promise<RetouchClip | null> {
  try {
    const clip = JSON.parse(await (await readFile(`${clipDir(root, id)}/clip.json`)).text()) as RetouchClip;
    // Os vídeos ganham URL para os <video> da revisão.
    for (const p of [clip.reference, clip.restored]) if (p) publish(p, await readFile(p));
    return clip;
  } catch {
    return null;
  }
}

async function clipIds(root: string): Promise<string[]> {
  const ids: string[] = [];
  try {
    for await (const [name, h] of (await dirAt(layout(root).retouch)).entries()) if (h.kind === "directory") ids.push(name);
  } catch {
    /* ainda sem clipes */
  }
  return ids;
}

let applying = null as AbortController | null;

export const retouchApi: RetouchApi = {
  async cameras(projectDir) {
    const l = layout(projectDir);
    const lens = readIntrinsics(await (await readFile(`${l.model}/cameras.bin`)).arrayBuffer());
    if (!lens) throw new Error("cameras.bin sem câmera legível");
    return { lens, cameras: readPoses(await (await readFile(`${l.model}/images.bin`)).arrayBuffer()) };
  },

  async listClips(projectDir) {
    const clips = await Promise.all((await clipIds(projectDir)).map((id) => readClip(projectDir, id)));
    return clips.filter((c): c is RetouchClip => !!c).sort((a, b) => a.createdAt - b.createdAt);
  },

  async saveClip(projectDir, clip, files) {
    const dir = clipDir(projectDir, safeId(clip.id));
    const saved = { ...clip };
    for (const [rel, blob] of Object.entries(files)) {
      if (rel.includes("..")) throw new Error(`caminho inválido: ${rel}`);
      const path = `${dir}/${rel}`;
      const file = await writeFile(path, blob);
      if (rel === "reference.mp4") saved.reference = path;
      if (rel === "restored.mp4") saved.restored = path;
      if (rel.endsWith(".mp4")) publish(path, file);
    }
    await writeFile(`${dir}/clip.json`, JSON.stringify(saved));
    return saved;
  },

  async deleteClip(projectDir, id) {
    await remove(clipDir(projectDir, safeId(id)));
  },

  async apply(projectDir, clipIds, options, onEvent) {
    if (applying && !applying.signal.aborted) throw new Error("já existe um retoque em andamento");
    const ctl = new AbortController();
    applying = ctl;
    try {
      return await applyRetouch(projectDir, clipIds, options, onEvent, ctl.signal);
    } finally {
      if (applying === ctl) applying = null;
    }
  },

  async cancel() {
    applying?.abort();
  },

  async undo(projectDir): Promise<ProjectState> {
    const l = layout(projectDir);
    const state = await readState(projectDir);
    if (!state.retouch) return state;
    const base = await readFile(l.retouchBase);
    publish(l.finalPly, await writeFile(l.finalPly, base));
    await remove(l.retouchBase);
    const next: ProjectState = { ...state, retouch: null, lastExport: null };
    await saveState(projectDir, next);
    return next;
  },
};

/** Frames alpha de um clipe, em ordem. */
export const clipAlphaFiles = (root: string, id: string) =>
  listFiles(`${clipDir(root, id)}/alpha`, /^\d+\.png$/).then((names) => names.map((n) => `${clipDir(root, id)}/alpha/${n}`));

const cancelled = () => new DOMException("cancelado", "AbortError");

/** Erosão de 1 px (mínimo 3×3) do alpha binarizado, como os frames reais
 *  (masks.rs, ALPHA_ERODE): a borda do render mistura o fundo. */
function erodedAlpha(src: Uint8ClampedArray, w: number, h: number): Uint8Array {
  const bin = new Uint8Array(w * h);
  for (let i = 0; i < bin.length; i++) bin[i] = src[i * 4] > 127 ? 255 : 0;
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let m = 255;
      for (let dy = -1; dy <= 1 && m; dy++) {
        const yy = Math.min(h - 1, Math.max(0, y + dy));
        for (let dx = -1; dx <= 1; dx++) {
          const v = bin[yy * w + Math.min(w - 1, Math.max(0, x + dx))];
          if (v < m) m = v;
        }
      }
      out[y * w + x] = m;
    }
  return out;
}

async function imageDataOf(blob: Blob, w: number, h: number): Promise<Uint8ClampedArray> {
  const bmp = await createImageBitmap(blob);
  const c = new OffscreenCanvas(w, h);
  const g = c.getContext("2d")!;
  g.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  return g.getImageData(0, 0, w, h).data;
}

/** Treino incremental: o splat de antes (base) é ajustado com as fotos e os
 *  frames restaurados, num dataset à parte (dataset_retouch). Reaplicar
 *  parte sempre da mesma base, então os ajustes não se acumulam. */
async function applyRetouch(
  projectDir: string,
  clipIds: string[],
  options: Parameters<RetouchApi["apply"]>[2],
  onEvent: (e: RetouchEvent) => void,
  signal: AbortSignal,
): Promise<ProjectState> {
  const l = layout(projectDir);
  let state = await readState(projectDir);
  if (!state.finalPly) throw new Error("o projeto ainda não tem splat treinado");
  const progress = (fraction: number, message: Msg) => {
    if (signal.aborted) throw cancelled();
    onEvent({ kind: "progress", fraction, message });
  };
  const log = (line: string) => onEvent({ kind: "log", line });

  // Base: o splat antes do primeiro retoque.
  if (!state.retouch || !(await exists(l.retouchBase))) await writeFile(l.retouchBase, await readFile(l.finalPly));
  const base = await readFile(l.retouchBase);
  publish(l.retouchBase, base);

  const clips = (await Promise.all(clipIds.map((id) => readClip(projectDir, safeId(id))))).filter(
    (c): c is RetouchClip => !!c?.restored,
  );
  if (!clips.length) throw new Error("nenhum clipe restaurado para aplicar");

  const ds = l.retouchDataset;
  await remove(ds);
  const out = await dirAt(`${ds}/images`, true);

  // Fotos reais, como no treino normal (com fundo transparente se houve recorte).
  const realImages = await listFiles(l.images, IMAGE_EXT);
  if (options.masks) {
    await buildAlpha(await dirAt(l.images), await dirAt(l.masks), out, options.masks, log, (f, msg) => progress(0.3 * f, msg));
  } else {
    for (const [i, name] of realImages.entries()) {
      await writeFile(`${ds}/images/${name}`, await readFile(`${l.images}/${name}`));
      progress((0.3 * (i + 1)) / realImages.length, { key: "retouch.dataset", args: { i: i + 1, n: realImages.length } });
    }
  }

  // Frames restaurados: no tamanho em que o restaurador os devolveu (não
  // reduz o que veio maior que a referência), com o alpha do render ampliado.
  const newImages: NewImage[] = [];
  // Só uma parte dos frames entra (restaurados demais tiram a nitidez do que
  // as fotos veem); o vídeo é lido inteiro, frame a frame, de qualquer jeito.
  const picked = pickRestoredFrames(
    clips.map((c) => c.poses.length),
    options.maxRestored ?? Infinity,
  ).map((idx) => new Set(idx));
  const total = picked.reduce((n, s) => n + s.size, 0);
  const trainSize = (c: RetouchClip) => c.restoredSize ?? c.size;
  let done = 0;
  for (const [ci, clip] of clips.entries()) {
    const [w, h] = trainSize(clip);
    const alphas = await clipAlphaFiles(projectDir, clip.id);
    let i = 0;
    for await (const frame of videoFrames(await readFile(clip.restored!), w, h)) {
      if (signal.aborted) throw cancelled();
      if (i >= clip.poses.length) break;
      if (!picked[ci].has(i)) {
        i++;
        continue;
      }
      const alpha = erodedAlpha(await imageDataOf(await readFile(alphas[i]), w, h), w, h);
      const rgba = new Uint8ClampedArray(frame.data);
      for (let k = 0; k < alpha.length; k++) rgba[k * 4 + 3] = alpha[k];
      const c = new OffscreenCanvas(w, h);
      c.getContext("2d")!.putImageData(new ImageData(rgba, w, h), 0, 0);
      const name = `retouch_${clip.id}_${String(i).padStart(5, "0")}.png`;
      await writeFile(`${ds}/images/${name}`, await c.convertToBlob({ type: "image/png" }));
      newImages.push({ name, qvec: clip.poses[i].qvec, tvec: clip.poses[i].tvec, camera: ci });
      i++;
      done++;
      progress(0.3 + (0.2 * done) / total, { key: "retouch.restoredFrames", args: { i: done, n: total } });
    }
    if (i !== clip.poses.length) throw new Error(`o vídeo restaurado de "${clip.label}" tem ${i} frames; esperava ${clip.poses.length}`);
  }

  // Modelo: as câmeras e imagens do projeto mais as dos clipes.
  const cams = appendCameras(
    await (await readFile(`${l.model}/cameras.bin`)).arrayBuffer(),
    clips.map((c) => {
      // Intrínsecos na escala do tamanho de treino (a proporção é a mesma).
      const [w, h] = trainSize(c);
      const sx = w / c.size[0], sy = h / c.size[1];
      const k = c.intrinsics;
      return { width: w, height: h, fx: k.fx * sx, fy: k.fy * sy, cx: k.cx * sx, cy: k.cy * sy };
    }),
  );
  let images = await (await readFile(`${l.model}/images.bin`)).arrayBuffer();
  if (options.masks) images = await renameImages(images, (n) => `${stem(n)}.png`).arrayBuffer();
  await writeFile(`${ds}/sparse/0/cameras.bin`, cams.bin);
  await writeFile(`${ds}/sparse/0/images.bin`, appendImages(images, newImages, cams.ids));
  await writeFile(`${ds}/sparse/0/points3D.bin`, await readFile(`${l.model}/points3D.bin`));
  // O Brush começa do splat que achar em init.ply, em vez dos pontos do COLMAP.
  await writeFile(`${ds}/init.ply`, base);
  log(
    `retoque: ${realImages.length} fotos + ${newImages.length} de ${clips.reduce((n, c) => n + c.poses.length, 0)} frames restaurados de ${clips.length} clipe(s)`,
  );

  // Ajuste curto, sem avaliação (as vistas novas cairiam no conjunto de teste).
  const steps = Math.max(options.steps, 1);
  const snapshots = `${l.root}/training_retouch`;
  let snapshot: string | null = null;
  progress(0.5, { key: "train.loading" });
  const ply = await train(
    await dirAt(ds),
    {
      ...options.train,
      totalSteps: steps,
      evalSplitEvery: 0,
      exportEvery: Math.max(500, Math.round(steps / 3)),
      lrMean: options.lrMean,
      growthStopIter: Math.round(steps / 2),
    },
    !!options.masks,
    {
      status: (msg) => progress(0.5, msg),
      dataset: (n) => log(`retoque: ${n} vistas de treino`),
      step: (iter, splats) => onEvent({ kind: "train", event: { kind: "step", iter, splats } }),
      eval: () => {},
      snapshot: (iter, data) => {
        const path = `${snapshots}/export_${iter}.ply`;
        publish(path, new Blob([data as BlobPart]));
        if (snapshot) unpublishUnder(snapshot);
        snapshot = path;
        onEvent({ kind: "train", event: { kind: "snapshot", iter, path } });
      },
      log,
    },
    signal,
  );
  if (snapshot) unpublishUnder(snapshot);
  publish(l.finalPly, await writeFile(l.finalPly, ply as BufferSource));
  await remove(ds);
  state = { ...state, retouch: { base: l.retouchBase, clips: clips.map((c) => c.id), appliedAt: Date.now() }, lastExport: null };
  await saveState(projectDir, state);
  onEvent({ kind: "progress", fraction: 1, message: { key: "retouch.done" } });
  return state;
}
