// Poses no navegador: features (AKAZE em workers) → matching (WebGPU) →
// SfM incremental (TS) → modelo COLMAP. Prova de conceito.

import type { Msg } from "../../../src/lib/api";
import { colmapFiles } from "./colmap";
import type { FeatureRequest, FeatureResult } from "./features.worker";
import { HammingMatcher } from "./matcher";
import type { ImageFeatures, SfmModel } from "./reconstruct";
import type { SfmWorkerMessage } from "./sfm.worker";

export interface SfmOptions {
  maxSide: number;
  maxFeatures: number;
  /** Vizinhos casados de cada lado (como o `matchWindow` do app). */
  window: number;
  closedLoop: boolean;
  /** Máscaras `<nome sem extensão>.png` (0 = ignorar): features só no sujeito. */
  masksDir?: FileSystemDirectoryHandle;
  /** Cancela: os workers são encerrados e a promessa rejeita. */
  signal?: AbortSignal;
}

export const DEFAULT_SFM: SfmOptions = { maxSide: 1600, maxFeatures: 4000, window: 8, closedLoop: true };

export interface SfmRun {
  model: SfmModel;
  files: Record<string, Blob>;
  names: string[];
  timings: Record<string, number>;
}

const IMAGE_EXT = /\.(jpe?g|png|webp)$/i;

/** Mesma escolha de pares do app (colmap.rs: write_pairs). */
export function orbitPairs(n: number, window: number, closed: boolean): [number, number][] {
  const offsets = [...Array(window).keys()].map((k) => k + 1);
  for (const o of [16, 32, 64]) if (o > window) offsets.push(o);
  const limit = closed ? Math.floor(n / 2) : n - 1;
  const set = new Set<string>();
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++)
    for (const o of offsets) {
      if (o > limit) continue;
      const j = i + o;
      if (j >= n && !closed) continue;
      const a = Math.min(i, j % n), b = Math.max(i, j % n);
      const key = `${a},${b}`;
      if (a !== b && !set.has(key)) {
        set.add(key);
        out.push([a, b]);
      }
    }
  return out;
}

async function extractAll(
  files: File[],
  masks: (File | undefined)[],
  opts: SfmOptions,
  progress: (done: number) => void,
): Promise<FeatureResult[]> {
  const nWorkers = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
  const workers = Array.from(
    { length: nWorkers },
    () => new Worker(new URL("./features.worker.ts", import.meta.url), { type: "module" }),
  );
  const results: FeatureResult[] = new Array(files.length);
  let next = 0, done = 0;
  try {
    await Promise.all(
      workers.map(
        (w) =>
          new Promise<void>((resolve, reject) => {
            const send = () => {
              if (opts.signal?.aborted) return reject(new DOMException("cancelado", "AbortError"));
              if (next >= files.length) return resolve();
              const id = next++;
              const req: FeatureRequest = { id, file: files[id], maxSide: opts.maxSide, maxFeatures: opts.maxFeatures, mask: masks[id] };
              w.postMessage(req);
            };
            w.onmessage = (ev: MessageEvent<FeatureResult>) => {
              results[ev.data.id] = ev.data;
              progress(++done);
              send();
            };
            w.onerror = (e) => reject(new Error(`worker de features: ${e.message}`));
            send();
          }),
      ),
    );
  } finally {
    workers.forEach((w) => w.terminate());
  }
  return results;
}

export async function runSfm(
  dir: FileSystemDirectoryHandle,
  opts: SfmOptions,
  log: (s: string) => void,
  progress: (frac: number, msg: Msg) => void,
): Promise<SfmRun> {
  const timings: Record<string, number> = {};
  const t0 = performance.now();
  const imagesDir = await dir.getDirectoryHandle("images");
  const names: string[] = [];
  for await (const [name, h] of imagesDir.entries()) if (h.kind === "file" && IMAGE_EXT.test(name)) names.push(name);
  names.sort();
  if (names.length < 3) throw new Error("Poucas imagens em images/.");
  const files = await Promise.all(names.map(async (n) => (await imagesDir.getFileHandle(n)).getFile()));

  const masks = await Promise.all(
    names.map(async (n) => {
      if (!opts.masksDir) return undefined;
      return (await opts.masksDir.getFileHandle(n.replace(/\.[^.]+$/, ".png"))).getFile();
    }),
  );
  progress(0, { key: "poses.webFeatures" });
  const feats = await extractAll(files, masks, opts, (d) => progress((0.3 * d) / names.length, { key: "poses.webFeaturesN", args: { i: d, n: names.length } }));
  timings.features = performance.now() - t0;
  const meanFeat = feats.reduce((s, f) => s + f.n, 0) / feats.length;
  log(`Features AKAZE: média ${meanFeat.toFixed(0)}/imagem em ${(timings.features / 1000).toFixed(1)} s`);

  const t1 = performance.now();
  const pairs = orbitPairs(names.length, opts.window, opts.closedLoop);
  const matcher = await HammingMatcher.create();
  let matches;
  try {
    matches = await matcher.match(
      feats.map((f) => f.desc),
      pairs,
      {},
      (d, n) => progress(0.3 + (0.2 * d) / n, { key: "poses.webMatching", args: { i: d, n } }),
    );
  } finally {
    matcher.destroy();
  }
  timings.matching = performance.now() - t1;
  const meanMatches = matches.reduce((s, m) => s + m.ia.length, 0) / Math.max(matches.length, 1);
  log(`Matching WebGPU: ${pairs.length} pares, média ${meanMatches.toFixed(0)} matches em ${(timings.matching / 1000).toFixed(1)} s`);

  const t2 = performance.now();
  const imgs: ImageFeatures[] = feats.map((f, i) => ({ name: names[i], width: f.width, height: f.height, n: f.n, xy: f.xy, rgb: f.rgb }));
  const model = await new Promise<SfmModel>((resolve, reject) => {
    const w = new Worker(new URL("./sfm.worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (ev: MessageEvent<SfmWorkerMessage>) => {
      const m = ev.data;
      if (m.kind === "log") log(m.text);
      else if (m.kind === "progress") progress(0.5 + 0.5 * m.frac, m.msg);
      else {
        w.terminate();
        if (m.kind === "done") resolve(m.model);
        else reject(new Error(m.text));
      }
    };
    w.onerror = (e) => {
      w.terminate();
      reject(new Error(`worker do SfM: ${e.message}`));
    };
    opts.signal?.addEventListener("abort", () => {
      w.terminate();
      reject(new DOMException("cancelado", "AbortError"));
    });
    w.postMessage({ imgs, matches });
  });
  timings.reconstruction = performance.now() - t2;
  timings.total = performance.now() - t0;
  log(
    `Reconstrução: ${model.stats.registered}/${names.length} câmeras, ${model.stats.points} pontos, ` +
      `erro médio ${Number(model.stats.meanError).toFixed(2)} px, f = ${model.f.toFixed(1)} em ${(timings.reconstruction / 1000).toFixed(1)} s`,
  );
  return { model, files: colmapFiles(model, names, feats.map((f) => f.xy)), names, timings };
}
