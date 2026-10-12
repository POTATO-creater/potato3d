// Poses no navegador: features (AKAZE em workers) → matching (WebGPU) →
// SfM incremental (TS) → modelo COLMAP. Prova de conceito.

import type { Msg } from "../../../src/lib/api";
import { colmapFiles } from "./colmap";
import type { FeatureRequest, FeatureResult } from "./features.worker";
import { HammingMatcher } from "./matcher";
import { correspondences, fillMissing, flattenOrbit } from "./orbit.ts";
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
  /** Câmeras forçadas no mesmo plano da órbita. */
  flatOrbit?: boolean;
  /** Pose inferida pelos vizinhos para os frames não registrados. */
  fillMissing?: boolean;
  /** Instante de cada imagem pelo nome, para a inferência seguir a velocidade
   *  da câmera; sem ele, vale a ordem dos nomes. */
  times?: Map<string, number>;
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

/** Instante de cada imagem (na ordem dos nomes); se faltar ou não crescer
 *  (frames de imagens soltas), o número no nome (00012.jpg → 12), que guarda
 *  os buracos dos frames recusados. */
export function orbitTimes(names: string[], times?: Map<string, number>): number[] {
  const t = names.map((n) => times?.get(n));
  if (t.every((v, i) => v !== undefined && (i === 0 || v > t[i - 1]!))) return t as number[];
  const idx = names.map((n) => Number(n.replace(/\.[^.]+$/, "")));
  return idx.every((v, i) => Number.isFinite(v) && (i === 0 || v > idx[i - 1])) ? idx : names.map((_, i) => i);
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
    w.postMessage({ imgs, matches, closedLoop: opts.closedLoop });
  });
  timings.reconstruction = performance.now() - t2;
  log(
    `Reconstrução: ${model.stats.registered}/${names.length} câmeras, ${model.stats.points} pontos, ` +
      `erro médio ${Number(model.stats.meanError).toFixed(2)} px, f = ${model.f.toFixed(1)} em ${(timings.reconstruction / 1000).toFixed(1)} s`,
  );

  // Testes (scripts/web-bench/headless.mjs): `?sfmDrop=0,20,21` tira essas
  // câmeras do modelo, como se a reconstrução não as tivesse registrado;
  // `?sfmRefine=0` desliga o refinamento das poses inferidas.
  const test = new URLSearchParams(globalThis.location?.search ?? "");
  const drop = new Set((test.get("sfmDrop") ?? "").split(",").filter(Boolean).map(Number));
  if (drop.size) {
    for (const i of drop) model.poses[i] = null;
    model.points = model.points
      .map((p) => ({ ...p, obs: p.obs.filter(([i]) => !drop.has(i)) }))
      .filter((p) => p.obs.length >= 2);
    model.stats.registered = model.poses.filter(Boolean).length;
    log(`teste: ${drop.size} câmeras tiradas do modelo`);
  }

  // Correções da órbita, como em colmap.rs (órbita plana) e pipeline.rs.
  const xy = feats.map((f) => f.xy);
  if (opts.flatOrbit) {
    progress(0.99, { key: "poses.flattening" });
    const dev = flattenOrbit(model, xy);
    if (dev == null) log("órbita plana não aplicada: as câmeras quase não formam um arco");
    else {
      model.stats.flat = 1;
      log(`órbita plana: maior desvio de altura corrigido ${(dev * 100).toFixed(2)}% do raio, erro ${Number(model.stats.meanError).toFixed(2)} px`);
    }
  }
  if (opts.fillMissing) {
    const at = orbitTimes(names, opts.times);
    const corr = test.get("sfmRefine") === "0" ? undefined : correspondences(model, xy, matches);
    const { inferred, refined } = fillMissing(model, at, { flat: opts.flatOrbit, correspondences: corr });
    model.stats.inferred = inferred.length;
    model.stats.refined = refined.length;
    if (inferred.length) log(`poses inferidas pelos vizinhos: ${inferred.map((i) => names[i]).join(", ")}`);
    for (const [i, n] of refined) log(`pose de ${names[i]} refinada contra ${n} pontos 3D`);
  }
  timings.total = performance.now() - t0;
  return { model, files: colmapFiles(model, names, feats.map((f) => f.xy)), names, timings };
}
