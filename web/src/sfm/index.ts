// In-browser poses: features (AKAZE in workers) → matching (WebGPU) →
// incremental SfM (TS) → COLMAP model. Proof of concept.

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
  /** Neighbours matched on each side (like the app's `matchWindow`). */
  window: number;
  closedLoop: boolean;
  /** Masks `<name without extension>.png` (0 = ignore): features only on the subject. */
  masksDir?: FileSystemDirectoryHandle;
  /** Force cameras onto the same orbit plane. */
  flatOrbit?: boolean;
  /** Infer poses from neighbours for unregistered frames. */
  fillMissing?: boolean;
  /** Timestamp of each image by name, so inference follows the camera's speed;
   *  without it, name order is used. */
  times?: Map<string, number>;
  /** Cancels: the workers are terminated and the promise rejects. */
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

/** Same pair selection as the app (colmap.rs: write_pairs). */
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

/** Timestamp of each image (in name order); if missing or not increasing
 *  (frames from loose images), the number in the name (00012.jpg → 12), which
 *  preserves the gaps left by rejected frames. */
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
              if (opts.signal?.aborted) return reject(new DOMException("cancelled", "AbortError"));
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
            w.onerror = (e) => reject(new Error(`features worker: ${e.message}`));
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
  if (names.length < 3) throw new Error("Too few images in images/.");
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
  log(`AKAZE features: mean ${meanFeat.toFixed(0)}/image in ${(timings.features / 1000).toFixed(1)} s`);

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
  log(`WebGPU matching: ${pairs.length} pairs, mean ${meanMatches.toFixed(0)} matches in ${(timings.matching / 1000).toFixed(1)} s`);

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
      reject(new Error(`SfM worker: ${e.message}`));
    };
    opts.signal?.addEventListener("abort", () => {
      w.terminate();
      reject(new DOMException("cancelled", "AbortError"));
    });
    w.postMessage({ imgs, matches, closedLoop: opts.closedLoop });
  });
  timings.reconstruction = performance.now() - t2;
  log(
    `Reconstruction: ${model.stats.registered}/${names.length} cameras, ${model.stats.points} points, ` +
      `mean error ${Number(model.stats.meanError).toFixed(2)} px, f = ${model.f.toFixed(1)} in ${(timings.reconstruction / 1000).toFixed(1)} s`,
  );

  // Tests (scripts/web-bench/headless.mjs): `?sfmDrop=0,20,21` removes those
  // cameras from the model, as if reconstruction had not registered them;
  // `?sfmRefine=0` disables refinement of the inferred poses.
  const test = new URLSearchParams(globalThis.location?.search ?? "");
  const drop = new Set((test.get("sfmDrop") ?? "").split(",").filter(Boolean).map(Number));
  if (drop.size) {
    for (const i of drop) model.poses[i] = null;
    model.points = model.points
      .map((p) => ({ ...p, obs: p.obs.filter(([i]) => !drop.has(i)) }))
      .filter((p) => p.obs.length >= 2);
    model.stats.registered = model.poses.filter(Boolean).length;
    log(`test: ${drop.size} cameras removed from the model`);
  }

  // Orbit corrections, as in colmap.rs (flat orbit) and pipeline.rs.
  const xy = feats.map((f) => f.xy);
  if (opts.flatOrbit) {
    progress(0.99, { key: "poses.flattening" });
    const dev = flattenOrbit(model, xy);
    if (dev == null) log("flat orbit not applied: the cameras barely form an arc");
    else {
      model.stats.flat = 1;
      log(`flat orbit: largest height deviation corrected ${(dev * 100).toFixed(2)}% of the radius, error ${Number(model.stats.meanError).toFixed(2)} px`);
    }
  }
  if (opts.fillMissing) {
    const at = orbitTimes(names, opts.times);
    const corr = test.get("sfmRefine") === "0" ? undefined : correspondences(model, xy, matches);
    const { inferred, refined } = fillMissing(model, at, { flat: opts.flatOrbit, correspondences: corr });
    model.stats.inferred = inferred.length;
    model.stats.refined = refined.length;
    if (inferred.length) log(`poses inferred from neighbours: ${inferred.map((i) => names[i]).join(", ")}`);
    for (const [i, n] of refined) log(`pose of ${names[i]} refined against ${n} 3D points`);
  }
  timings.total = performance.now() - t0;
  return { model, files: colmapFiles(model, names, feats.map((f) => f.xy)), names, timings };
}
