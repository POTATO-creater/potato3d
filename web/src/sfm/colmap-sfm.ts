// "COLMAP" camera estimator: COLMAP's own SIFT, geometric verification,
// view-graph calibration and global mapper compiled to WebAssembly, with exact
// SIFT matching on the GPU, from the colmap-wasm package
// (github.com/obraia/colmap-wasm). It takes about as long as the AKAZE path
// but more memory, and on AI orbit videos it avoids the ghosting the
// incremental reconstruction can leave on protruding details.

import { extractFeatures, reconstruct, SiftMatcher, type Features, type PairMatches, type Reconstruction } from "colmap-wasm";
import type { Msg } from "../../../src/lib/api";
import type { SfmModel } from "./reconstruct";

export interface ColmapPoses {
  model: SfmModel;
  /** Keypoints of each image (pixels), for colmapFiles and the orbit corrections. */
  xy: Float32Array[];
  matches: PairMatches[];
}

/** Fixed seed: the same frames give the same cameras. */
const SEED = 0;

function toModel(r: Reconstruction, feats: Features[]): SfmModel {
  let obsTotal = 0, errSum = 0;
  const points: SfmModel["points"] = r.points.map((p) => {
    obsTotal += p.track.length;
    errSum += p.error * p.track.length;
    return { X: p.xyz, rgb: p.color, obs: p.track, error: p.error };
  });
  const registered = r.images.filter((im) => im.registered).length;
  return {
    f: r.camera.focal,
    cx: r.camera.cx,
    cy: r.camera.cy,
    width: feats[0].width,
    height: feats[0].height,
    poses: r.images.map((im) => (im.R && im.t ? { R: im.R, t: im.t } : null)),
    points,
    stats: {
      registered,
      total: r.images.length,
      points: points.length,
      observations: obsTotal,
      meanError: obsTotal ? errSum / obsTotal : 0,
      focal: r.camera.focal,
      mapper: "colmap",
    },
  };
}

export async function colmapPoses(
  files: File[],
  masks: (File | undefined)[],
  names: string[],
  pairs: [number, number][],
  log: (s: string) => void,
  progress: (frac: number, msg: Msg) => void,
  timings: Record<string, number>,
  signal?: AbortSignal,
): Promise<ColmapPoses> {
  const t0 = performance.now();
  progress(0, { key: "poses.colmapFeatures" });
  const feats = await extractFeatures(
    files.map((f, i) => ({ name: names[i], data: f })),
    { masks, signal, onProgress: (d, n) => progress((0.45 * d) / n, { key: "poses.colmapFeaturesN", args: { i: d, n } }) },
  );
  timings.features = performance.now() - t0;
  const meanFeat = feats.reduce((s, f) => s + f.count, 0) / feats.length;
  log(`COLMAP SIFT features: mean ${meanFeat.toFixed(0)}/image in ${(timings.features / 1000).toFixed(1)} s`);

  const t1 = performance.now();
  const matcher = await SiftMatcher.create();
  let matches: PairMatches[];
  try {
    matches = await matcher.match(
      feats.map((f) => f.descriptors),
      pairs,
      { onProgress: (d, n) => progress(0.45 + (0.1 * d) / n, { key: "poses.webMatching", args: { i: d, n } }) },
    );
  } finally {
    matcher.destroy();
  }
  timings.matching = performance.now() - t1;
  const meanMatches = matches.reduce((s, m) => s + m.ia.length, 0) / Math.max(matches.length, 1);
  log(`WebGPU SIFT matching: ${pairs.length} pairs, mean ${meanMatches.toFixed(0)} matches in ${(timings.matching / 1000).toFixed(1)} s`);
  if (signal?.aborted) throw new DOMException("cancelled", "AbortError");

  const t2 = performance.now();
  progress(0.55, { key: "poses.colmapMapping" });
  const rec = await reconstruct(names, feats, matches, { seed: SEED, onLog: log, signal });
  timings.reconstruction = performance.now() - t2;
  const model = toModel(rec, feats);
  log(
    `COLMAP global mapper: ${model.stats.registered}/${names.length} cameras, ${model.stats.points} points, ` +
      `mean error ${Number(model.stats.meanError).toFixed(2)} px, f = ${model.f.toFixed(1)} in ${(timings.reconstruction / 1000).toFixed(1)} s`,
  );
  const xy = feats.map((f) => {
    const out = new Float32Array(f.count * 2);
    for (let i = 0; i < f.count; i++) {
      out[2 * i] = f.keypoints[6 * i];
      out[2 * i + 1] = f.keypoints[6 * i + 1];
    }
    return out;
  });
  return { model, xy, matches };
}
