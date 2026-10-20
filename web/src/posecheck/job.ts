// Pose check after the poses stage (masked projects): two short training
// screens, one on the even frames and one on the odd ones; each frame is
// compared with the silhouette of the screen that never saw it (score.ts).
// When the orbit disagrees with itself, or always if asked, each frame's pose
// is aligned to that screen's render (align.ts), the screens are retrained on
// the new poses and the alignment repeats for a few rounds.

import type { MaskOptions, Msg, PoseCheckMode, PoseCheckStats } from "../../../src/lib/api";
import type { Training } from "../../pkg/brush_js.js";
import { train } from "../brush";
import { dirAt, exists, readFile, remove, stem, writeFile } from "../host/files";
import { layout } from "../host/project";
import { readViews, type View } from "../mesh/colmap";
import { buildAlpha } from "../pipeline/masks";
import { rewriteImages } from "../sfm/colmap";
import { quatFromMat, transpose3, mulVec3, type Mat3 } from "../sfm/linalg";
import { alignCost, alignStep, applyStep, choosePose, DEFAULT_ALIGN, edgeOffset, prepareTarget, rotationAngle, type AlignOptions, type Intrinsics } from "./align";
import { fillHoles } from "./image";
import { frameValue, openingRadius, orbitScore, xorScore } from "./score";

export interface PoseCheckSettings {
  /** Screens for the check: steps and training resolution (longer side). */
  detect: { steps: number; resolution: number };
  /** Screens for the correction rounds (from the previous screen of the
   *  same frames when `warmStart`). */
  refine: { steps: number; resolution: number };
  rounds: number;
  /** Alignment steps (re-renders) per frame and round. */
  steps: number;
  /** A frame leaves its pose only for one whose alignment cost is at least
   *  this much lower (relative), measured on a render at each pose. With a
   *  margin, frames that already fit stay put, but real corrections are often
   *  small too: 0 only undoes steps that made things worse. */
  accept: number;
  /** Orbit score above which the poses are corrected ("auto"). */
  threshold: number;
  /** Frames are compared at this size (longer side). */
  analysis: number;
  /** Fill the masks' holes before scoring: a misplaced camera moves the
   *  outline, while holes are usually the cutout missing parts of the subject
   *  in some frames (the model then sticks out of them). */
  fillHoles: boolean;
  /** Start each correction screen from the previous screen of its fold (the
   *  check's, then the previous round's): as good as training from scratch
   *  with twice the steps. The check screens always start from scratch, so
   *  scores before and after the correction compare. */
  warmStart: boolean;
  /** Measure the orbit again after the last round (two more screens). */
  verify: boolean;
  align: AlignOptions;
}

export const POSE_CHECK: PoseCheckSettings = {
  detect: { steps: 600, resolution: 480 },
  refine: { steps: 750, resolution: 640 },
  rounds: 3,
  steps: 2,
  accept: 0,
  threshold: 0.028,
  analysis: 640,
  fillHoles: true,
  warmStart: true,
  verify: true,
  align: DEFAULT_ALIGN,
};

/** Development: `localStorage["gausify.poseCheck"]` (JSON) overrides fields. */
function settings(): PoseCheckSettings {
  try {
    const o = JSON.parse(localStorage.getItem("gausify.poseCheck") ?? "null");
    if (o && typeof o === "object") return { ...POSE_CHECK, ...o, align: { ...POSE_CHECK.align, ...(o.align ?? {}) } };
  } catch {
    /* no storage */
  }
  return POSE_CHECK;
}

interface Pose {
  R: Mat3;
  t: [number, number, number];
}

const poseOf = (v: View): Pose => ({ R: Float64Array.from(v.R), t: [v.t[0], v.t[1], v.t[2]] });

/** Brush camera: centre and camera → world quaternion [x, y, z, w]. */
function brushCamera(p: Pose) {
  const c = mulVec3(transpose3(p.R), p.t);
  const [w, x, y, z] = quatFromMat(p.R);
  return { position: new Float32Array([-c[0], -c[1], -c[2]]), rotation: new Float32Array([-x, -y, -z, w]) };
}

interface Frame {
  view: View;
  /** Index in name order; the fold is index % 2. */
  index: number;
  png: string;
}

type Report = { progress(fraction: number, message: Msg): void; log(line: string): void };

/** Frame at analysis size: RGBA with the training alpha. */
async function loadFrame(path: string) {
  const bmp = await createImageBitmap(await readFile(path), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
  const c = new OffscreenCanvas(bmp.width, bmp.height).getContext("2d", { willReadFrequently: true })!;
  c.drawImage(bmp, 0, 0);
  const out = { rgba: c.getImageData(0, 0, bmp.width, bmp.height).data, w: bmp.width, h: bmp.height };
  bmp.close();
  return out;
}

/** Development: what each correction step did (scripts/web-bench/posecheck.mjs). */
export interface PoseCheckTrace {
  round: number;
  name: string;
  step: number;
  contourOffset: number;
  contourPoints: number;
  interiorPoints: number;
  /** The step's rotation (degrees) and translation (dataset units). */
  rotation: number;
  translation: number;
  /** Alignment cost at the step's start (the last trace of a frame, with
   *  step = steps, is the end pose; `kept` = the pose index kept). */
  cost: number;
  kept?: number;
  /** Raw data, for the frames listed in `dump`. */
  data?: { w: number; h: number; K: Intrinsics; rgba: Float32Array; depth: Float32Array; frame: Uint8ClampedArray };
}

export async function checkPoses(
  root: string,
  masks: MaskOptions,
  mode: Exclude<PoseCheckMode, "off">,
  rep: Report,
  signal: AbortSignal,
  trace?: { step(t: PoseCheckTrace): void; dump?: string[] },
): Promise<PoseCheckStats | null> {
  const t0 = performance.now();
  const S = settings();
  const l = layout(root);
  const work = l.poseCheck;
  const camerasBin = await (await readFile(`${l.model}/cameras.bin`)).arrayBuffer();
  let imagesBin = await (await readFile(`${l.model}/images.bin`)).arrayBuffer();
  const views = readViews(camerasBin, imagesBin);
  if (views.length < 8) {
    rep.log(`pose check: skipped (${views.length} cameras)`);
    return null;
  }
  const frames: Frame[] = views.map((view, index) => ({ view, index, png: `${work}/frames/${stem(view.name)}.png` }));
  const poses = new Map(views.map((v) => [v.name, poseOf(v)]));
  const radius = openingRadius(S.analysis);

  // Progress bar: building the frames, then each screen of the phase (the
  // check, then the correction with its own bar).
  let screenAt = 0, phase = { start: 0, screens: 2, from: 0.05 };
  const share = (f: number) => phase.from + ((1 - phase.from) * (screenAt - phase.start + f)) / phase.screens;

  await remove(work);
  try {
    rep.progress(0, { key: "check.frames", args: { i: 0, n: views.length } });
    await buildAlpha(await dirAt(l.images), await dirAt(l.masks), await dirAt(`${work}/frames`, true), masks, rep.log, (f) =>
      rep.progress(0.05 * f, { key: "check.frames", args: { i: Math.round(f * views.length), n: views.length } }),
      S.analysis,
    );
    // One dataset per fold (Brush loads everything in its folder).
    for (const k of [0, 1]) {
      const dir = `${work}/fold${k}`;
      for (const f of ["cameras.bin", "points3D.bin"]) await writeFile(`${dir}/sparse/0/${f}`, await readFile(`${l.model}/${f}`));
      for (const fr of frames) if (fr.index % 2 === k) await writeFile(`${dir}/images/${stem(fr.view.name)}.png`, await readFile(fr.png));
    }

    const plys: (Uint8Array | null)[] = [null, null];
    /** Trains fold k on the current poses and visits each held-out frame. */
    const screen = async (
      k: number,
      budget: { steps: number; resolution: number },
      label: Msg,
      visit: (t: Training, fr: Frame) => Promise<void>,
      { warm = false, first }: { warm?: boolean; first?: (t: Training, held: Frame[]) => Promise<void> } = {},
    ) => {
      const dir = `${work}/fold${k}`;
      const names = new Set(frames.filter((f) => f.index % 2 === k).map((f) => f.view.name));
      await writeFile(
        `${dir}/sparse/0/images.bin`,
        rewriteImages(imagesBin, (r) => {
          if (!names.has(r.name)) return null;
          const p = poses.get(r.name)!;
          return { name: `${stem(r.name)}.png`, q: quatFromMat(p.R), t: p.t };
        }),
      );
      const init = `${dir}/init.ply`;
      if (warm && plys[k]) await writeFile(init, plys[k] as BufferSource);
      else await remove(init);
      const held = frames.filter((f) => f.index % 2 !== k);
      const ply = await train(
        await dirAt(dir),
        {
          totalSteps: budget.steps,
          shDegree: 1,
          maxSplats: 1_000_000,
          maxResolution: budget.resolution,
          exportEvery: 0,
          evalSplitEvery: 0,
          ssimWeight: 0.2,
          depthDistortionWeight: 10,
        },
        true,
        {
          status: () => {},
          dataset: () => {},
          step: (iter) => rep.progress(share((0.8 * iter) / budget.steps), { key: "check.screen", args: { what: label, fold: k + 1, iter, total: budget.steps } }),
          eval: () => {},
          snapshot: () => {},
          log: (line) => rep.log(`pose check: ${line}`),
        },
        signal,
        {
          finish: async (t) => {
            if (first) await first(t, held);
            for (const [i, fr] of held.entries()) {
              rep.progress(share(0.8 + (0.2 * i) / held.length), { key: "check.frame", args: { what: label, i: i + 1, n: held.length } });
              await visit(t, fr);
            }
          },
        },
      );
      if (S.warmStart) plys[k] = ply;
      screenAt++;
    };

    const render = async (t: Training, fr: Frame, p: Pose, w: number, h: number, color: boolean) => {
      const v = fr.view;
      const K: Intrinsics = { fx: (v.fx * w) / v.width, fy: (v.fy * h) / v.height, cx: (v.cx * w) / v.width, cy: (v.cy * h) / v.height };
      const { position, rotation } = brushCamera(p);
      const za = await t.renderDepth(position, rotation, K.fx, K.fy, K.cx, K.cy, w, h);
      if (!za) throw new Error("Brush returned no splats to render");
      const depth = new Float32Array(w * h), alpha = new Float32Array(w * h);
      for (let i = 0; i < w * h; i++) {
        depth[i] = za[2 * i];
        alpha[i] = za[2 * i + 1];
      }
      const rgba = color ? await t.renderColor(position, rotation, K.fx, K.fy, K.cx, K.cy, w, h) : null;
      if (color && !rgba) throw new Error("Brush returned no splats to render");
      return { K, depth, alpha, rgba: rgba ?? null };
    };

    /** Orbit score with the current poses (detection screens). */
    const measure = async (label: Msg) => {
      const values = new Array<number>(frames.length).fill(0);
      for (const k of [0, 1])
        await screen(k, S.detect, label, async (t, fr) => {
          const { rgba, w, h } = await loadFrame(fr.png);
          let mask: Uint8Array = new Uint8Array(w * h);
          for (let i = 0; i < w * h; i++) mask[i] = rgba[4 * i + 3] > 127 ? 1 : 0;
          if (S.fillHoles) mask = fillHoles(mask, w, h);
          const r = await render(t, fr, poses.get(fr.view.name)!, w, h, false);
          values[fr.index] = frameValue(xorScore(r.alpha, mask, w, h, radius));
        });
      return values;
    };

    const before = await measure({ key: "check.measuring" });
    const score = orbitScore(before);
    const flagged = score >= S.threshold;
    rep.log(`pose check: orbit score ${(100 * score).toFixed(2)}% (threshold ${(100 * S.threshold).toFixed(1)}%), ${S.detect.steps} steps × ${S.detect.resolution} px`);
    const result: PoseCheckStats = { score, threshold: S.threshold, frames: before, flagged, rounds: 0, after: null, afterFrames: null, seconds: 0 };
    if (mode === "auto" && !flagged) {
      result.seconds = (performance.now() - t0) / 1000;
      return result;
    }

    phase = { start: screenAt, screens: 2 * S.rounds + (S.verify ? 2 : 0), from: 0 };
    for (let round = 1; round <= S.rounds; round++) {
      const label: Msg = { key: "check.round", args: { round, rounds: S.rounds } };
      const next = new Map<string, Pose>();
      let moved = 0;
      // The screen's halo: the median offset of its edge over the frames it
      // did not see (most of them are right).
      let bias = 0;
      const halo = async (t: Training, held: Frame[]) => {
        const offsets: number[] = [];
        for (const fr of held) {
          const { rgba, w, h } = await loadFrame(fr.png);
          const r = await render(t, fr, poses.get(fr.view.name)!, w, h, false);
          offsets.push(edgeOffset(r.alpha, prepareTarget(rgba, w, h, 1)));
        }
        offsets.sort((a, b) => a - b);
        bias = offsets[offsets.length >> 1] ?? 0;
        rep.log(`pose check: round ${round}, screen halo ${bias.toFixed(2)} px`);
      };
      for (const k of [0, 1])
        await screen(k, S.refine, label, async (t, fr) => {
          const { rgba, w, h } = await loadFrame(fr.png);
          const target = prepareTarget(rgba, w, h, S.align.levels);
          const o = { ...S.align, contourBias: bias };
          let p = poses.get(fr.view.name)!;
          const visited: Pose[] = [], costs: number[] = [];
          for (let s = 0; s <= S.steps; s++) {
            const r = await render(t, fr, p, w, h, true);
            if (s === S.steps) {
              // The end pose: measured, not moved.
              visited.push(p);
              costs.push(alignCost({ w, h, rgba: r.rgba!, depth: r.depth }, target, r.K, o));
              break;
            }
            const step = alignStep({ w, h, rgba: r.rgba!, depth: r.depth }, target, r.K, o);
            visited.push(p);
            costs.push(step.cost);
            p = applyStep(p.R, p.t, step.dx);
            trace?.step({
              round,
              name: fr.view.name,
              step: s,
              contourOffset: step.contourOffset,
              contourPoints: step.contourPoints,
              interiorPoints: step.interiorPoints,
              rotation: (Math.hypot(step.dx[0], step.dx[1], step.dx[2]) * 180) / Math.PI,
              translation: Math.hypot(step.dx[3], step.dx[4], step.dx[5]),
              cost: step.cost,
              data: trace.dump?.includes(fr.view.name) ? { w, h, K: r.K, rgba: r.rgba!, depth: r.depth, frame: rgba } : undefined,
            });
          }
          const kept = choosePose(costs, S.accept);
          trace?.step({ round, name: fr.view.name, step: S.steps, contourOffset: 0, contourPoints: 0, interiorPoints: 0, rotation: 0, translation: 0, cost: costs[S.steps], kept });
          next.set(fr.view.name, visited[kept]);
          moved += rotationAngle(visited[kept].R, poses.get(fr.view.name)!.R);
        }, { warm: S.warmStart, first: halo });
      // Both folds aligned to screens trained on the same poses, then applied.
      for (const [n, p] of next) poses.set(n, p);
      rep.log(`pose check: round ${round}, mean rotation change ${((moved / frames.length) * 180 / Math.PI).toFixed(3)}°`);
      result.rounds = round;
    }
    // The model keeps its points; only the poses change.
    imagesBin = await rewriteImages(imagesBin, (r) => {
      const p = poses.get(r.name);
      return p ? { ...r, q: quatFromMat(p.R), t: p.t } : r;
    }).arrayBuffer();
    await writeFile(`${l.model}/images.bin`, imagesBin);
    if (S.verify) {
      const after = await measure({ key: "check.verifying" });
      result.after = orbitScore(after);
      result.afterFrames = after;
      rep.log(`pose check: orbit score after ${S.rounds} rounds ${(100 * result.after).toFixed(2)}%`);
    }
    result.seconds = (performance.now() - t0) / 1000;
    return result;
  } finally {
    if (await exists(work)) await remove(work);
  }
}
