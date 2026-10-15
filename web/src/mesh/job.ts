// Mesh export (web): geometric fine-tune of the trained splat, depth maps of
// it from every photo, sparse TSDF fusion on the GPU, then surface extraction
// and simplification in a worker. Recipe and measurements in docs/mesh.md.

import { DEFAULT_CHROMA, type Msg, type TrainOptions } from "../../../src/lib/api";
import { brushRuntime, train } from "../brush";
import { readPly } from "../export/splats";
import { dirAt, exists, readFile, remove, requestPersistence, storageLost, writeFile } from "../host/files";
import { buildAlphaDataset } from "../host/job";
import { layout, readState } from "../host/project";
import { center, cameraToWorldXyzw, readViews, type View } from "./colmap";
import type { MeshReply, MeshRequest } from "./mesh.worker";
import type { BuildStats } from "./process";
import { BlockSet, type DepthFrame, TsdfGpu } from "./tsdf";

export interface MeshOptions {
  /** Target triangle count. */
  triangles: number;
}

/** Fine-tune: Brush steps over the trained splat with the geometric losses
 *  (flat Gaussians whose normals agree with the rendered depth). Development:
 *  `localStorage["gausify.meshSteps"]` shortens it to test the later stages. */
const FINE_TUNE_STEPS = 4000;
const TRAIN_MAX_SIDE = 1920;
function fineTuneSteps() {
  try {
    const n = Number(localStorage.getItem("gausify.meshSteps"));
    if (n > 0) return Math.round(n);
  } catch {
    /* no storage */
  }
  return FINE_TUNE_STEPS;
}
const GEOMETRY_CONFIG = {
  "flatten-weight": 100,
  "normal-consistency-weight": 0.05,
  "depth-distortion-weight": 0,
  "geom-reg-start": 0,
  "geom-reg-every": 2,
  "geom-reg-downscale": 2,
  "geom-reg-crop": true,
  // The Mip-Splatting floor would compensate flat Gaussians to zero opacity.
  "min-scale-factor": 0,
};
/** Depth maps at this fraction of the photo size. */
const DEPTH_SCALE = 0.25;
/** Voxel = bounding-box diagonal of the subject / this. */
const VOXELS_PER_DIAGONAL = 1000;

type Progress = (p: { fraction: number; message: Msg }) => void;

const cancelled = () => new DOMException("cancelled", "AbortError");

/** "Vertical axis: x y z" written by Brush in the .ply header. */
function plyUp(bytes: Uint8Array): [number, number, number] | null {
  const head = new TextDecoder().decode(bytes.subarray(0, 4096));
  const m = /Vertical axis:\s*(\S+)\s+(\S+)\s+(\S+)/.exec(head);
  const v = m ? [+m[1], +m[2], +m[3]] : null;
  return v && v.every(Number.isFinite) && Math.hypot(...v) > 0 ? (v as [number, number, number]) : null;
}

/** Cameras' mean "up" (−Y of the camera, in world space). */
function camerasUp(views: View[]): [number, number, number] {
  const u = [0, 0, 0];
  for (const v of views) for (let k = 0; k < 3; k++) u[k] -= v.R[3 + k];
  return u as [number, number, number];
}

/** Voxel size from the opaque Gaussians' robust bounding box. */
function voxelSize(ply: Uint8Array) {
  const s = readPly(ply);
  const xs: number[][] = [[], [], []];
  for (let i = 0; i < s.count; i++) {
    if (1 / (1 + Math.exp(-s.opacity[i])) < 0.5) continue;
    for (let k = 0; k < 3; k++) xs[k].push(s.pos[3 * i + k]);
  }
  if (xs[0].length < 100) throw new Error("the splat has too few opaque Gaussians to mesh");
  const ext = xs.map((a) => {
    a.sort((x, y) => x - y);
    return a[Math.floor(a.length * 0.99)] - a[Math.floor(a.length * 0.01)];
  });
  return { voxel: Math.hypot(...ext) / VOXELS_PER_DIAGONAL, splats: s.count, shDegree: s.shDegree };
}

/** Image decoded and resized, as RGBA bytes. */
async function pixels(file: Blob, w: number, h: number): Promise<Uint8ClampedArray> {
  const bmp = await createImageBitmap(file, { resizeWidth: w, resizeHeight: h, resizeQuality: "high" });
  const c = new OffscreenCanvas(w, h).getContext("2d", { willReadFrequently: true })!;
  c.drawImage(bmp, 0, 0);
  bmp.close();
  return c.getImageData(0, 0, w, h).data;
}

export async function exportMesh(
  dir: string,
  options: MeshOptions,
  onProgress: Progress,
  signal: AbortSignal = new AbortController().signal,
): Promise<{ name: string; bytes: Uint8Array; stats: BuildStats & { splats: number; seconds: number } }> {
  const t0 = performance.now();
  requestPersistence();
  const l = layout(dir);
  const state = await readState(dir);
  if (!state.finalPly) throw new Error("the project has no trained splat yet");
  if (!(await exists(`${l.model}/images.bin`))) throw new Error("the project has no camera poses");
  const progress = (fraction: number, message: Msg) => {
    if (signal.aborted) throw cancelled();
    onProgress({ fraction, message });
  };

  const views = readViews(
    await (await readFile(`${l.model}/cameras.bin`)).arrayBuffer(),
    await (await readFile(`${l.model}/images.bin`)).arrayBuffer(),
  );
  const splat = new Uint8Array(await (await readFile(l.finalPly)).arrayBuffer());
  const { voxel, splats, shDegree } = voxelSize(splat);
  const trunc = 3 * voxel;
  const up = plyUp(splat) ?? camerasUp(views);

  // Training data: the masked dataset when the project has masks. Without
  // dataset_alpha (a project opened from a package), a temporary one at the
  // fine-tune's resolution: a full-size copy can exceed the browser's quota.
  const masked = !!state.masks && (await exists(l.masks));
  let dataset = masked ? l.alpha : l.dataset;
  let temporary: string | null = null;
  const steps = fineTuneSteps();
  if (masked && !(await exists(`${l.alpha}/sparse/0/images.bin`))) {
    const chroma = !!state.masks?.keyColor; // despill only for a solid-colour background
    temporary = dataset = l.meshDataset;
    await buildAlphaDataset(
      dir,
      { enabled: true, forPoses: false, method: chroma ? "chroma" : "rmbg", chroma: DEFAULT_CHROMA, despill: chroma, despillStrength: 100 },
      (line) => console.info(`[mesh] ${line}`),
      (msg) => progress(0, msg),
      { out: temporary, maxSide: TRAIN_MAX_SIDE },
    );
  }
  const init = `${dataset}/init.ply`;
  await writeFile(init, splat as BufferSource);

  const { device } = await brushRuntime();
  const result: { volume?: MeshRequest["volume"] } = {};
  const trainOpts: TrainOptions = {
    totalSteps: steps,
    shDegree,
    maxSplats: Math.max(2_000_000, splats * 2),
    maxResolution: TRAIN_MAX_SIDE,
    exportEvery: 0,
    evalSplitEvery: 0,
    ssimWeight: 0.2,
  };
  const FT = 0.6; // share of the progress bar for the fine-tune
  try {
    progress(0, { key: "train.loading" });
    await train(
      await dirAt(dataset),
      trainOpts,
      masked,
      {
        status: (msg) => progress(0, msg),
        dataset: () => {},
        step: (iter) => progress((FT * iter) / steps, { key: "mesh.fineTune", args: { iter, total: steps } }),
        eval: () => {},
        snapshot: () => {},
        log: (line) => console.info(`[mesh] ${line}`),
      },
      signal,
      {
        config: GEOMETRY_CONFIG,
        finish: async (t) => {
          // Depth frame of view i at DEPTH_SCALE, valid where the splat is
          // opaque (and inside the subject mask); with the photo's colour.
          const frame = async (v: View, withColor: boolean): Promise<DepthFrame> => {
            const w = Math.max(8, Math.round(v.width * DEPTH_SCALE)), h = Math.max(8, Math.round(v.height * DEPTH_SCALE));
            const sx = w / v.width, sy = h / v.height;
            const [fx, fy, cx, cy] = [v.fx * sx, v.fy * sy, v.cx * sx, v.cy * sy];
            const out = await t.renderDepth(new Float32Array(center(v)), new Float32Array(cameraToWorldXyzw(v)), fx, fy, cx, cy, w, h);
            if (!out) throw new Error("Brush returned no splats to render");
            const mask = masked ? await pixels(await readFile(`${l.masks}/${v.name.replace(/\.[^.]+$/, "")}.png`), w, h) : null;
            const photo = withColor ? await pixels(await readFile(`${l.images}/${v.name}`), w, h) : null;
            const depth = new Float32Array(w * h), rgba = new Uint32Array(w * h);
            for (let i = 0; i < w * h; i++) {
              const valid = out[2 * i + 1] > 0.5 && out[2 * i] > 0 && (!mask || mask[4 * i] > 127);
              if (!valid) continue;
              depth[i] = out[2 * i];
              rgba[i] = photo ? ((255 << 24) | (photo[4 * i + 2] << 16) | (photo[4 * i + 1] << 8) | photo[4 * i]) >>> 0 : 0xff000000;
            }
            return { view: v, width: w, height: h, fx, fy, cx, cy, depth, rgba };
          };
          const set = new BlockSet(voxel);
          for (const [i, v] of views.entries()) {
            progress(FT + (0.1 * i) / views.length, { key: "mesh.depth", args: { i: i + 1, n: views.length } });
            set.markFrame(await frame(v, false), trunc);
          }
          const tsdf = new TsdfGpu(device, set, trunc);
          try {
            for (const [i, v] of views.entries()) {
              progress(FT + 0.1 + (0.15 * i) / views.length, { key: "mesh.fusing", args: { i: i + 1, n: views.length } });
              tsdf.integrate(await frame(v, true));
            }
            result.volume = await tsdf.read();
          } finally {
            tsdf.destroy();
          }
        },
      },
    );
  } catch (e) {
    if (e instanceof DOMException && e.name === "QuotaExceededError")
      throw new Error("not enough browser storage for the mesh export; free disk space or delete old projects");
    throw storageLost(e);
  } finally {
    await remove(init);
    if (temporary) await remove(temporary);
  }
  const vol = result.volume;
  if (!vol) throw new Error("the fine-tune ended before the depth maps were made");

  // Surface, cleanup, simplification and GLB in a worker.
  const w = new Worker(new URL("./mesh.worker.ts", import.meta.url), { type: "module" });
  const abort = () => w.terminate();
  signal.addEventListener("abort", abort);
  try {
    const { glb, stats } = await new Promise<{ glb: Uint8Array; stats: BuildStats }>((resolve, reject) => {
      w.onmessage = (ev: MessageEvent<MeshReply>) => {
        const m = ev.data;
        if (m.kind === "progress") onProgress({ fraction: 0.85 + 0.15 * m.fraction, message: m.message });
        else if (m.kind === "done") resolve(m);
        else reject(new Error(m.message));
      };
      w.onerror = (e) => reject(new Error(e.message ? `mesh worker: ${e.message}` : "could not load the mesh worker; reload the page"));
      signal.addEventListener("abort", () => reject(cancelled()));
      w.postMessage({ volume: vol, options: { triangles: options.triangles, up } } satisfies MeshRequest, [
        vol.blocks.buffer,
        vol.tsdfw.buffer,
        vol.color.buffer,
      ]);
    });
    const k = Math.round(stats.triangles / 1000);
    return { name: `mesh_${k}k.glb`, bytes: glb, stats: { ...stats, splats, seconds: (performance.now() - t0) / 1000 } };
  } finally {
    signal.removeEventListener("abort", abort);
    w.terminate();
  }
}
