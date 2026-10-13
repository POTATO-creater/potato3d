// SAM 2.1 tiny (Meta, Apache-2.0) for video, on ONNX Runtime Web: a prompt
// (box and/or points) on one frame, and the mask of the same object propagated
// through the others using the model's memory (masks consistent across frames,
// which per-frame RMBG doesn't guarantee).
//
// The four graphs come from jax-image-tools/sam21-tiny-video-onnx (an export of
// Transformers' Sam2VideoModel) or, on GPUs with shader-f16, from the fp16
// version of it that we published (scripts/sam2-export/export_fp16.py: half the
// size, ~1.5× faster, same masks). The memory bank is not a graph: it lives here,
// mirroring _prepare_memory_conditioned_features and friends from Transformers
// 4.57, with the three parameters it uses outside the graphs in sam2-params.bin
// (no_memory_embedding, memory_temporal_positional_encoding and the object
// pointer temporal-encoding projection, extracted from facebook/sam2.1-hiera-tiny).

import type { InferenceSession, Tensor } from "onnxruntime-web/webgpu";
import {
  fixVisionShape,
  HIDDEN,
  MAX_POINTERS,
  MEAN,
  MEM,
  NUM_MASKMEM,
  parseParams,
  pointerPosEnc,
  roundBf16,
  SIZE,
  STD,
  TOKENS,
  toNchw,
  toSeq,
  type Params,
  type Sam2Prompt,
} from "./sam2-math";
import { cachedModel, loadModel, loadOrt, type ModelFile, type Ort } from "./segment";

async function loadParams(): Promise<Params> {
  const res = await fetch(new URL("./sam2-params.bin", import.meta.url));
  if (!res.ok) throw new Error(`SAM 2.1: parameters (${res.status})`);
  return parseParams(await res.arrayBuffer());
}

const FP32_REPO = "https://huggingface.co/jax-image-tools/sam21-tiny-video-onnx/resolve/3cb090d9d0b4c897281fdcac7765c0f3a888c49f";
const FP16_REPO = "https://huggingface.co/obraia/sam2.1-tiny-video-onnx-fp16/resolve/c6f0b87d6562f8926ff6619bbd70e1e7400e1569";
const file = (repo: string, prefix: string, name: string, sha256: string, bytes: number): ModelFile => ({
  label: `SAM 2.1 (${name})`,
  file: `${prefix}-${name}.onnx`,
  url: `${repo}/${name}.onnx`,
  sha256,
  bytes,
});

export type Sam2Variant = "fp16" | "fp32";

/** The four graphs of each variant, in order: encoder, decoder, attention and memory encoder. */
export const SAM2_VARIANTS: Record<Sam2Variant, ModelFile[]> = {
  fp32: [
    file(FP32_REPO, "sam21-tiny", "vision_encoder", "98311f41182f40c57cac2522180fff34b53931ce981412ddd8643ec27fb830e3", 109_497_134),
    file(FP32_REPO, "sam21-tiny", "mask_decoder", "6e75023b52586314ab1dcf5a49fb9731a983ffe2b7b2a419bd42cd2b2e77effa", 17_800_545),
    file(FP32_REPO, "sam21-tiny", "memory_attention", "afd5045a8fe48628d4da49a82b77351a43f7554f1f101a5052ab74aae69e3737", 32_364_264),
    file(FP32_REPO, "sam21-tiny", "memory_encoder", "3a97e8ba25180b39e938ab52a0c4d085007b25f3e919f5672d4ecbcdd2ab9ee3", 5_586_836),
  ],
  fp16: [
    file(FP16_REPO, "sam21-tiny-fp16", "vision_encoder", "3d900ac4aa30cba77383e4ec0e61e9dfb3e3bf0598cdb838ffa15d5dd237e82d", 55_234_395),
    file(FP16_REPO, "sam21-tiny-fp16", "mask_decoder", "1c443d3ab5ff98dda8d453d830baa7773f209bf563f544c7c3b7a0b7b982c839", 9_091_532),
    file(FP16_REPO, "sam21-tiny-fp16", "memory_attention", "38bbdd70bda2bffe4666e60ee0c7ed7ce9b63fdc2da0f14d996e4255bf09b995", 16_440_396),
    file(FP16_REPO, "sam21-tiny-fp16", "memory_encoder", "d5c21573d2de61d2c35702c4e50f18215314b0de82bb3f9145e1a075a169345c", 2_837_008),
  ],
};
const bytesOf = (v: Sam2Variant) => SAM2_VARIANTS[v].reduce((n, f) => n + f.bytes, 0);

let variant: Promise<Sam2Variant> | null = null;
/** fp16 where the GPU has shader-f16 (ORT Web uses it on its own); otherwise the fp32 graphs. */
export function sam2Variant(): Promise<Sam2Variant> {
  variant ??= (async () => {
    try {
      const adapter = await navigator.gpu?.requestAdapter();
      return adapter?.features.has("shader-f16") ? "fp16" : "fp32";
    } catch {
      return "fp32";
    }
  })();
  return variant;
}

export const SAM2_LICENSE = "SAM 2.1 (Meta), Apache-2.0";

/** What a frame leaves in the memory bank. */
interface Memory {
  feat: Float32Array; // (4096, 1, 64), rounded to bf16 as in Transformers
  pointer: Float32Array; // (256)
}

export interface Sam2Frame {
  /** Mask logits at 256² (> 0 = object). */
  logits: Float32Array;
  /** > 0: the object appears in the frame. */
  objectScore: number;
}

type Session = InferenceSession;
type Outputs = Record<string, "gpu-buffer">;

const gpu = (names: string[]): Outputs => Object.fromEntries(names.map((n) => [n, "gpu-buffer" as const]));
// Outputs that only pass from one graph to another stay on the GPU: the encoder
// returns ~90 MB of positional encoding per frame that is never read.
const OUTPUTS = [
  gpu(["feats0", "feats1", "feats2", "pos0", "pos1", "pos2"]),
  gpu(["high_res_masks"]),
  {},
  gpu(["memory_pos_enc"]),
];

/** Are the four graphs of this GPU's variant already cached? */
export async function sam2Cached(): Promise<boolean> {
  const files = SAM2_VARIANTS[await sam2Variant()];
  return (await Promise.all(files.map((f) => cachedModel(f)))).every(Boolean);
}

/** The four graphs (cached or downloaded); the fp32 encoder comes out already
 *  patched (the fp16 one was generated without the If that needs the fix). */
async function loadSam2(v: Sam2Variant, onProgress: (done: number, total: number) => void): Promise<ArrayBuffer[]> {
  const files = SAM2_VARIANTS[v];
  const done = files.map(() => 0);
  const models = await Promise.all(
    files.map((f, i) =>
      loadModel(f, (d) => {
        done[i] = d;
        onProgress(done.reduce((a, b) => a + b, 0), bytesOf(v));
      }),
    ),
  );
  if (v === "fp32") fixVisionShape(models[0]);
  return models;
}

/** Sessions for the requested graphs (indices into the four): WebGPU, or WASM as fallback. */
async function openSessions(
  ort: Ort,
  models: ArrayBuffer[],
  which: number[],
  v: Sam2Variant,
): Promise<{ s: Session[]; backend: string }> {
  // One at a time: ONNX Runtime Web refuses two WebGPU sessions being created concurrently.
  const open = async (ep: "webgpu" | "wasm") => {
    const s: Session[] = [];
    for (const i of which)
      s.push(
        await ort.InferenceSession.create(models[i], {
          executionProviders: [ep],
          preferredOutputLocation: ep === "webgpu" ? OUTPUTS[i] : undefined,
        }),
      );
    return s;
  };
  try {
    return { s: await open("webgpu"), backend: `SAM 2.1 · WebGPU${v === "fp16" ? " fp16" : ""}` };
  } catch (e) {
    console.warn("WebGPU unavailable for SAM 2.1, using WASM", e);
    return { s: await open("wasm"), backend: "SAM 2.1 · WASM" };
  }
}

function frameTensor(ort: Ort, frame: ImageBitmap | OffscreenCanvas): Tensor {
  const c = new OffscreenCanvas(SIZE, SIZE);
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(frame, 0, 0, SIZE, SIZE);
  const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const plane = SIZE * SIZE;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++)
    for (let ch = 0; ch < 3; ch++) out[ch * plane + i] = (rgba[4 * i + ch] / 255 - MEAN[ch]) / STD[ch];
  return new ort.Tensor("float32", out, [1, 3, SIZE, SIZE]);
}

/** feats2 + no_memory_embedding: the conditioning frame uses no memory. */
function withoutMemory(ort: Ort, feats2: Float32Array, p: Params): Tensor {
  const f = new Float32Array(feats2.length);
  for (let ch = 0; ch < HIDDEN; ch++)
    for (let t = 0; t < TOKENS; t++) f[ch * TOKENS + t] = feats2[ch * TOKENS + t] + p.noMemory[ch];
  return new ort.Tensor("float32", f, [1, HIDDEN, 64, 64]);
}

/** Box (labels 2 and 3) and points in 1024² input coordinates. */
function promptTensors(ort: Ort, prompt: Sam2Prompt, w: number, h: number): [Tensor, Tensor] {
  const pts: number[] = [], lab: number[] = [];
  const sx = SIZE / w, sy = SIZE / h;
  if (prompt.box) {
    const [x0, y0, x1, y1] = prompt.box;
    pts.push(x0 * sx, y0 * sy, x1 * sx, y1 * sy);
    lab.push(2, 3);
  }
  for (const p of prompt.points ?? []) {
    pts.push(p.x * sx, p.y * sy);
    lab.push(p.label);
  }
  if (!lab.length) throw new Error("SAM 2.1: empty prompt");
  return [
    new ort.Tensor("float32", new Float32Array(pts), [1, 1, lab.length, 2]),
    new ort.Tensor("int32", new Int32Array(lab), [1, 1, lab.length]),
  ];
}

/**
 * Single-frame preview (the cutout the prompt would produce on the marked frame).
 * The encoder runs once per frame; each prompt change only reruns the decoder.
 */
export class Sam2Preview {
  private ort: Ort;
  private vision: Session;
  private decoder: Session;
  private params: Params;
  private key: string | null = null;
  private feats: { f0: Tensor; f1: Tensor; f2: Tensor } | null = null;

  private constructor(ort: Ort, s: Session[], params: Params) {
    this.ort = ort;
    [this.vision, this.decoder] = s;
    this.params = params;
  }

  /** Downloads the four graphs if needed (the cutout step will use them), opens two. */
  static async create(onProgress: (done: number, total: number) => void): Promise<Sam2Preview> {
    const v = await sam2Variant();
    const models = await loadSam2(v, onProgress);
    const [ort, params] = await Promise.all([loadOrt(), loadParams()]);
    return new Sam2Preview(ort, (await openSessions(ort, models, [0, 1], v)).s, params);
  }

  /** 256² logits for the prompt on the frame; `key` identifies the frame (encoder cache). */
  async mask(key: string, image: ImageBitmap, w: number, h: number, prompt: Sam2Prompt): Promise<Float32Array> {
    if (this.key !== key || !this.feats) {
      this.dropFeats();
      const px = frameTensor(this.ort, image);
      const vis = await this.vision.run({ pixel_values: px });
      px.dispose();
      for (const k of ["pos0", "pos1", "pos2"]) vis[k].dispose();
      const f2 = withoutMemory(this.ort, (await vis.feats2.getData()) as Float32Array, this.params);
      vis.feats2.dispose();
      this.feats = { f0: vis.feats0, f1: vis.feats1, f2 };
      this.key = key;
    }
    const [points, labels] = promptTensors(this.ort, prompt, w, h);
    const dec = await this.decoder.run({
      feats0: this.feats.f0,
      feats1: this.feats.f1,
      feats2: this.feats.f2,
      input_points: points,
      input_labels: labels,
    });
    points.dispose();
    labels.dispose();
    const logits = new Float32Array((await dec.pred_masks.getData()) as Float32Array);
    for (const k in dec) dec[k].dispose();
    return logits;
  }

  private dropFeats() {
    if (this.feats) for (const t of Object.values(this.feats)) t.dispose();
    this.feats = null;
    this.key = null;
  }

  release() {
    this.dropFeats();
    void this.vision.release();
    void this.decoder.release();
  }
}

export class Sam2Tracker {
  private ort: Ort;
  private vision: Session;
  private decoder: Session;
  private attention: Session;
  private memory: Session;
  private params: Params;
  readonly backend: string;

  private cond = new Map<number, Memory>();
  private nonCond = new Map<number, Memory>();
  private memPos: Float32Array | null = null; // spatial encoding of the memories: depends only on position
  private visionPos: Float32Array | null = null; // same, for the frame features (seq)
  /** Video frame count (caps the pointers, like num_total_frames in Transformers). */
  private numFrames: number;

  private constructor(ort: Ort, s: Session[], params: Params, backend: string, numFrames: number) {
    this.ort = ort;
    [this.vision, this.decoder, this.attention, this.memory] = s;
    this.params = params;
    this.backend = backend;
    this.numFrames = numFrames;
  }

  /** `v`: graph variant (default: the GPU's). */
  static async create(
    numFrames: number,
    onProgress: (done: number, total: number) => void,
    v?: Sam2Variant,
  ): Promise<Sam2Tracker> {
    v ??= await sam2Variant();
    const models = await loadSam2(v, onProgress);
    const [ort, params] = await Promise.all([loadOrt(), loadParams()]);
    const { s, backend } = await openSessions(ort, models, [0, 1, 2, 3], v);
    return new Sam2Tracker(ort, s, params, backend, numFrames);
  }

  private tensor(data: Float32Array | Int32Array, dims: number[]): Tensor {
    return new this.ort.Tensor(data instanceof Int32Array ? "int32" : "float32", data, dims);
  }

  /**
   * Segments frame `idx` (size w×h). With `prompt`, the frame becomes a
   * conditioning frame (no memory, like a first click); without a prompt it
   * uses the memory of frames already seen. `reverse`: propagating backwards.
   */
  async frame(
    idx: number,
    image: ImageBitmap | OffscreenCanvas,
    w: number,
    h: number,
    prompt?: Sam2Prompt,
    reverse = false,
  ): Promise<Sam2Frame> {
    if (!prompt && !this.cond.size) throw new Error("SAM 2.1: the first frame needs a prompt");
    const px = frameTensor(this.ort, image);
    const vis = await this.vision.run({ pixel_values: px });
    px.dispose();
    vis.pos0.dispose();
    vis.pos1.dispose();
    const feats2 = (await vis.feats2.getData()) as Float32Array;
    this.visionPos ??= toSeq((await vis.pos2.getData()) as Float32Array, HIDDEN);
    vis.pos2.dispose();

    let pix: Tensor, points: Tensor, labels: Tensor;
    if (prompt) {
      pix = withoutMemory(this.ort, feats2, this.params);
      [points, labels] = promptTensors(this.ort, prompt, w, h);
    } else {
      pix = this.tensor(toNchw(await this.conditioned(idx, feats2, reverse), HIDDEN), [1, HIDDEN, 64, 64]);
      points = this.tensor(new Float32Array(2), [1, 1, 1, 2]);
      labels = this.tensor(new Int32Array([-1]), [1, 1, 1]);
    }

    const dec = await this.decoder.run({ feats0: vis.feats0, feats1: vis.feats1, feats2: pix, input_points: points, input_labels: labels });
    vis.feats0.dispose();
    vis.feats1.dispose();
    pix.dispose();
    points.dispose();
    labels.dispose();
    const logits = (await dec.pred_masks.getData()) as Float32Array;
    const pointer = (await dec.object_pointer.getData()) as Float32Array;
    const objectScore = ((await dec.object_score_logits.getData()) as Float32Array)[0];

    const score = this.tensor(new Float32Array([objectScore]), [1, 1]);
    const fromPts = new this.ort.Tensor("bool", new Uint8Array([prompt ? 1 : 0]), [1]);
    const mem = await this.memory.run({
      vision_features: vis.feats2,
      pred_masks_high_res: dec.high_res_masks,
      object_score_logits: score,
      is_mask_from_pts: fromPts,
    });
    vis.feats2.dispose();
    dec.high_res_masks.dispose();
    score.dispose();
    fromPts.dispose();
    const feat = roundBf16(new Float32Array((await mem.memory_tokens.getData()) as Float32Array));
    this.memPos ??= new Float32Array((await mem.memory_pos_enc.getData()) as Float32Array);
    mem.memory_pos_enc.dispose();

    const entry: Memory = { feat, pointer: new Float32Array(pointer) };
    if (prompt) this.cond.set(idx, entry);
    else this.nonCond.set(idx, entry);
    for (const t of [dec, mem]) for (const k in t) t[k].dispose();
    this.prune(idx, reverse);
    return { logits: new Float32Array(logits), objectScore };
  }

  /** Frame features conditioned on memory (memory_attention). */
  private async conditioned(idx: number, feats2: Float32Array, reverse: boolean): Promise<Float32Array> {
    const { tpos } = this.params;
    const blocks: Memory[] = [];
    const blockTpos: Float32Array[] = [];
    // Conditioning frames (all of them), with the offset-0 temporal encoding (the last one).
    for (const m of this.cond.values()) {
      blocks.push(m);
      blockTpos.push(tpos[NUM_MASKMEM - 1]);
    }
    for (let off = NUM_MASKMEM - 1; off > 0; off--) {
      const m = this.nonCond.get(reverse ? idx + off : idx - off);
      if (m) {
        blocks.push(m);
        blockTpos.push(tpos[off - 1]);
      }
    }
    const memPos = this.memPos!;
    const spatial = new Float32Array(blocks.length * TOKENS * MEM);
    const spatialPos = new Float32Array(spatial.length);
    blocks.forEach((m, b) => {
      spatial.set(m.feat, b * TOKENS * MEM);
      const tp = blockTpos[b];
      for (let t = 0; t < TOKENS; t++)
        for (let c = 0; c < MEM; c++) spatialPos[(b * TOKENS + t) * MEM + c] = memPos[t * MEM + c] + tp[c];
    });

    // Pointers: conditioning frames up to the current one, then the most recent tracked ones.
    const maxPtrs = Math.min(this.numFrames, MAX_POINTERS);
    const sign = reverse ? -1 : 1;
    const ptrs: { diff: number; m: Memory }[] = [];
    for (const [t, m] of this.cond) if (reverse ? t >= idx : t <= idx) ptrs.push({ diff: (idx - t) * sign, m });
    for (let d = 1; d < maxPtrs; d++) {
      const t = reverse ? idx + d : idx - d;
      if (t < 0 || t >= this.numFrames) break;
      const m = this.nonCond.get(t);
      if (m) ptrs.push({ diff: d, m });
    }
    const split = HIDDEN / MEM; // each 256-d pointer becomes 4 tokens of 64
    const pointer = new Float32Array(ptrs.length * HIDDEN);
    const pointerPos = new Float32Array(ptrs.length * HIDDEN);
    ptrs.forEach(({ diff, m }, i) => {
      pointer.set(m.pointer, i * HIDDEN);
      const pe = pointerPosEnc(diff / Math.max(maxPtrs - 1, 1), this.params);
      for (let s = 0; s < split; s++) pointerPos.set(pe, (i * split + s) * MEM);
    });

    const feeds = {
      current_vision_features: this.tensor(toSeq(feats2, HIDDEN), [TOKENS, 1, HIDDEN]),
      current_vision_position_embeddings: this.tensor(this.visionPos!, [TOKENS, 1, HIDDEN]),
      spatial_memory: this.tensor(spatial, [blocks.length * TOKENS, 1, MEM]),
      spatial_memory_position_embeddings: this.tensor(spatialPos, [blocks.length * TOKENS, 1, MEM]),
      pointer_memory: this.tensor(pointer, [ptrs.length * split, 1, MEM]),
      pointer_memory_position_embeddings: this.tensor(pointerPos, [ptrs.length * split, 1, MEM]),
    };
    const res = await this.attention.run(feeds);
    const out = new Float32Array((await res.conditioned_features.getData()) as Float32Array);
    for (const t of Object.values(feeds)) t.dispose();
    res.conditioned_features.dispose();
    return out; // (1, 1, 4096, 256) = seq
  }

  /** Keeps only memories that can still be used: those of the latest frames
   *  and those near the conditioning frames (the backward pass starts there). */
  private prune(idx: number, reverse: boolean) {
    for (const t of this.nonCond.keys()) {
      const recent = reverse ? t - idx < MAX_POINTERS && t >= idx : idx - t < MAX_POINTERS && t <= idx;
      const nearCond = [...this.cond.keys()].some((c) => Math.abs(t - c) < MAX_POINTERS);
      if (!recent && !nearCond) this.nonCond.delete(t);
    }
  }

  release() {
    for (const s of [this.vision, this.decoder, this.attention, this.memory]) void s.release();
  }
}
