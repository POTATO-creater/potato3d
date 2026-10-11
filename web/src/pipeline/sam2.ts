// SAM 2.1 tiny (Meta, Apache-2.0) para vídeo, no ONNX Runtime Web: um prompt
// (caixa e/ou pontos) em um frame e a máscara do mesmo objeto propagada pelos
// outros, com a memória do modelo (máscaras consistentes entre frames, que o
// RMBG, frame a frame, não garante).
//
// Os quatro grafos vêm de jax-image-tools/sam21-tiny-video-onnx (export do
// Sam2VideoModel do Transformers). O banco de memória não é grafo: fica aqui,
// espelhando _prepare_memory_conditioned_features e companhia do Transformers
// 4.57, com os três parâmetros que ele usa fora dos grafos em sam2-params.bin
// (no_memory_embedding, memory_temporal_positional_encoding e a projeção da
// codificação temporal dos ponteiros, extraídos de facebook/sam2.1-hiera-tiny).

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
import { loadModel, loadOrt, type ModelFile, type Ort } from "./segment";

async function loadParams(): Promise<Params> {
  const res = await fetch(new URL("./sam2-params.bin", import.meta.url));
  if (!res.ok) throw new Error(`SAM 2.1: parâmetros (${res.status})`);
  return parseParams(await res.arrayBuffer());
}

const REPO = "https://huggingface.co/jax-image-tools/sam21-tiny-video-onnx/resolve/3cb090d9d0b4c897281fdcac7765c0f3a888c49f";
const file = (name: string, sha256: string, bytes: number): ModelFile => ({
  label: `SAM 2.1 (${name})`,
  file: `sam21-tiny-${name}.onnx`,
  url: `${REPO}/${name}.onnx`,
  sha256,
  bytes,
});

export const SAM2_FILES = {
  vision: file("vision_encoder", "98311f41182f40c57cac2522180fff34b53931ce981412ddd8643ec27fb830e3", 109_497_134),
  decoder: file("mask_decoder", "6e75023b52586314ab1dcf5a49fb9731a983ffe2b7b2a419bd42cd2b2e77effa", 17_800_545),
  attention: file("memory_attention", "afd5045a8fe48628d4da49a82b77351a43f7554f1f101a5052ab74aae69e3737", 32_364_264),
  memory: file("memory_encoder", "3a97e8ba25180b39e938ab52a0c4d085007b25f3e919f5672d4ecbcdd2ab9ee3", 5_586_836),
};
export const SAM2_BYTES = Object.values(SAM2_FILES).reduce((n, f) => n + f.bytes, 0);
export const SAM2_LICENSE = "SAM 2.1 (Meta), Apache-2.0";

/** O que um frame deixa no banco de memória. */
interface Memory {
  feat: Float32Array; // (4096, 1, 64), arredondado a bf16 como no Transformers
  pointer: Float32Array; // (256)
}

export interface Sam2Frame {
  /** Logits da máscara em 256² (> 0 = objeto). */
  logits: Float32Array;
  /** > 0: o objeto aparece no frame. */
  objectScore: number;
}

type Session = InferenceSession;

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
  private memPos: Float32Array | null = null; // codificação espacial das memórias: só depende da posição
  private visionPos: Float32Array | null = null; // idem, das features do frame (seq)
  /** Frames do vídeo (limita os ponteiros, como num_total_frames no Transformers). */
  private numFrames: number;

  private constructor(ort: Ort, s: Session[], params: Params, backend: string, numFrames: number) {
    this.ort = ort;
    [this.vision, this.decoder, this.attention, this.memory] = s;
    this.params = params;
    this.backend = backend;
    this.numFrames = numFrames;
  }

  static async create(numFrames: number, onProgress: (done: number, total: number) => void): Promise<Sam2Tracker> {
    const files = [SAM2_FILES.vision, SAM2_FILES.decoder, SAM2_FILES.attention, SAM2_FILES.memory];
    const done = files.map(() => 0);
    const models = await Promise.all(
      files.map((f, i) =>
        loadModel(f, (d) => {
          done[i] = d;
          onProgress(done.reduce((a, b) => a + b, 0), SAM2_BYTES);
        }),
      ),
    );
    fixVisionShape(models[0]);
    const [ort, params] = await Promise.all([loadOrt(), loadParams()]);
    // Saídas que só passam de um grafo para outro ficam na GPU: o encoder
    // devolve ~90 MB de codificação posicional por frame que nunca é lida.
    const gpu = (names: string[]) => Object.fromEntries(names.map((n) => [n, "gpu-buffer" as const]));
    const outputs = [
      gpu(["feats0", "feats1", "feats2", "pos0", "pos1", "pos2"]),
      gpu(["high_res_masks"]),
      {},
      gpu(["memory_pos_enc"]),
    ];
    // Uma de cada vez: o ONNX Runtime Web recusa duas sessões WebGPU sendo criadas juntas.
    const open = async (ep: "webgpu" | "wasm") => {
      const s: Session[] = [];
      for (const [i, m] of models.entries())
        s.push(
          await ort.InferenceSession.create(m, {
            executionProviders: [ep],
            preferredOutputLocation: ep === "webgpu" ? outputs[i] : undefined,
          }),
        );
      return s;
    };
    try {
      return new Sam2Tracker(ort, await open("webgpu"), params, "SAM 2.1 · WebGPU", numFrames);
    } catch (e) {
      console.warn("WebGPU indisponível para o SAM 2.1, usando WASM", e);
      return new Sam2Tracker(ort, await open("wasm"), params, "SAM 2.1 · WASM", numFrames);
    }
  }

  private tensor(data: Float32Array | Int32Array, dims: number[]): Tensor {
    return new this.ort.Tensor(data instanceof Int32Array ? "int32" : "float32", data, dims);
  }

  private static pixels(frame: ImageBitmap | OffscreenCanvas): Float32Array {
    const c = new OffscreenCanvas(SIZE, SIZE);
    const ctx = c.getContext("2d", { willReadFrequently: true })!;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(frame, 0, 0, SIZE, SIZE);
    const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;
    const plane = SIZE * SIZE;
    const out = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++)
      for (let ch = 0; ch < 3; ch++) out[ch * plane + i] = (rgba[4 * i + ch] / 255 - MEAN[ch]) / STD[ch];
    return out;
  }

  /**
   * Segmenta o frame `idx` (de tamanho w×h). Com `prompt`, o frame vira um
   * frame condicionante (sem memória, como um primeiro clique); sem prompt,
   * usa a memória dos frames já vistos. `reverse`: propagando para trás.
   */
  async frame(
    idx: number,
    image: ImageBitmap | OffscreenCanvas,
    w: number,
    h: number,
    prompt?: Sam2Prompt,
    reverse = false,
  ): Promise<Sam2Frame> {
    if (!prompt && !this.cond.size) throw new Error("SAM 2.1: o primeiro frame precisa de um prompt");
    const px = this.tensor(Sam2Tracker.pixels(image), [1, 3, SIZE, SIZE]);
    const vis = await this.vision.run({ pixel_values: px });
    px.dispose();
    vis.pos0.dispose();
    vis.pos1.dispose();
    const feats2 = (await vis.feats2.getData()) as Float32Array;
    this.visionPos ??= toSeq((await vis.pos2.getData()) as Float32Array, HIDDEN);
    vis.pos2.dispose();

    let pix: Tensor, points: Tensor, labels: Tensor;
    if (prompt) {
      const nm = this.params.noMemory;
      const f = new Float32Array(feats2.length);
      for (let ch = 0; ch < HIDDEN; ch++)
        for (let t = 0; t < TOKENS; t++) f[ch * TOKENS + t] = feats2[ch * TOKENS + t] + nm[ch];
      pix = this.tensor(f, [1, HIDDEN, 64, 64]);
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
      if (!lab.length) throw new Error("SAM 2.1: prompt vazio");
      points = this.tensor(new Float32Array(pts), [1, 1, lab.length, 2]);
      labels = this.tensor(new Int32Array(lab), [1, 1, lab.length]);
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

  /** Features do frame condicionadas pela memória (memory_attention). */
  private async conditioned(idx: number, feats2: Float32Array, reverse: boolean): Promise<Float32Array> {
    const { tpos } = this.params;
    const blocks: Memory[] = [];
    const blockTpos: Float32Array[] = [];
    // Frames condicionantes (todos), com a codificação temporal de deslocamento 0 (a última).
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

    // Ponteiros: frames condicionantes até o atual, depois os rastreados mais recentes.
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
    const split = HIDDEN / MEM; // cada ponteiro de 256 vira 4 tokens de 64
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

  /** Mantém só as memórias que ainda podem ser usadas: as dos últimos frames
   *  e as vizinhas dos condicionantes (a volta para trás parte deles). */
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
