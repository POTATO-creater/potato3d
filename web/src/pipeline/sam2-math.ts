// Partes puras do SAM 2.1 da versão web (sem ONNX Runtime nem DOM), para
// testar no Node: constantes, parâmetros do banco de memória, conversões de
// layout, pós-processamento da máscara e o prompt.

import type { SubjectPrompt } from "../../../src/lib/api";

export const SIZE = 1024; // entrada do encoder
export const LOW = 256; // máscaras de baixa resolução do decoder
export const TOKENS = 64 * 64; // posições do mapa de features de mais baixa resolução
export const HIDDEN = 256;
export const MEM = 64;
export const NUM_MASKMEM = 7; // o frame atual + 6 memórias
export const MAX_POINTERS = 16;
export const MEAN = [0.485, 0.456, 0.406];
export const STD = [0.229, 0.224, 0.225];

/** Prompt em pixels do frame: caixa [x0, y0, x1, y1] e/ou pontos (label 1 = objeto, 0 = fundo). */
export interface Sam2Prompt {
  box?: [number, number, number, number];
  points?: { x: number; y: number; label: 0 | 1 }[];
}

export interface Params {
  noMemory: Float32Array; // (256)
  tpos: Float32Array[]; // 7 × (64)
  projW: Float32Array; // (64, 256)
  projB: Float32Array; // (64)
}

/** sam2-params.bin: no_memory_embedding (256), memory_temporal_positional_encoding
 *  (7 × 64), peso (64 × 256) e viés (64) da projeção temporal dos ponteiros, em f32. */
export function parseParams(buf: ArrayBuffer): Params {
  const a = new Float32Array(buf);
  let o = 0;
  const take = (n: number) => a.subarray(o, (o += n));
  const noMemory = take(HIDDEN);
  const tpos = Array.from({ length: NUM_MASKMEM }, () => take(MEM));
  const projW = take(MEM * HIDDEN);
  const projB = take(MEM);
  if (o !== a.length) throw new Error("SAM 2.1: parâmetros com tamanho inesperado");
  return { noMemory, tpos, projW, projB };
}

/** Arredonda para bf16 (o Transformers guarda as memórias assim). */
export function roundBf16(a: Float32Array): Float32Array {
  const u = new Uint32Array(a.buffer, a.byteOffset, a.length);
  for (let i = 0; i < u.length; i++) u[i] = (u[i] + 0x7fff + ((u[i] >>> 16) & 1)) & 0xffff0000;
  return a;
}

/** Codificação temporal dos ponteiros: seno 1D de dim 256 → Linear(256, 64). */
export function pointerPosEnc(t: number, p: Params): Float32Array {
  const half = HIDDEN / 2;
  const pe = new Float32Array(HIDDEN);
  for (let i = 0; i < half; i++) {
    const e = t / 10000 ** ((2 * Math.floor(i / 2)) / half);
    pe[i] = Math.sin(e);
    pe[half + i] = Math.cos(e);
  }
  const out = new Float32Array(MEM);
  for (let o = 0; o < MEM; o++) {
    let s = p.projB[o];
    for (let i = 0; i < HIDDEN; i++) s += p.projW[o * HIDDEN + i] * pe[i];
    out[o] = s;
  }
  return out;
}

/**
 * O export do encoder declara a saída do ramo `then` de um If (a interpolação
 * da posição do Hiera, que nunca roda em 1024²) com 5 dimensões, onde a
 * inferência de shapes acha 4; o ORT Web 1.30 recusa o modelo por isso (o ORT
 * de Python 1.19 só avisa). Corrige o dim_value no próprio buffer, um byte, sem
 * mudar o tamanho; o arquivo baixado já teve o SHA-256 conferido.
 */
export function fixVisionShape(model: ArrayBuffer) {
  const at = 10137;
  const want = [0x0a, 0x0a, 0x08, 0x08, 0x07, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x05];
  const b = new Uint8Array(model, at, want.length);
  if (b[10] === 0x04 && want.slice(0, 10).every((v, i) => b[i] === v)) return; // já corrigido
  if (!want.every((v, i) => b[i] === v)) throw new Error("SAM 2.1: encoder com layout inesperado");
  b[10] = 0x04;
}

/** NCHW (1, C, 64, 64) → (4096, 1, C). */
export function toSeq(nchw: Float32Array, c: number): Float32Array {
  const out = new Float32Array(TOKENS * c);
  for (let ch = 0; ch < c; ch++) for (let t = 0; t < TOKENS; t++) out[t * c + ch] = nchw[ch * TOKENS + t];
  return out;
}

/** (4096, C) → NCHW (1, C, 64, 64). */
export function toNchw(seq: Float32Array, c: number): Float32Array {
  const out = new Float32Array(TOKENS * c);
  for (let t = 0; t < TOKENS; t++) for (let ch = 0; ch < c; ch++) out[ch * TOKENS + t] = seq[t * c + ch];
  return out;
}

/** Logits 256² → máscara u8 (sigmoide) no tamanho do frame, bilinear como o
 *  F.interpolate(align_corners=False) do pós-processamento do Transformers. */
export function logitsToMask(logits: Float32Array, W: number, H: number): Uint8Array {
  const out = new Uint8Array(W * H);
  const xs = new Int32Array(W), xf = new Float32Array(W);
  for (let x = 0; x < W; x++) {
    const s = Math.min(Math.max(((x + 0.5) * LOW) / W - 0.5, 0), LOW - 1);
    xs[x] = Math.min(Math.floor(s), LOW - 2);
    xf[x] = s - xs[x];
  }
  for (let y = 0; y < H; y++) {
    const s = Math.min(Math.max(((y + 0.5) * LOW) / H - 0.5, 0), LOW - 1);
    const y0 = Math.min(Math.floor(s), LOW - 2), fy = s - y0;
    const r0 = y0 * LOW, r1 = r0 + LOW;
    for (let x = 0; x < W; x++) {
      const x0 = xs[x], fx = xf[x];
      const top = logits[r0 + x0] * (1 - fx) + logits[r0 + x0 + 1] * fx;
      const bot = logits[r1 + x0] * (1 - fx) + logits[r1 + x0 + 1] * fx;
      const v = top * (1 - fy) + bot * fy;
      out[y * W + x] = Math.round(255 / (1 + Math.exp(-v)));
    }
  }
  return out;
}

/** Frame do prompt: o mais próximo do instante marcado (o primeiro, sem instante). */
export function promptFrame(names: string[], prompt: SubjectPrompt | null, times?: Map<string, number>): number {
  if (prompt?.time == null || !times) return 0;
  let best = 0;
  names.forEach((n, i) => {
    const t = times.get(n), b = times.get(names[best]);
    if (t != null && (b == null || Math.abs(t - prompt.time!) < Math.abs(b - prompt.time!))) best = i;
  });
  return best;
}

/** Prompt em pixels; sem marcação, um ponto no centro (vídeos orbitais giram em volta do sujeito). */
export function promptPixels(prompt: SubjectPrompt | null, w: number, h: number): Sam2Prompt {
  if (!prompt || (!prompt.box && !prompt.points.length)) return { points: [{ x: w / 2, y: h / 2, label: 1 }] };
  return {
    box: prompt.box ? [prompt.box[0] * w, prompt.box[1] * h, prompt.box[2] * w, prompt.box[3] * h] : undefined,
    points: prompt.points.map((p) => ({ x: p.x * w, y: p.y * h, label: p.label })),
  };
}
