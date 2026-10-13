// In-browser AI cutout: RMBG-1.4 on ONNX Runtime Web (WebGPU, with WASM as
// fallback), with the same pre/post-processing as src-tauri/src/masks.rs. Models
// come from the same pinned URLs as the installer, SHA-256 checked, and are
// cached in OPFS.

export type ModelId = "rmbg";

export interface ModelSpec {
  label: string;
  file: string;
  url: string;
  sha256: string;
  bytes: number;
  license: string;
  mean: [number, number, number];
  std: [number, number, number];
}

export const MODELS: Record<ModelId, ModelSpec> = {
  rmbg: {
    label: "RMBG-1.4",
    file: "rmbg14.onnx",
    url: "https://huggingface.co/briaai/RMBG-1.4/resolve/2ceba5a5efaec153162aedea169f76caf9b46cf8/onnx/model.onnx",
    sha256: "8cafcf770b06757c4eaced21b1a88e57fd2b66de01b8045f35f01535ba742e0f",
    bytes: 176_153_355,
    license: "BRIA RMBG-1.4, non-commercial use",
    mean: [0.5, 0.5, 0.5],
    std: [1, 1, 1],
  },
};

/** Input size: 1024² (squashed, aspect ratio not preserved). */
export function inputSize(_id: ModelId, _w: number, _h: number): [number, number] {
  return [1024, 1024];
}

/** RGBA (already at input size) → normalized NCHW. */
export function preprocess(rgba: Uint8ClampedArray, iw: number, ih: number, spec: ModelSpec): Float32Array {
  const plane = iw * ih;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++)
    for (let c = 0; c < 3; c++) out[c * plane + i] = (rgba[4 * i + c] / 255 - spec.mean[c]) / spec.std[c];
  return out;
}

/** Model output → u8 at input size. */
export function postprocess(_id: ModelId, out: Float32Array): Uint8Array {
  const m = new Uint8Array(out.length);
  // min–max stretch, unless the frame is nearly constant
  let lo = Infinity, hi = -Infinity;
  for (const v of out) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const [l, scale] = hi - lo > 0.5 ? [lo, 1 / (hi - lo)] : [0, 1];
  for (let i = 0; i < out.length; i++) m[i] = Math.round(Math.min(1, Math.max(0, (out[i] - l) * scale)) * 255);
  return m;
}

async function sha256(buf: ArrayBuffer) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", buf));
  return [...d].map((b) => b.toString(16).padStart(2, "0")).join("");
}

const modelsDir = async () => (await navigator.storage.getDirectory()).getDirectoryHandle("models", { create: true });

/** A model file downloaded from a pinned URL, SHA-256 checked. */
export type ModelFile = Pick<ModelSpec, "label" | "file" | "url" | "sha256" | "bytes">;

/** Is the file already in the OPFS cache (with the right size)? */
export async function cachedModel(spec: ModelFile): Promise<File | null> {
  try {
    const file = await (await (await modelsDir()).getFileHandle(spec.file)).getFile();
    return file.size === spec.bytes ? file : null;
  } catch {
    return null;
  }
}

/** Downloads the file into the cache, checking size and SHA-256. */
export async function downloadModel(
  spec: ModelFile,
  onProgress: (done: number, total: number) => void,
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  // No Referer: Hugging Face answers 404 (without CORS) when it is a
  // *.workers.dev origin, where the web version is deployed.
  const res = await fetch(spec.url, { signal, referrerPolicy: "no-referrer" });
  if (!res.ok || !res.body) throw new Error(`${spec.label}: download failed (${res.status})`);
  const buf = new Uint8Array(spec.bytes);
  const reader = res.body.getReader();
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (n + value.length > buf.length) throw new Error(`${spec.label}: file larger than expected`);
    buf.set(value, n);
    n += value.length;
    onProgress(n, spec.bytes);
  }
  if (n !== spec.bytes) throw new Error(`${spec.label}: size ${n}, expected ${spec.bytes}`);
  if ((await sha256(buf.buffer)) !== spec.sha256) throw new Error(`${spec.label}: SHA-256 mismatch`);
  const w = await (await (await modelsDir()).getFileHandle(spec.file, { create: true })).createWritable();
  await w.write(buf);
  await w.close();
  return buf.buffer;
}

/** File from the cache, or downloaded on the spot. */
export async function loadModel(spec: ModelFile, onProgress: (done: number, total: number) => void): Promise<ArrayBuffer> {
  const file = await cachedModel(spec);
  return file ? file.arrayBuffer() : downloadModel(spec, onProgress);
}

export type Ort = typeof import("onnxruntime-web/webgpu");

/** The ONNX Runtime .wasm, deployed gzipped (web/ort-wasm-plugin.ts). If the
 *  server already decompressed it (Content-Encoding), use it as is. */
async function ortWasmBinary(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`ONNX Runtime: ${url} (${res.status})`);
  const buf = await res.arrayBuffer();
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] !== 0x1f || head[1] !== 0x8b) return buf;
  return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}

/** ONNX Runtime Web, ready to create sessions. */
export async function loadOrt(): Promise<Ort> {
  const ort = await import("onnxruntime-web/webgpu");
  // Dynamic import: the (Node) tests import this file and can't resolve Vite virtual modules.
  const { default: ortWasmGz } = await import("virtual:ort-wasm-gz");
  if (ortWasmGz) ort.env.wasm.wasmBinary ??= await ortWasmBinary(ortWasmGz);
  return ort;
}

export class Segmenter {
  private ort: Ort;
  private session: import("onnxruntime-web/webgpu").InferenceSession;
  readonly id: ModelId;
  readonly backend: string;

  private constructor(ort: Ort, session: import("onnxruntime-web/webgpu").InferenceSession, id: ModelId, backend: string) {
    this.ort = ort;
    this.session = session;
    this.id = id;
    this.backend = backend;
  }

  static async create(id: ModelId, onProgress: (done: number, total: number) => void): Promise<Segmenter> {
    const model = await loadModel(MODELS[id], onProgress);
    const ort = await loadOrt();
    const opts = { freeDimensionOverrides: { batch_size: 1 } };
    try {
      const s = await ort.InferenceSession.create(model, { ...opts, executionProviders: ["webgpu"] });
      return new Segmenter(ort, s, id, `${MODELS[id].label} · WebGPU`);
    } catch (e) {
      console.warn("WebGPU unavailable for the model, using WASM", e);
      const s = await ort.InferenceSession.create(model, { ...opts, executionProviders: ["wasm"] });
      return new Segmenter(ort, s, id, `${MODELS[id].label} · WASM`);
    }
  }

  /** Mask (255 = subject) at frame size. */
  async mask(frame: ImageBitmap | OffscreenCanvas, w: number, h: number): Promise<Uint8Array> {
    const [iw, ih] = inputSize(this.id, w, h);
    const c = new OffscreenCanvas(iw, ih);
    const ctx = c.getContext("2d", { willReadFrequently: true })!;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(frame, 0, 0, iw, ih);
    const input = preprocess(ctx.getImageData(0, 0, iw, ih).data, iw, ih, MODELS[this.id]);
    const tensor = new this.ort.Tensor("float32", input, [1, 3, ih, iw]);
    const res = await this.session.run({ [this.session.inputNames[0]]: tensor });
    const outName = this.session.outputNames[this.session.outputNames.length - 1];
    const out = (await res[outName].getData()) as Float32Array;
    tensor.dispose();
    for (const k in res) res[k].dispose();
    const small = postprocess(this.id, out);
    return resizeMask(small, iw, ih, w, h);
  }

  release() {
    void this.session.release();
  }
}

/** Resizes a u8 mask with the browser's filter. */
export function resizeMask(m: Uint8Array, w: number, h: number, W: number, H: number): Uint8Array {
  if (w === W && h === H) return m;
  const src = new OffscreenCanvas(w, h);
  const sctx = src.getContext("2d")!;
  const img = sctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = m[i];
    img.data[4 * i + 3] = 255;
  }
  sctx.putImageData(img, 0, 0);
  const dst = new OffscreenCanvas(W, H);
  const dctx = dst.getContext("2d", { willReadFrequently: true })!;
  dctx.imageSmoothingQuality = "high";
  dctx.drawImage(src, 0, 0, W, H);
  const d = dctx.getImageData(0, 0, W, H).data;
  const out = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) out[i] = d[4 * i];
  return out;
}
