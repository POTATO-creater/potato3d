// Mask preview in the panel (Host.maskPreview): the pipeline's model on a single
// frame. Models stay open between previews and are released before a job,
// which opens its own.

import type { MaskMethod, MaskOptions, MaskPreviewer } from "../../../src/lib/api";
import { borderKey, chromaMask } from "../pipeline/chroma";
import { Sam2Preview, sam2Cached } from "../pipeline/sam2";
import { logitsToMask, promptPixels } from "../pipeline/sam2-math";
import { cachedModel, MODELS, Segmenter } from "../pipeline/segment";
import { maskSourceApi } from "./masksource";

let sam2: Promise<Sam2Preview> | null = null;
let rmbg: Promise<Segmenter> | null = null;
// One preview at a time on the GPU; ones that arrive meanwhile wait for the previous.
let queue: Promise<unknown> = Promise.resolve();

function rgba(frame: ImageBitmap) {
  const c = new OffscreenCanvas(frame.width, frame.height).getContext("2d", { willReadFrequently: true })!;
  c.drawImage(frame, 0, 0);
  return c.getImageData(0, 0, frame.width, frame.height).data;
}

async function run(key: string, frame: ImageBitmap, m: MaskOptions, onProgress: (d: number, t: number) => void, at?: Parameters<MaskPreviewer["mask"]>[4]) {
  const { width: w, height: h } = frame;
  if (m.method === "external") {
    const mask = at ? await maskSourceApi.frame(at.projectDir, at, w, h) : null;
    if (!mask) throw new Error("no mask for this frame");
    return mask;
  }
  if (m.method === "chroma") {
    const px = rgba(frame);
    return chromaMask(px, w, h, m.chroma.autoColor ? borderKey(px, w, h) : m.chroma.color, m.chroma);
  }
  if (m.method === "sam2") {
    sam2 ??= Sam2Preview.create(onProgress).catch((e) => {
      sam2 = null;
      throw e;
    });
    const logits = await (await sam2).mask(key, frame, w, h, promptPixels(m.prompt ?? null, w, h));
    return logitsToMask(logits, w, h);
  }
  rmbg ??= Segmenter.create("rmbg", onProgress).catch((e) => {
    rmbg = null;
    throw e;
  });
  return (await rmbg).mask(frame, w, h);
}

export const maskPreview: MaskPreviewer = {
  async ready(method: MaskMethod) {
    if (method === "chroma" || method === "external") return true;
    if (method === "sam2") return sam2 != null || (await sam2Cached());
    return rmbg != null || (await cachedModel(MODELS.rmbg)) != null;
  },
  mask(key, frame, masks, onProgress = () => {}, at) {
    const job = queue.then(() => run(key, frame, masks, onProgress, at));
    queue = job.catch(() => {});
    return job;
  },
  release() {
    const [s, r] = [sam2, rmbg];
    sam2 = rmbg = null;
    queue = queue.then(async () => {
      (await s?.catch(() => null))?.release();
      (await r?.catch(() => null))?.release();
    });
  },
};

/** Waits for the release to finish (before opening a job's sessions). */
export const previewIdle = () => queue;
