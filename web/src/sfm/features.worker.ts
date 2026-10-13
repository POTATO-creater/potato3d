// AKAZE feature extraction (OpenCV.js) in a Web Worker. Takes an image file and
// returns keypoints in original-image pixels, binary descriptors (486 bits in
// 16 u32) and each keypoint's color.

import cvModule from "@techstark/opencv-js";

export interface FeatureRequest {
  id: number;
  file: File;
  maxSide: number;
  maxFeatures: number;
  /** Frame mask (PNG, same size): 0 = ignore, as in COLMAP. */
  mask?: File;
}

export interface FeatureResult {
  id: number;
  width: number;
  height: number;
  n: number;
  xy: Float32Array;
  desc: Uint32Array;
  rgb: Uint8Array;
  ms: number;
}

type CV = typeof cvModule;

interface Akaze {
  setThreshold(t: number): void;
  detectAndCompute(img: InstanceType<CV["Mat"]>, mask: InstanceType<CV["Mat"]>, kps: InstanceType<CV["KeyPointVector"]>, desc: InstanceType<CV["Mat"]>): void;
  delete(): void;
}

// The Emscripten module has `.then`: resolving a Promise with it makes the
// Promise "adopt" the thenable and never settle. Hence the wrapper.
const ready: Promise<{ cv: CV }> = new Promise((resolve) => {
  const m = cvModule as unknown as CV & { onRuntimeInitialized?: () => void };
  if (typeof (m as { Mat?: unknown }).Mat === "function") resolve({ cv: m });
  else m.onRuntimeInitialized = () => resolve({ cv: m });
});

self.onmessage = async (ev: MessageEvent<FeatureRequest>) => {
  const { id, file, maxSide, maxFeatures, mask } = ev.data;
  const { cv } = await ready;
  const t0 = performance.now();
  const bmp = await createImageBitmap(file);
  const W = bmp.width, H = bmp.height;
  const canvas = new OffscreenCanvas(W, H);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  const rgba = ctx.getImageData(0, 0, W, H);

  const src = cv.matFromImageData(rgba);
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  src.delete();
  const scale = Math.min(1, maxSide / Math.max(W, H));
  if (scale < 1) cv.resize(gray, gray, new cv.Size(Math.round(W * scale), Math.round(H * scale)), 0, 0, cv.INTER_AREA);

  // Adaptive threshold: smooth scenes (common in AI video) yield few points at
  // AKAZE's default.
  // The package's type definitions omit AKAZE, which exists in the build.
  const akaze = new (cv as unknown as { AKAZE: new () => Akaze }).AKAZE();
  let kps = new cv.KeyPointVector();
  let desc = new cv.Mat();
  const none = new cv.Mat();
  for (const thr of [0.0008, 0.0002, 0.00005]) {
    kps.delete();
    desc.delete();
    kps = new cv.KeyPointVector();
    desc = new cv.Mat();
    akaze.setThreshold(thr);
    akaze.detectAndCompute(gray, none, kps, desc);
    if (kps.size() >= maxFeatures * 0.6) break;
  }
  none.delete();
  akaze.delete();
  gray.delete();

  let maskPx: Uint8ClampedArray | null = null;
  if (mask) {
    const mb = await createImageBitmap(mask);
    const mc = new OffscreenCanvas(W, H).getContext("2d", { willReadFrequently: true })!;
    mc.drawImage(mb, 0, 0, W, H);
    mb.close();
    maskPx = mc.getImageData(0, 0, W, H).data;
  }
  const total = kps.size();
  const order = [...Array(total).keys()].filter((i) => {
    if (!maskPx) return true;
    const p = kps.get(i).pt;
    const x = Math.min(W - 1, Math.floor((p.x + 0.5) / scale)), y = Math.min(H - 1, Math.floor((p.y + 0.5) / scale));
    return maskPx[(y * W + x) * 4] > 0;
  });
  const resp = new Float32Array(total);
  for (let i = 0; i < total; i++) resp[i] = kps.get(i).response;
  order.sort((a, b) => resp[b] - resp[a]);
  const n = Math.min(order.length, maxFeatures);
  const xy = new Float32Array(n * 2);
  const out = new Uint32Array(n * 16);
  const rgb = new Uint8Array(n * 3);
  const bytes = desc.data as Uint8Array; // n × 61
  const cols = desc.cols;
  const tmp = new Uint8Array(64);
  const tmp32 = new Uint32Array(tmp.buffer);
  for (let k = 0; k < n; k++) {
    const i = order[k];
    const p = kps.get(i).pt;
    // OpenCV centers pixels at integer coordinates; COLMAP at +0.5.
    const x = (p.x + 0.5) / scale, y = (p.y + 0.5) / scale;
    xy[2 * k] = x;
    xy[2 * k + 1] = y;
    tmp.fill(0);
    tmp.set(bytes.subarray(i * cols, i * cols + cols));
    out.set(tmp32, k * 16);
    const px = Math.min(W - 1, Math.max(0, Math.floor(x)));
    const py = Math.min(H - 1, Math.max(0, Math.floor(y)));
    const o = (py * W + px) * 4;
    rgb[3 * k] = rgba.data[o];
    rgb[3 * k + 1] = rgba.data[o + 1];
    rgb[3 * k + 2] = rgba.data[o + 2];
  }
  kps.delete();
  desc.delete();
  const res: FeatureResult = { id, width: W, height: H, n, xy, desc: out, rgb, ms: performance.now() - t0 };
  (self as unknown as Worker).postMessage(res, [xy.buffer, out.buffer, rgb.buffer]);
};
