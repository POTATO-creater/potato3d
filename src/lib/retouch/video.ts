// Retouch clip video in the browser (WebCodecs via Mediabunny): encodes the
// render frame by frame with exact i/fps timestamps (independent of machine
// speed), and reads the restored video back frame by frame.
// Based on web/src/reference/record.ts from the web-orbita-referencia branch.

import {
  ALL_FORMATS,
  BlobSource,
  BufferTarget,
  CanvasSink,
  CanvasSource,
  getFirstEncodableVideoCodec,
  Input,
  Mp4OutputFormat,
  Output,
  Quality,
  type VideoCodec,
} from "mediabunny";

/** H.264 first: it is what ComfyUI (ffmpeg/PyAV) opens without surprises. */
const CODECS: VideoCodec[] = ["avc", "hevc", "vp9", "av1"];

/** Background gray of the reference video (the same mid-gray as masked training). */
export const CLIP_BACKGROUND = 128;

export interface ClipEncoder {
  /** Frame `index` as top-down RGBA, non-premultiplied color. */
  add(index: number, rgba: Uint8ClampedArray): Promise<void>;
  finish(): Promise<Blob>;
  cancel(): Promise<void>;
}

/** High bitrate so the restorer doesn't spend effort on our own compression
 *  artifacts. Mbps. */
const BITRATE = 16;

export async function createClipEncoder(width: number, height: number, fps: number): Promise<ClipEncoder> {
  if (typeof VideoEncoder === "undefined") throw new Error("this browser cannot encode video (WebCodecs)");
  const quality = new Quality({ bitrate: BITRATE * 1e6 });
  const codec = await getFirstEncodableVideoCodec(CODECS, { width, height, frameRate: fps, quality });
  if (!codec) throw new Error(`no video codec available for ${width}×${height}`);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const g = canvas.getContext("2d", { willReadFrequently: false })!;
  const image = g.createImageData(width, height);
  const target = new BufferTarget();
  const output = new Output({ format: new Mp4OutputFormat({ fastStart: "in-memory" }), target });
  // Every frame is a keyframe: the restorer and decoder can access any frame
  // without its neighbours, and quality stays uniform.
  const source = new CanvasSource(canvas, { codec, quality, keyFrameInterval: 0, latencyMode: "quality" });
  output.addVideoTrack(source, { frameRate: fps });
  await output.start();
  return {
    async add(index, rgba) {
      const d = image.data;
      for (let k = 0; k < d.length; k += 4) {
        const a = rgba[k + 3] / 255, bg = (1 - a) * CLIP_BACKGROUND;
        d[k] = rgba[k] * a + bg;
        d[k + 1] = rgba[k + 1] * a + bg;
        d[k + 2] = rgba[k + 2] * a + bg;
        d[k + 3] = 255;
      }
      g.putImageData(image, 0, 0);
      await source.add(index / fps, 1 / fps);
    },
    async finish() {
      await output.finalize();
      return new Blob([target.buffer!], { type: "video/mp4" });
    },
    async cancel() {
      await output.cancel().catch(() => {});
    },
  };
}

/** Frame alpha as a grayscale PNG (255 = subject). */
export async function alphaPng(width: number, height: number, rgba: Uint8ClampedArray): Promise<Blob> {
  const canvas = new OffscreenCanvas(width, height);
  const g = canvas.getContext("2d")!;
  const img = g.createImageData(width, height);
  for (let k = 0; k < rgba.length; k += 4) {
    const a = rgba[k + 3];
    img.data[k] = img.data[k + 1] = img.data[k + 2] = a;
    img.data[k + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  return canvas.convertToBlob({ type: "image/png" });
}

export interface VideoInfo {
  width: number;
  height: number;
  frames: number;
}

/** Size and frame count (counted, not estimated from duration). */
export async function videoInfo(blob: Blob): Promise<VideoInfo> {
  const input = new Input({ source: new BlobSource(blob), formats: ALL_FORMATS });
  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error("the file has no video track");
    const { packetCount } = await track.computePacketStats();
    return { width: track.displayWidth, height: track.displayHeight, frames: packetCount };
  } finally {
    input.dispose();
  }
}

/** Video frames in order, resized to `width`×`height` (the restorer may
 *  return a different size; the aspect ratio is the same). */
export async function* videoFrames(blob: Blob, width: number, height: number): AsyncGenerator<ImageData> {
  const input = new Input({ source: new BlobSource(blob), formats: ALL_FORMATS });
  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error("the file has no video track");
    const sink = new CanvasSink(track, { width, height, fit: "fill", poolSize: 2 });
    for await (const { canvas } of sink.canvases()) {
      const g = (canvas as HTMLCanvasElement | OffscreenCanvas).getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
      yield g.getImageData(0, 0, width, height);
    }
  } finally {
    input.dispose();
  }
}
