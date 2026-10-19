// Mask video or image sequence for the "external" masking method: attached by
// the user or returned by a ComfyUI workflow (e.g. LTX 2.5 + IC-LoRA Alpha
// Gen). Lives in <project>/mask_source/, outside the frames, so re-extracting
// frames keeps it.

import {
  ALL_FORMATS,
  BlobSource,
  BufferTarget,
  CanvasSink,
  CanvasSource,
  EncodedPacketSink,
  getFirstEncodableVideoCodec,
  Input,
  Mp4OutputFormat,
  Output,
  Quality,
  type InputVideoTrack,
  type VideoCodec,
} from "mediabunny";
import type { MaskSource, MaskSourceApi, MaskSourceOf, MaskWorkflowInput } from "../../../src/lib/api";
import { fpsStep, IMAGES_FPS, MAX_GAP, maskFrameFor, paddedFrames, rgbaToMask, workflowSize } from "../../../src/lib/masksource";
import { exists, inputImage, listFiles, readFile, remove, video, writeFile } from "./files";

/** Stored meta: the public fields plus where the files are. */
interface Stored extends MaskSource {
  /** Video file name, inside the folder (kind "video"). */
  file?: string;
  /** Source image names in order (image sources): frames made from images
   *  only know their image's name. */
  images?: string[] | null;
}

export const maskSourceDir = (root: string) => `${root}/mask_source`;
const metaPath = (root: string) => `${maskSourceDir(root)}/meta.json`;
const framesDir = (root: string) => `${maskSourceDir(root)}/frames`;

const IMAGE = /\.(png|jpe?g|webp)$/i;
const byName = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
const pad = (i: number) => String(i).padStart(5, "0");
const ext = (name: string, fallback: string) => (/\.[a-z0-9]+$/i.exec(name)?.[0] ?? fallback).toLowerCase();

async function readMeta(root: string): Promise<Stored | null> {
  try {
    return JSON.parse(await (await readFile(metaPath(root))).text()) as Stored;
  } catch {
    return null;
  }
}

const strip = ({ file: _, images: _i, ...m }: Stored): MaskSource => m;
const imageNames = (src: MaskSourceOf) => (src.kind === "images" ? src.paths.map((p) => p.split("/").pop()!) : null);

/** Video track, its size and its frames' timestamps in presentation order. */
async function openVideo(blob: Blob) {
  const input = new Input({ source: new BlobSource(blob), formats: ALL_FORMATS });
  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error("the mask file has no video track");
    const times: number[] = [];
    for await (const p of new EncodedPacketSink(track).packets(undefined, undefined, { metadataOnly: true })) times.push(p.timestamp);
    times.sort((a, b) => a - b);
    if (!times.length) throw new Error("the mask video has no frames");
    return { input, track: track as InputVideoTrack, times };
  } catch (e) {
    input.dispose();
    throw e;
  }
}

/** Replaces the project's mask source. */
async function store(root: string, meta: Stored, files: Record<string, Blob>) {
  await remove(maskSourceDir(root));
  for (const [name, blob] of Object.entries(files)) await writeFile(`${maskSourceDir(root)}/${name}`, blob);
  await writeFile(metaPath(root), JSON.stringify(meta, null, 2));
  cache.delete(root);
  return strip(meta);
}

async function saveVideo(root: string, blob: Blob, name: string, base: Pick<Stored, "positions" | "fps" | "origin" | "images">) {
  const { input, track, times } = await openVideo(blob);
  input.dispose();
  const file = `mask${ext(name, ".mp4")}`;
  const meta: Stored = {
    id: crypto.randomUUID(),
    kind: "video",
    count: times.length,
    width: track.displayWidth,
    height: track.displayHeight,
    ...base,
    createdAt: Date.now(),
    file,
  };
  return store(root, meta, { [file]: blob });
}

// ------------------------------------------------------------ frames reader --

/** Open mask sources, so the preview doesn't reopen the video on every frame. */
const cache = new Map<string, Promise<Reader>>();

interface Reader {
  meta: Stored;
  /** Mask frame k at width×height (255 = subject). */
  frames(ks: number[], width: number, height: number): AsyncGenerator<{ k: number; mask: Uint8Array }>;
  close(): void;
}

function toMask(src: CanvasImageSource, width: number, height: number): Uint8Array {
  const g = new OffscreenCanvas(width, height).getContext("2d", { willReadFrequently: true })!;
  g.imageSmoothingQuality = "high";
  g.drawImage(src, 0, 0, width, height);
  return rgbaToMask(g.getImageData(0, 0, width, height).data);
}

async function openReader(root: string): Promise<Reader> {
  const meta = await readMeta(root);
  if (!meta) throw new Error("no mask video or frames attached; attach them or generate them in ComfyUI (Masking panel)");
  if (meta.kind === "frames") {
    const names = (await listFiles(framesDir(root), IMAGE)).sort(byName);
    return {
      meta,
      async *frames(ks, width, height) {
        for (const k of ks) {
          const bmp = await createImageBitmap(await readFile(`${framesDir(root)}/${names[k]}`));
          const mask = toMask(bmp, width, height);
          bmp.close();
          yield { k, mask };
        }
      },
      close() {},
    };
  }
  const { input, track, times } = await openVideo(await readFile(`${maskSourceDir(root)}/${meta.file}`));
  const sink = new CanvasSink(track, { poolSize: 1 });
  return {
    meta,
    async *frames(ks, width, height) {
      let i = 0;
      for await (const c of sink.canvasesAtTimestamps(ks.map((k) => times[k]))) {
        if (!c) throw new Error(`mask video frame ${ks[i]} could not be decoded`);
        yield { k: ks[i++], mask: toMask(c.canvas, width, height) };
      }
    },
    close: () => input.dispose(),
  };
}

function reader(root: string): Promise<Reader> {
  let r = cache.get(root);
  if (!r) {
    r = openReader(root);
    cache.set(root, r);
    r.catch(() => cache.delete(root));
  }
  return r;
}

function forget(root: string) {
  const r = cache.get(root);
  cache.delete(root);
  void r?.then((x) => x.close(), () => {});
}

/** Masks for the job: `frames` are the project frames, with their time in
 *  the video or their source image; `onMask` writes each one, in mask order. */
export async function externalMasks(
  root: string,
  projectFrames: { name: string; time: number; source?: string; width: number; height: number }[],
  onMask: (name: string, mask: Uint8Array, width: number, height: number) => Promise<void>,
  log: (s: string) => void,
): Promise<{ origin: string; maxGap: number }> {
  forget(root);
  const r = await openReader(root);
  try {
    const { meta } = r;
    log(`mask ${meta.kind === "video" ? "video" : "frames"}: ${meta.count} frames ${meta.width}x${meta.height}, from ${meta.origin}`);
    const frames = projectFrames.map((f) => {
      if (f.source == null) return { ...f, position: f.time };
      const position = meta.images?.indexOf(f.source) ?? -1;
      if (position < 0) throw new Error(`the mask was made for another source (${f.source} is not in it); attach or generate it again`);
      return { ...f, position };
    });
    const wanted = frames.map((f) => ({ ...f, ...maskFrameFor(meta, f.position) }));
    const far = wanted.filter((f) => f.gap > MAX_GAP);
    if (far.length)
      throw new Error(
        `the mask covers ${meta.count} frames and doesn't reach ${far.length} of the ${frames.length} frames (${far[0].name}…): ` +
          "generate or attach it again for the current source and trim",
      );
    const maxGap = Math.max(0, ...wanted.map((f) => f.gap));
    wanted.sort((a, b) => a.index - b.index);
    // Frames of one size go through the reader together.
    const sizes = new Map<string, typeof wanted>();
    for (const f of wanted) {
      const key = `${f.width}x${f.height}`;
      sizes.set(key, [...(sizes.get(key) ?? []), f]);
    }
    for (const group of sizes.values()) {
      const ks = [...new Set(group.map((f) => f.index))];
      for await (const { k, mask } of r.frames(ks, group[0].width, group[0].height))
        for (const f of group) if (f.index === k) await onMask(f.name, mask, f.width, f.height);
    }
    return { origin: meta.origin, maxGap };
  } finally {
    r.close();
  }
}

// ---------------------------------------------------------- workflow input --

const CODECS: VideoCodec[] = ["avc", "hevc", "vp9", "av1"];

async function encoder(width: number, height: number, fps: number) {
  if (typeof VideoEncoder === "undefined") throw new Error("this browser cannot encode video (WebCodecs)");
  const quality = new Quality({ bitrate: 16e6 });
  const codec = await getFirstEncodableVideoCodec(CODECS, { width, height, frameRate: fps, quality });
  if (!codec) throw new Error(`no video codec available for ${width}×${height}`);
  const canvas = new OffscreenCanvas(width, height);
  const g = canvas.getContext("2d")!;
  g.imageSmoothingQuality = "high";
  const target = new BufferTarget();
  const output = new Output({ format: new Mp4OutputFormat({ fastStart: "in-memory" }), target });
  const source = new CanvasSource(canvas, { codec, quality, latencyMode: "quality" });
  output.addVideoTrack(source, { frameRate: fps });
  await output.start();
  let n = 0;
  return {
    /** Draws the frame (stretched to the video size) and adds it. */
    async add(frame: CanvasImageSource) {
      g.drawImage(frame, 0, 0, width, height);
      await source.add(n / fps, 1 / fps);
      n++;
    },
    /** Repeats the last frame. */
    async repeat() {
      await source.add(n / fps, 1 / fps);
      n++;
    },
    get count() {
      return n;
    },
    async finish() {
      await output.finalize();
      return new Blob([target.buffer!], { type: "video/mp4" });
    },
    cancel: () => output.cancel().catch(() => {}),
  };
}

async function workflowInput(src: MaskSourceOf, onProgress: (f: number) => void, signal?: AbortSignal): Promise<MaskWorkflowInput> {
  const check = () => {
    if (signal?.aborted) throw new DOMException("cancelled", "AbortError");
  };
  const positions: number[] = [];
  let enc: Awaited<ReturnType<typeof encoder>> | null = null;
  let fps: number;
  try {
    if (src.kind === "images") {
      fps = IMAGES_FPS;
      const first = await createImageBitmap(inputImage(src.paths[0]));
      const [w, h] = workflowSize(first.width, first.height);
      first.close();
      enc = await encoder(w, h, fps);
      for (const [i, path] of src.paths.entries()) {
        check();
        const bmp = await createImageBitmap(inputImage(path));
        await enc.add(bmp);
        bmp.close();
        positions.push(i);
        onProgress((i + 1) / src.paths.length);
      }
    } else {
      const input = new Input({ source: new BlobSource(video(src.path).file), formats: ALL_FORMATS });
      try {
        const track = await input.getPrimaryVideoTrack();
        if (!track) throw new Error("the video has no video track");
        const step = fpsStep(src.fps);
        fps = src.fps / step;
        const [w, h] = workflowSize(track.displayWidth, track.displayHeight);
        enc = await encoder(w, h, fps);
        const start = src.trim.start;
        const end = src.trim.end ?? (await track.computeDuration());
        const sink = new CanvasSink(track, { width: w, height: h, fit: "fill", poolSize: 2 });
        let i = 0;
        for await (const c of sink.canvases(start, end)) {
          check();
          if (i++ % step) continue;
          await enc.add(c.canvas);
          positions.push(c.timestamp);
          onProgress(Math.min((c.timestamp - start) / Math.max(end - start, 1e-6), 1));
        }
      } finally {
        input.dispose();
      }
    }
    if (!positions.length) throw new Error("no frames in the chosen range");
    while (enc.count < paddedFrames(positions.length)) await enc.repeat();
    const blob = await enc.finish();
    enc = null;
    return { video: blob, fps, positions, sourceFps: src.kind === "video" ? src.fps : null, images: imageNames(src) };
  } finally {
    await enc?.cancel();
  }
}

// --------------------------------------------------------------------- api --

export const maskSourceApi: MaskSourceApi = {
  async get(projectDir) {
    const m = await readMeta(projectDir);
    return m && (await exists(maskSourceDir(projectDir))) ? strip(m) : null;
  },

  async attach(projectDir, files, source) {
    forget(projectDir);
    const fps = source.kind === "video" ? source.fps : null;
    const vid = files.find((f) => f.type.startsWith("video/") || /\.(mp4|m4v|mov|webm|mkv)$/i.test(f.name));
    const images = imageNames(source);
    if (vid) return saveVideo(projectDir, vid, vid.name, { positions: null, fps, origin: vid.name, images });
    const masks = files.filter((f) => IMAGE.test(f.name)).sort((a, b) => byName(a.name, b.name));
    if (!masks.length) throw new Error("choose a mask video or mask images (PNG, JPG or WebP)");
    const bmp = await createImageBitmap(masks[0]);
    const meta: Stored = {
      id: crypto.randomUUID(),
      kind: "frames",
      count: masks.length,
      width: bmp.width,
      height: bmp.height,
      positions: null,
      fps,
      origin: masks.length > 1 ? `${masks[0].name} … ${masks[masks.length - 1].name}` : masks[0].name,
      createdAt: Date.now(),
      images,
    };
    bmp.close();
    return store(projectDir, meta, Object.fromEntries(masks.map((f, i) => [`frames/${pad(i)}${ext(f.name, ".png")}`, f])));
  },

  async saveGenerated(projectDir, blob, input, origin) {
    forget(projectDir);
    // Positions only for the real frames: the padding copies are never used.
    return saveVideo(projectDir, blob, "mask.mp4", { positions: input.positions, fps: input.sourceFps, origin, images: input.images });
  },

  async remove(projectDir) {
    forget(projectDir);
    await remove(maskSourceDir(projectDir));
  },

  workflowInput,

  async video(projectDir) {
    const m = await readMeta(projectDir);
    return m?.kind === "video" && m.file ? readFile(`${maskSourceDir(projectDir)}/${m.file}`).catch(() => null) : null;
  },

  async frame(projectDir, at, width, height) {
    if (!(await exists(metaPath(projectDir)))) return null;
    const r = await reader(projectDir);
    const position = "image" in at ? (r.meta.images?.indexOf(at.image) ?? -1) : at.time;
    if (position < 0) return null;
    const { index, gap } = maskFrameFor(r.meta, position);
    if (gap > MAX_GAP) return null;
    for await (const { mask } of r.frames([index], width, height)) return mask;
    return null;
  },
};
