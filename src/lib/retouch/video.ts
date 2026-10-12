// Vídeo dos clipes de retoque no navegador (WebCodecs via Mediabunny):
// codifica o render quadro a quadro, com timestamp exato i/fps (não depende
// da velocidade da máquina), e lê o vídeo restaurado de volta, frame a frame.
// Base: web/src/reference/record.ts da branch web-orbita-referencia.

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

/** H.264 primeiro: é o que o ComfyUI (ffmpeg/PyAV) abre sem surpresa. */
const CODECS: VideoCodec[] = ["avc", "hevc", "vp9", "av1"];

/** Cinza do fundo do vídeo de referência (o mesmo meio do treino com recorte). */
export const CLIP_BACKGROUND = 128;

export interface ClipEncoder {
  /** Frame `index` em RGBA de cima para baixo, cor sem pré-multiplicar. */
  add(index: number, rgba: Uint8ClampedArray): Promise<void>;
  finish(): Promise<Blob>;
  cancel(): Promise<void>;
}

/** Bitrate alto: o restaurador não deve gastar esforço com artefatos de
 *  compressão nossos. Mbps. */
const BITRATE = 16;

export async function createClipEncoder(width: number, height: number, fps: number): Promise<ClipEncoder> {
  if (typeof VideoEncoder === "undefined") throw new Error("este navegador não codifica vídeo (WebCodecs)");
  const quality = new Quality({ bitrate: BITRATE * 1e6 });
  const codec = await getFirstEncodableVideoCodec(CODECS, { width, height, frameRate: fps, quality });
  if (!codec) throw new Error(`nenhum codec de vídeo disponível para ${width}×${height}`);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const g = canvas.getContext("2d", { willReadFrequently: false })!;
  const image = g.createImageData(width, height);
  const target = new BufferTarget();
  const output = new Output({ format: new Mp4OutputFormat({ fastStart: "in-memory" }), target });
  // Todo frame é keyframe: o restaurador e o decodificador acessam qualquer
  // frame sem depender dos vizinhos, e a qualidade fica uniforme.
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

/** Alpha do frame como PNG em tons de cinza (255 = sujeito). */
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

/** Tamanho e número de frames (contados, não estimados pela duração). */
export async function videoInfo(blob: Blob): Promise<VideoInfo> {
  const input = new Input({ source: new BlobSource(blob), formats: ALL_FORMATS });
  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error("o arquivo não tem trilha de vídeo");
    const { packetCount } = await track.computePacketStats();
    return { width: track.displayWidth, height: track.displayHeight, frames: packetCount };
  } finally {
    input.dispose();
  }
}

/** Frames do vídeo em ordem, redimensionados para `width`×`height` (o
 *  restaurador pode devolver outro tamanho; a proporção é a mesma). */
export async function* videoFrames(blob: Blob, width: number, height: number): AsyncGenerator<ImageData> {
  const input = new Input({ source: new BlobSource(blob), formats: ALL_FORMATS });
  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error("o arquivo não tem trilha de vídeo");
    const sink = new CanvasSink(track, { width, height, fit: "fill", poolSize: 2 });
    for await (const { canvas } of sink.canvases()) {
      const g = (canvas as HTMLCanvasElement | OffscreenCanvas).getContext("2d") as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
      yield g.getImageData(0, 0, width, height);
    }
  } finally {
    input.dispose();
  }
}
