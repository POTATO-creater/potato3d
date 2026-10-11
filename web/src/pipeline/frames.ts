// Vídeo → frames no navegador, com a mesma lógica de src-tauri/src/frames.rs:
// candidatos uniformes no trecho (targetFrames × oversample, no máximo o fps
// do vídeo), escala que cabe em maxSide × maxSide sem ampliar, e o candidato
// mais nítido (variância do Laplaciano) de cada janela vira 00000.jpg, …
// O próprio navegador decodifica (um seek por candidato). Imagens soltas
// passam pela mesma escolha (framesFromImages).

import type { FrameOptions, VideoInfo } from "../../../src/lib/api";

/** Frame escolhido: nome em images/ e de onde saiu no vídeo. */
export interface PickedFrame {
  name: string;
  time: number;
  sharpness: number;
  /** Imagem de origem, quando não veio de um vídeo. */
  source?: string;
}

/** Janela de cada candidato quando `total` viram `target` frames:
 *  lo(w) = ⌊w·total/target⌋, como no frames.rs. */
function windowsOf(total: number, target: number) {
  const windowOf = new Int32Array(total);
  for (let w = 0; w < target; w++) {
    const lo = Math.floor((w * total) / target);
    const hi = Math.max(Math.floor(((w + 1) * total) / target), lo + 1);
    for (let k = lo; k < Math.min(hi, total); k++) windowOf[k] = w;
  }
  return windowOf;
}

const frameName = (i: number) => `${String(i).padStart(5, "0")}.jpg`;

async function writeBlob(dir: FileSystemDirectoryHandle, name: string, blob: Blob) {
  const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
  await w.write(blob);
  await w.close();
}

function loadVideo(file: File): Promise<HTMLVideoElement> {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = URL.createObjectURL(file);
  return new Promise((resolve, reject) => {
    video.onloadeddata = () => resolve(video);
    video.onerror = () => reject(new Error("O navegador não conseguiu abrir este vídeo (codec não suportado?)."));
  });
}

function release(video: HTMLVideoElement) {
  URL.revokeObjectURL(video.src);
  video.removeAttribute("src");
  video.load();
}

/** fps do contêiner MP4/MOV (como o ffprobe): amostras ÷ duração na trilha
 *  de vídeo (mdhd + stts). null se não for MP4 ou faltar alguma caixa. */
export async function containerFps(file: Blob): Promise<number | null> {
  const read = async (at: number, n: number) => new DataView(await file.slice(at, at + n).arrayBuffer());
  // caixas de primeiro nível até achar o moov (pode estar no fim do arquivo)
  let moov: DataView | null = null;
  for (let at = 0; at + 8 <= file.size; ) {
    const h = await read(at, 16);
    let size = h.getUint32(0);
    const type = String.fromCharCode(h.getUint8(4), h.getUint8(5), h.getUint8(6), h.getUint8(7));
    let head = 8;
    if (size === 1) {
      size = Number(h.getBigUint64(8));
      head = 16;
    } else if (size === 0) size = file.size - at;
    if (size < head) return null;
    if (type === "moov") {
      moov = await read(at + head, size - head);
      break;
    }
    at += size;
  }
  if (!moov) return null;
  const v = moov;
  const type = (o: number) => String.fromCharCode(v.getUint8(o + 4), v.getUint8(o + 5), v.getUint8(o + 6), v.getUint8(o + 7));
  /** Filhos de [start, end) como [tipo, início do conteúdo, fim]. */
  const children = (start: number, end: number) => {
    const out: [string, number, number][] = [];
    for (let o = start; o + 8 <= end; ) {
      const size = v.getUint32(o);
      if (size < 8 || o + size > end) break;
      out.push([type(o), o + 8, o + size]);
      o += size;
    }
    return out;
  };
  const child = (start: number, end: number, name: string) => children(start, end).find((c) => c[0] === name);
  for (const [t, s, e] of children(0, v.byteLength)) {
    if (t !== "trak") continue;
    const mdia = child(s, e, "mdia");
    if (!mdia) continue;
    const hdlr = child(mdia[1], mdia[2], "hdlr");
    const mdhd = child(mdia[1], mdia[2], "mdhd");
    const stbl = (() => {
      const minf = child(mdia[1], mdia[2], "minf");
      return minf && child(minf[1], minf[2], "stbl");
    })();
    const stts = stbl && child(stbl[1], stbl[2], "stts");
    if (!hdlr || !mdhd || !stts || type(hdlr[1] + 4) !== "vide") continue;
    const timescale = v.getUint32(mdhd[1] + (v.getUint8(mdhd[1]) === 1 ? 20 : 12));
    let samples = 0, ticks = 0;
    for (let i = 0, n = v.getUint32(stts[1] + 4); i < n; i++) {
      const count = v.getUint32(stts[1] + 8 + 8 * i);
      samples += count;
      ticks += count * v.getUint32(stts[1] + 12 + 8 * i);
    }
    return samples && ticks && timescale ? Math.round(((samples * timescale) / ticks) * 1000) / 1000 : null;
  }
  return null;
}

/** fps pelo intervalo entre quadros apresentados (requestVideoFrameCallback),
 *  para contêineres sem `containerFps`. O menor intervalo resiste a quadros
 *  pulados (aba em segundo plano), que só aumentam os outros. */
async function measureFps(video: HTMLVideoElement): Promise<number> {
  if (!("requestVideoFrameCallback" in video)) return 30;
  const times: number[] = [];
  const done = new Promise<void>((resolve) => {
    const tick = (_: number, meta: VideoFrameCallbackMetadata) => {
      times.push(meta.mediaTime);
      if (times.length >= 24) resolve();
      else video.requestVideoFrameCallback(tick);
    };
    video.requestVideoFrameCallback(tick);
  });
  video.playbackRate = 1;
  await video.play().catch(() => {});
  await Promise.race([done, new Promise((r) => setTimeout(r, 3000))]);
  video.pause();
  const d = times
    .slice(1)
    .map((t, i) => t - times[i])
    .filter((x) => x > 1e-4);
  if (!d.length) return 30;
  // Cada intervalo cobre round(d/mín) quadros; a média no trecho todo tira o
  // arredondamento dos timestamps (WebM: ms inteiros, 41/42 ms a 24 fps).
  const min = Math.min(...d);
  const frames = d.reduce((s, x) => s + Math.max(1, Math.round(x / min)), 0);
  return Math.round((frames / d.reduce((s, x) => s + x, 0)) * 1000) / 1000;
}

export async function probeVideo(file: File): Promise<VideoInfo> {
  const video = await loadVideo(file);
  try {
    const fps = (await containerFps(file).catch(() => null)) ?? (await measureFps(video));
    // O navegador não expõe o codec; o tipo do arquivo é o que há.
    const codec = file.type.replace(/^video\//, "") || (file.name.split(".").pop() ?? "").toLowerCase();
    return { width: video.videoWidth, height: video.videoHeight, duration: video.duration, fps, codec };
  } finally {
    release(video);
  }
}

function seek(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve) => {
    video.onseeked = () => resolve();
    video.currentTime = t;
  });
}

/** Variância do Laplaciano (4 vizinhos) na miniatura que cabe em 640×640,
 *  luma Rec.709 arredondada para 8 bits — igual ao frames.rs. */
export function sharpness(rgba: Uint8ClampedArray, w: number, h: number) {
  if (w < 3 || h < 3) return 0;
  const l = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++)
    l[i] = Math.round(0.2126 * rgba[4 * i] + 0.7152 * rgba[4 * i + 1] + 0.0722 * rgba[4 * i + 2]);
  let sum = 0, sq = 0, n = 0;
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap = l[i - 1] + l[i + 1] + l[i - w] + l[i + w] - 4 * l[i];
      sum += lap;
      sq += lap * lap;
      n++;
    }
  const mean = sum / n;
  return sq / n - mean * mean;
}

export async function extractFrames(
  file: File,
  info: VideoInfo,
  opts: FrameOptions,
  imagesDir: FileSystemDirectoryHandle,
  progress: (frac: number, msg: string) => void,
): Promise<PickedFrame[]> {
  const start = Math.max(0, opts.trimStart);
  const end = Math.min(opts.trimEnd != null && opts.trimEnd > start ? opts.trimEnd : info.duration, info.duration);
  const span = end - start;
  if (span <= 0.1) throw new Error("O trecho selecionado do vídeo é vazio.");
  const wanted = opts.targetFrames * Math.max(1, opts.oversample);
  const sampleFps = Math.max(0.1, Math.min(wanted / span, info.fps));
  const total = Math.max(1, Math.round(span * sampleFps));
  const target = Math.min(Math.max(1, opts.targetFrames), total);

  const scale = Math.min(1, opts.maxSide / Math.max(info.width, info.height));
  const W = Math.round(info.width * scale), H = Math.round(info.height * scale);
  const ts = Math.min(1, 640 / Math.max(W, H));
  const tw = Math.max(1, Math.round(W * ts)), th = Math.max(1, Math.round(H * ts));
  const full = new OffscreenCanvas(W, H);
  const fctx = full.getContext("2d")!;
  const thumb = new OffscreenCanvas(tw, th);
  const tctx = thumb.getContext("2d", { willReadFrequently: true })!;
  for (const c of [fctx, tctx]) {
    c.imageSmoothingEnabled = true;
    c.imageSmoothingQuality = "high";
  }

  const windowOf = windowsOf(total, target);

  const video = await loadVideo(file);
  const frames: PickedFrame[] = [];
  let best: { score: number; blob: Blob; k: number } | null = null;
  const flush = async () => {
    if (!best) return;
    const name = frameName(frames.length);
    await writeBlob(imagesDir, name, best.blob);
    frames.push({ name, time: opts.trimStart + best.k / sampleFps, sharpness: best.score });
    best = null;
  };
  try {
    for (let k = 0; k < total; k++) {
      if (k > 0 && windowOf[k] !== windowOf[k - 1]) await flush();
      // meio quadro da fonte adiante: o instante exato cai na fronteira entre
      // dois quadros e o seek às vezes pega o anterior; assim vem o mais
      // próximo, como no filtro fps do ffmpeg
      await seek(video, Math.min(start + k / sampleFps + 0.5 / info.fps, end - 1e-3));
      tctx.drawImage(video, 0, 0, tw, th);
      const score = sharpness(tctx.getImageData(0, 0, tw, th).data, tw, th);
      // empate: fica o último, como o max_by do Rust
      if (!best || score >= best.score) {
        fctx.drawImage(video, 0, 0, W, H);
        best = { score, blob: await full.convertToBlob({ type: "image/jpeg", quality: 0.95 }), k };
      }
      progress((k + 1) / total, `Frames: candidato ${k + 1}/${total}`);
    }
    await flush();
  } finally {
    release(video);
  }
  return frames;
}

/** Frames a partir de imagens, na ordem dada: a mais nítida de cada janela,
 *  girada pelo EXIF (createImageBitmap) e reduzida para caber em maxSide. */
export async function framesFromImages(
  files: File[],
  opts: FrameOptions,
  imagesDir: FileSystemDirectoryHandle,
  progress: (frac: number) => void,
): Promise<PickedFrame[]> {
  const total = files.length;
  if (!total) throw new Error("Nenhuma imagem para usar como frame.");
  const target = Math.min(Math.max(1, opts.targetFrames), total);
  const windowOf = windowsOf(total, target);
  const thumb = new OffscreenCanvas(1, 1);
  const tctx = thumb.getContext("2d", { willReadFrequently: true })!;
  const full = new OffscreenCanvas(1, 1);
  const fctx = full.getContext("2d")!;
  let size: { w: number; h: number; from: string } | null = null;

  const frames: PickedFrame[] = [];
  let best: { score: number; bmp: ImageBitmap; k: number } | null = null;
  const flush = async () => {
    if (!best) return;
    const { bmp, k, score } = best;
    best = null;
    const s = Math.min(1, opts.maxSide / Math.max(bmp.width, bmp.height));
    const W = Math.max(1, Math.round(bmp.width * s)), H = Math.max(1, Math.round(bmp.height * s));
    // O SfM e o Brush usam uma câmera só para todos os frames.
    size ??= { w: W, h: H, from: files[k].name };
    if (W !== size.w || H !== size.h) {
      bmp.close();
      throw new Error(
        `As imagens têm tamanhos diferentes (${size.from}: ${size.w}×${size.h}, ${files[k].name}: ${W}×${H}); use imagens de uma mesma câmera.`,
      );
    }
    full.width = W;
    full.height = H;
    fctx.imageSmoothingEnabled = true;
    fctx.imageSmoothingQuality = "high";
    fctx.drawImage(bmp, 0, 0, W, H);
    bmp.close();
    const name = frameName(frames.length);
    await writeBlob(imagesDir, name, await full.convertToBlob({ type: "image/jpeg", quality: 0.95 }));
    frames.push({ name, time: 0, sharpness: score, source: files[k].name });
  };
  try {
    for (let k = 0; k < total; k++) {
      if (k > 0 && windowOf[k] !== windowOf[k - 1]) await flush();
      const bmp = await createImageBitmap(files[k]).catch(() => {
        throw new Error(`O navegador não conseguiu abrir ${files[k].name}.`);
      });
      // Miniatura que cabe em 640×640, como na extração do vídeo.
      const ts = Math.min(1, 640 / Math.max(bmp.width, bmp.height));
      const tw = Math.max(1, Math.round(bmp.width * ts)), th = Math.max(1, Math.round(bmp.height * ts));
      if (thumb.width !== tw || thumb.height !== th) {
        thumb.width = tw;
        thumb.height = th;
      }
      tctx.imageSmoothingEnabled = true;
      tctx.imageSmoothingQuality = "high";
      tctx.drawImage(bmp, 0, 0, tw, th);
      const score = sharpness(tctx.getImageData(0, 0, tw, th).data, tw, th);
      // empate: fica o último, como o max_by do Rust
      if (!best || score >= best.score) {
        best?.bmp.close();
        best = { score, bmp, k };
      } else bmp.close();
      progress((k + 1) / total);
    }
    await flush();
  } finally {
    (best as { bmp: ImageBitmap } | null)?.bmp.close();
  }
  return frames;
}
