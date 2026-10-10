// Vídeo → frames no navegador, com a mesma lógica de src-tauri/src/frames.rs:
// candidatos uniformes no trecho (targetFrames × oversample, no máximo o fps
// do vídeo), escala que cabe em maxSide × maxSide sem ampliar, e o candidato
// mais nítido (variância do Laplaciano) de cada janela vira 00000.jpg, …
// O próprio navegador decodifica (um seek por candidato).

export interface FrameOptions {
  targetFrames: number;
  maxSide: number;
  oversample: number;
  trimStart: number;
  trimEnd: number | null;
}

export interface VideoInfo {
  width: number;
  height: number;
  duration: number;
  fps: number;
}

export interface FrameInfo {
  name: string;
  time: number;
  sharpness: number;
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

/** fps pelo intervalo entre quadros apresentados (requestVideoFrameCallback). */
async function measureFps(video: HTMLVideoElement): Promise<number> {
  if (!("requestVideoFrameCallback" in video)) return 30;
  const times: number[] = [];
  const done = new Promise<void>((resolve) => {
    const tick = (_: number, meta: VideoFrameCallbackMetadata) => {
      times.push(meta.mediaTime);
      if (times.length >= 12) resolve();
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
    .filter((x) => x > 1e-4)
    .sort((a, b) => a - b);
  const med = d[d.length >> 1];
  return med ? Math.round((1 / med) * 1000) / 1000 : 30;
}

export async function probeVideo(file: File): Promise<VideoInfo> {
  const video = await loadVideo(file);
  try {
    const fps = await measureFps(video);
    return { width: video.videoWidth, height: video.videoHeight, duration: video.duration, fps };
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
): Promise<FrameInfo[]> {
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

  // janela de cada candidato: lo(w) = ⌊w·total/target⌋
  const windowOf = new Int32Array(total);
  for (let w = 0; w < target; w++) {
    const lo = Math.floor((w * total) / target);
    const hi = Math.max(Math.floor(((w + 1) * total) / target), lo + 1);
    for (let k = lo; k < Math.min(hi, total); k++) windowOf[k] = w;
  }

  const video = await loadVideo(file);
  const frames: FrameInfo[] = [];
  let best: { score: number; blob: Blob; k: number } | null = null;
  const flush = async () => {
    if (!best) return;
    const name = `${String(frames.length).padStart(5, "0")}.jpg`;
    const w = await (await imagesDir.getFileHandle(name, { create: true })).createWritable();
    await w.write(best.blob);
    await w.close();
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
