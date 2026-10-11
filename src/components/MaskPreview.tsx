import { useEffect, useRef, useState } from "react";
import { host, type MaskOptions } from "../lib/api";
import { useI18n } from "../i18n";

/** De onde vem o quadro da prévia. */
export type PreviewSource = { kind: "video"; src: string; time: number } | { kind: "image"; src: string };

const HIDDEN_KEY = "splatforge.maskPreview.v1";
/** Lado maior do quadro da prévia: o SAM e o RMBG trabalham em 1024². */
const MAX_SIDE = 1024;

const fmtTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

function loadHidden() {
  try {
    return localStorage.getItem(HIDDEN_KEY) === "hidden";
  } catch {
    return false;
  }
}

function fit(w: number, h: number) {
  const k = Math.min(1, MAX_SIDE / Math.max(w, h));
  return { resizeWidth: Math.round(w * k), resizeHeight: Math.round(h * k), resizeQuality: "high" as const };
}

/** Quadro do vídeo num instante, por um <video> próprio (sem mexer no player). */
async function videoFrame(video: HTMLVideoElement, src: string, time: number): Promise<ImageBitmap> {
  if (video.getAttribute("src") !== src) {
    video.src = src;
    await new Promise<void>((ok, fail) => {
      video.onloadeddata = () => ok();
      video.onerror = () => fail(new Error("vídeo ilegível"));
    });
  }
  await new Promise<void>((ok) => {
    video.onseeked = () => ok();
    video.currentTime = Math.min(time, Math.max(video.duration - 0.05, 0));
  });
  return createImageBitmap(video, fit(video.videoWidth, video.videoHeight));
}

async function imageFrame(src: string): Promise<ImageBitmap> {
  const blob = await (await fetch(src)).blob();
  const full = await createImageBitmap(blob);
  const bmp = await createImageBitmap(full, fit(full.width, full.height));
  full.close();
  return bmp;
}

/** Quadro com o fundo removido (xadrez) e a marcação do SAM 2.1 por cima. */
function draw(canvas: HTMLCanvasElement, frame: ImageBitmap, mask: Uint8Array, masks: MaskOptions) {
  const { width: w, height: h } = frame;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(frame, 0, 0);
  const img = ctx.getImageData(0, 0, w, h);
  for (let i = 0; i < mask.length; i++) img.data[4 * i + 3] = mask[i];
  ctx.putImageData(img, 0, 0);
  const p = masks.method === "sam2" ? masks.prompt : null;
  if (!p) return;
  const lw = Math.max(2, w / 300);
  ctx.lineWidth = lw;
  if (p.box) {
    ctx.strokeStyle = "#ff9a4d";
    ctx.strokeRect(p.box[0] * w, p.box[1] * h, (p.box[2] - p.box[0]) * w, (p.box[3] - p.box[1]) * h);
  }
  for (const pt of p.points) {
    ctx.beginPath();
    ctx.arc(pt.x * w, pt.y * h, lw * 3, 0, 2 * Math.PI);
    ctx.fillStyle = pt.label ? "#4cc38a" : "#ff6b6b";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.stroke();
  }
}

/** Prévia do recorte num quadro (só onde o host oferece: o web, por enquanto). */
export function MaskPreview({ source, masks, locked }: { source: PreviewSource | null; masks: MaskOptions; locked: boolean }) {
  const { t } = useI18n();
  const tp = t.masks.preview;
  const previewer = host.maskPreview;
  const [hidden, setHidden] = useState(loadHidden);
  const [frame, setFrame] = useState<{ key: string; bmp: ImageBitmap } | null>(null);
  const [ready, setReady] = useState<boolean | null>(null);
  const [status, setStatus] = useState<{ kind: "idle" | "working" | "error"; text?: string }>({ kind: "idle" });
  const [shown, setShown] = useState(false);
  const canvas = useRef<HTMLCanvasElement>(null);
  const video = useRef<HTMLVideoElement | null>(null);
  const request = useRef(0);

  const src = source?.src;
  const time = source?.kind === "video" ? source.time : null;
  const active = !!previewer && !!source && !hidden && !locked;

  useEffect(() => {
    try {
      localStorage.setItem(HIDDEN_KEY, hidden ? "hidden" : "shown");
    } catch {
      /* armazenamento indisponível */
    }
  }, [hidden]);

  // Modelo já baixado? Sem ele a prévia só roda quando o usuário pede.
  useEffect(() => {
    if (!previewer) return;
    let live = true;
    setReady(null);
    previewer.ready(masks.method).then((r) => live && setReady(r));
    return () => {
      live = false;
    };
  }, [previewer, masks.method]);

  // Quadro: o marcado (ou o início do trecho); espera o instante parar de mudar.
  useEffect(() => {
    if (!active || !src) return;
    let live = true;
    const id = setTimeout(async () => {
      try {
        video.current ??= Object.assign(document.createElement("video"), { muted: true, preload: "auto" });
        const bmp = time == null ? await imageFrame(src) : await videoFrame(video.current, src, time);
        if (!live) return bmp.close();
        setFrame((old) => {
          old?.bmp.close();
          return { key: `${src}#${time ?? ""}`, bmp };
        });
      } catch (e) {
        if (live) setStatus({ kind: "error", text: tp.failed(String(e)) });
      }
    }, 250);
    return () => {
      live = false;
      clearTimeout(id);
    };
  }, [active, src, time, tp]);

  // Máscara: a cada mudança de método, marcação ou ajuste (só a mais recente vale).
  const promptJson = JSON.stringify(masks.method === "sam2" ? masks.prompt : null);
  const chromaJson = JSON.stringify(masks.method === "chroma" ? masks.chroma : null);
  useEffect(() => {
    if (!active || !frame || !ready || !previewer) return;
    const id = ++request.current;
    const timer = setTimeout(async () => {
      setStatus({ kind: "working", text: tp.working });
      try {
        const mask = await previewer.mask(frame.key, frame.bmp, masks, (d, n) => {
          if (id === request.current)
            setStatus({ kind: "working", text: tp.downloading(Math.round(d / 1e6), Math.round(n / 1e6)) });
        });
        if (id !== request.current || !canvas.current) return;
        draw(canvas.current, frame.bmp, mask, masks);
        setShown(true);
        setStatus({ kind: "idle" });
      } catch (e) {
        if (id === request.current) setStatus({ kind: "error", text: tp.failed(String(e)) });
      }
    }, 80);
    return () => clearTimeout(timer);
    // masks entra pelos campos que mudam o resultado
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, frame, ready, previewer, masks.method, promptJson, chromaJson, tp]);

  if (!previewer || !source) return null;
  return (
    <div className="mask-preview">
      <div className="mask-preview-head">
        <span className="mask-preview-title">{tp.title}</span>
        {time != null && !hidden && <span className="muted mono">{tp.at(fmtTime(time))}</span>}
        <button className="link" onClick={() => setHidden((h) => !h)}>
          {hidden ? tp.show : tp.hide}
        </button>
      </div>
      {!hidden && (
        <>
          {ready === false ? (
            <button className="btn" disabled={locked} onClick={() => setReady(true)}>
              {tp.load}
            </button>
          ) : (
            <div className={`mask-preview-stage ${shown ? "" : "empty"}`}>
              <canvas ref={canvas} />
            </div>
          )}
          <p className={`help ${status.kind === "error" ? "error" : "muted"}`}>
            {locked ? tp.paused : (status.text ?? (shown ? tp.legend : ""))}
          </p>
        </>
      )}
    </div>
  );
}
