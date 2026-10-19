import { useEffect, useRef, useState } from "react";
import { host, type MaskOptions } from "../lib/api";
import { useI18n } from "../i18n";

/** Where the preview frame comes from. */
export type PreviewSource = { kind: "video"; src: string; time: number } | { kind: "image"; src: string; name: string };

const HIDDEN_KEY = "splatforge.maskPreview.v1";
/** Long side of the preview frame: SAM and RMBG work at 1024². */
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

/** Video frame at a given time, via a dedicated <video> (leaves the player alone). */
async function videoFrame(video: HTMLVideoElement, src: string, time: number): Promise<ImageBitmap> {
  if (video.getAttribute("src") !== src) {
    video.src = src;
    await new Promise<void>((ok, fail) => {
      video.onloadeddata = () => ok();
      video.onerror = () => fail(new Error("unreadable video"));
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

/** Frame with the background removed (checkerboard) and the SAM 2.1 marking on top. */
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

/** Mask preview on one frame (only where the host provides it: web, for now). */
export function MaskPreview({
  source,
  masks,
  locked,
  external,
}: {
  source: PreviewSource | null;
  masks: MaskOptions;
  locked: boolean;
  /** Method "external": the project and the attached mask (its id refreshes the preview). */
  external?: { projectDir: string; id: string } | null;
}) {
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
  const image = source?.kind === "image" ? source.name : null;
  const active = !!previewer && !!source && !hidden && !locked;

  useEffect(() => {
    try {
      localStorage.setItem(HIDDEN_KEY, hidden ? "hidden" : "shown");
    } catch {
      /* storage unavailable */
    }
  }, [hidden]);

  // Model already downloaded? Without it the preview only runs when the user asks.
  useEffect(() => {
    if (!previewer) return;
    let live = true;
    setReady(null);
    previewer.ready(masks.method).then((r) => live && setReady(r));
    return () => {
      live = false;
    };
  }, [previewer, masks.method]);

  // Frame: the marked one (or the trim start); waits for the time to settle.
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

  // Mask: on every change of method, marking or setting (only the latest counts).
  const promptJson = JSON.stringify(masks.method === "sam2" ? masks.prompt : null);
  const chromaJson = JSON.stringify(masks.method === "chroma" ? masks.chroma : null);
  useEffect(() => {
    if (!active || !frame || !ready || !previewer) return;
    const id = ++request.current;
    const timer = setTimeout(async () => {
      setStatus({ kind: "working", text: tp.working });
      try {
        const at = masks.method === "external" && external ? { projectDir: external.projectDir, ...(time != null ? { time } : { image: image ?? "" }) } : undefined;
        const mask = await previewer.mask(
          frame.key,
          frame.bmp,
          masks,
          (d, n) => {
            if (id === request.current)
              setStatus({ kind: "working", text: tp.downloading(Math.round(d / 1e6), Math.round(n / 1e6)) });
          },
          at,
        );
        if (id !== request.current || !canvas.current) return;
        draw(canvas.current, frame.bmp, mask, masks);
        setShown(true);
        setStatus({ kind: "idle" });
      } catch (e) {
        if (id === request.current) setStatus({ kind: "error", text: tp.failed(String(e)) });
      }
    }, 80);
    return () => clearTimeout(timer);
    // masks is tracked via the fields that change the result
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, frame, ready, previewer, masks.method, promptJson, chromaJson, tp, external?.projectDir, external?.id]);

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
