import { useMemo, useState } from "react";
import { fileUrl, type FrameInfo } from "../lib/api";

/** Miniaturas dos frames escolhidos, com a nitidez relativa de cada um.
 *  Frames bem abaixo da mediana costumam virar borrão no splat. */
export function FramesGrid({ frames, version }: { frames: FrameInfo[]; version: number }) {
  const [open, setOpen] = useState<number | null>(null);
  const median = useMemo(() => {
    const v = frames.map((f) => f.sharpness).sort((a, b) => a - b);
    return v[Math.floor(v.length / 2)] || 1;
  }, [frames]);
  const max = useMemo(() => Math.max(1, ...frames.map((f) => f.sharpness)), [frames]);

  if (!frames.length) {
    return <div className="empty">Os frames selecionados aparecem aqui depois da primeira etapa.</div>;
  }

  const low = frames.filter((f) => f.sharpness < median * 0.5).length;

  return (
    <div className="frames">
      <div className="frames-summary">
        <span>{frames.length} frames</span>
        {low > 0 ? (
          <span className="warn">{low} com menos da metade da nitidez mediana</span>
        ) : (
          <span className="good">Nitidez uniforme</span>
        )}
      </div>
      <div className="frames-grid">
        {frames.map((f, i) => {
          const weak = f.sharpness < median * 0.5;
          return (
            <button key={f.path} className={`thumb ${weak ? "weak" : ""}`} onClick={() => setOpen(i)}>
              <img src={fileUrl(f.path, version)} loading="lazy" decoding="async" alt={`frame ${i}`} />
              <span className="thumb-meta mono">
                #{i} · {f.time.toFixed(2)} s
              </span>
              <span className="thumb-bar" style={{ width: `${(f.sharpness / max) * 100}%` }} />
            </button>
          );
        })}
      </div>
      {open != null && (
        <div className="lightbox" onClick={() => setOpen(null)}>
          <img src={fileUrl(frames[open].path, version)} alt="" />
          <span className="mono">
            #{open} · {frames[open].time.toFixed(2)} s · nitidez {frames[open].sharpness.toFixed(0)}
          </span>
        </div>
      )}
    </div>
  );
}

export function LogView({ lines }: { lines: string[] }) {
  if (!lines.length) return <div className="empty">Nenhum log ainda.</div>;
  return (
    <pre className="log">
      {lines.slice(-1500).join("\n")}
    </pre>
  );
}
