import { useMemo, useState } from "react";
import { fileUrl, type FrameInfo } from "../lib/api";
import { useI18n } from "../i18n";

const stemOf = (p: string) => (p.split(/[\\/]/).pop() ?? p).replace(/\.[^.]+$/, "");
/** De onde o frame veio: o instante no vídeo ou a imagem de origem. */
const origin = (f: FrameInfo) => f.source ?? `${f.time.toFixed(2)} s`;

/** Miniaturas dos frames escolhidos, com a nitidez relativa de cada um.
 *  Frames bem abaixo da mediana costumam virar borrão no splat. Com recorte,
 *  a máscara é aplicada por CSS (luminância) sobre um xadrez. */
export function FramesGrid({
  frames,
  version,
  masksDir,
}: {
  frames: FrameInfo[];
  version: number;
  masksDir: { path: string; version: number } | null;
}) {
  const { t } = useI18n();
  const tf = t.frames;
  const [open, setOpen] = useState<number | null>(null);
  const [showCut, setShowCut] = useState(true);
  const cut = showCut && !!masksDir;
  const maskStyle = (f: FrameInfo): React.CSSProperties | undefined => {
    if (!cut || !masksDir) return undefined;
    const url = `url("${fileUrl(`${masksDir.path}/${stemOf(f.path)}.png`, masksDir.version)}")`;
    // Mesmo enquadramento da <img> (object-fit: cover, centralizada).
    return {
      maskImage: url,
      WebkitMaskImage: url,
      maskMode: "luminance",
      maskSize: "cover",
      WebkitMaskSize: "cover",
      maskPosition: "center",
      WebkitMaskPosition: "center",
      maskRepeat: "no-repeat",
      WebkitMaskRepeat: "no-repeat",
    } as React.CSSProperties;
  };
  const median = useMemo(() => {
    const v = frames.map((f) => f.sharpness).sort((a, b) => a - b);
    return v[Math.floor(v.length / 2)] || 1;
  }, [frames]);
  const max = useMemo(() => Math.max(1, ...frames.map((f) => f.sharpness)), [frames]);

  if (!frames.length) {
    return <div className="empty">{tf.empty}</div>;
  }

  const low = frames.filter((f) => f.sharpness < median * 0.5).length;

  return (
    <div className="frames">
      <div className="frames-summary">
        <span>{tf.count(frames.length)}</span>
        {low > 0 ? <span className="warn">{tf.lowSharpness(low)}</span> : <span className="good">{tf.uniform}</span>}
        {masksDir && (
          <button className={`btn-sm ${showCut ? "" : "ghost"}`} onClick={() => setShowCut((v) => !v)}>
            {showCut ? tf.showingCut : tf.showCut}
          </button>
        )}
      </div>
      <div className="frames-grid">
        {frames.map((f, i) => {
          const weak = f.sharpness < median * 0.5;
          return (
            <button key={f.path} className={`thumb ${weak ? "weak" : ""} ${cut ? "cut" : ""}`} onClick={() => setOpen(i)}>
              <img src={fileUrl(f.path, version)} loading="lazy" decoding="async" alt={`frame ${i}`} style={maskStyle(f)} />
              <span className="thumb-meta mono">
                #{i} · {origin(f)}
              </span>
              <span className="thumb-bar" style={{ width: `${(f.sharpness / max) * 100}%` }} />
            </button>
          );
        })}
      </div>
      {open != null && (
        <div className={`lightbox ${cut ? "cut" : ""}`} onClick={() => setOpen(null)}>
          <img src={fileUrl(frames[open].path, version)} alt="" style={maskStyle(frames[open])} />
          <span className="mono">
            #{open} · {origin(frames[open])} · {tf.sharpness} {frames[open].sharpness.toFixed(0)}
          </span>
        </div>
      )}
    </div>
  );
}

export function LogView({ lines }: { lines: string[] }) {
  const { t } = useI18n();
  if (!lines.length) return <div className="empty">{t.frames.noLog}</div>;
  return (
    <pre className="log">
      {lines.slice(-1500).join("\n")}
    </pre>
  );
}
