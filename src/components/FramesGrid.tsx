import { useEffect, useMemo, useState } from "react";
import { fileUrl, type FrameInfo } from "../lib/api";
import { useI18n } from "../i18n";

const stemOf = (p: string) => (p.split(/[\\/]/).pop() ?? p).replace(/\.[^.]+$/, "");
/** De onde o frame veio: o instante no vídeo ou a imagem de origem. */
const origin = (f: FrameInfo) => f.source ?? `${f.time.toFixed(2)} s`;

/** Recorte por CSS (luminância da máscara), no mesmo enquadramento da <img>
 *  (object-fit: cover, centralizada). */
function frameMaskStyle(f: FrameInfo, masksDir: { path: string; version: number } | null) {
  if (!masksDir) return undefined;
  const url = `url("${fileUrl(`${masksDir.path}/${stemOf(f.path)}.png`, masksDir.version)}")`;
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
}

/** Um frame em tela cheia, com setas (e ← → no teclado) para os vizinhos.
 *  `masksDir` presente mostra o recorte. */
export function FrameLightbox({
  frames,
  index,
  onIndex,
  onClose,
  version,
  masksDir,
}: {
  frames: FrameInfo[];
  index: number;
  onIndex: (i: number) => void;
  onClose: () => void;
  version: number;
  masksDir: { path: string; version: number } | null;
}) {
  const { t } = useI18n();
  const tf = t.frames;
  const last = frames.length - 1;
  const go = (i: number) => onIndex(Math.max(0, Math.min(last, i)));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const to: Record<string, number> = { ArrowLeft: index - 1, ArrowRight: index + 1, Home: 0, End: last };
      if (e.key === "Escape") onClose();
      else if (e.key in to) go(to[e.key]);
      else return;
      // Captura na janela: o player de vídeo também usa as setas.
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  });

  // Os frames são grandes: adianta os vizinhos para a troca ser imediata.
  useEffect(() => {
    for (const i of [index - 1, index + 1]) {
      const f = frames[i];
      if (!f) continue;
      new Image().src = fileUrl(f.path, version);
      if (masksDir) new Image().src = fileUrl(`${masksDir.path}/${stemOf(f.path)}.png`, masksDir.version);
    }
  }, [index, frames, version, masksDir]);

  const f = frames[index];
  if (!f) return null;
  const stop = (fn: () => void) => (e: React.MouseEvent) => {
    e.stopPropagation();
    fn();
  };
  return (
    <div className={`lightbox ${masksDir ? "cut" : ""}`} onClick={onClose} role="dialog" aria-modal="true">
      <img src={fileUrl(f.path, version)} alt={`frame ${index}`} style={frameMaskStyle(f, masksDir)} />
      <span className="mono">
        #{index} · {index + 1}/{frames.length} · {origin(f)} · {tf.sharpness} {f.sharpness.toFixed(0)}
      </span>
      <button className="lightbox-nav prev" onClick={stop(() => go(index - 1))} disabled={index === 0} aria-label={tf.prev} title={`${tf.prev} (←)`}>
        ‹
      </button>
      <button className="lightbox-nav next" onClick={stop(() => go(index + 1))} disabled={index === last} aria-label={tf.next} title={`${tf.next} (→)`}>
        ›
      </button>
      <button className="lightbox-close" onClick={stop(onClose)} aria-label={tf.close} title={`${tf.close} (Esc)`}>
        ×
      </button>
    </div>
  );
}

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
              <img src={fileUrl(f.path, version)} loading="lazy" decoding="async" alt={`frame ${i}`} style={cut ? frameMaskStyle(f, masksDir) : undefined} />
              <span className="thumb-meta mono">
                #{i} · {origin(f)}
              </span>
              <span className="thumb-bar" style={{ width: `${(f.sharpness / max) * 100}%` }} />
            </button>
          );
        })}
      </div>
      {open != null && (
        <FrameLightbox
          frames={frames}
          index={open}
          onIndex={setOpen}
          onClose={() => setOpen(null)}
          version={version}
          masksDir={cut ? masksDir : null}
        />
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
