import { useEffect, useRef, useState } from "react";
import { fileUrl } from "../lib/api";
import { basename } from "../lib/platform";
import { useI18n } from "../i18n";

/** Velocidade da prévia animada da órbita. */
const PLAY_FPS = 12;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function Svg({ children }: { children: React.ReactNode }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
      {children}
    </svg>
  );
}

/** Prévia das imagens usadas como frames, na ordem em que entram (a da
 *  órbita), com a mesma cara do player de vídeo. */
export function ImageSequence({ paths }: { paths: string[] }) {
  const { t } = useI18n();
  const tp = t.player;
  const timeline = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const n = paths.length;
  const at = clamp(i, 0, n - 1);

  useEffect(() => {
    setI(0);
    setPlaying(false);
  }, [paths]);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setI((v) => (v + 1) % n), 1000 / PLAY_FPS);
    return () => clearInterval(id);
  }, [playing, n]);

  const step = (dir: 1 | -1) => {
    setPlaying(false);
    setI((v) => clamp(v + dir, 0, n - 1));
  };

  function scrub(clientX: number) {
    const r = timeline.current!.getBoundingClientRect();
    setI(Math.round(clamp((clientX - r.left) / r.width, 0, 1) * (n - 1)));
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === " ") setPlaying((v) => !v);
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "ArrowRight") step(1);
    else return;
    e.preventDefault();
  }

  const pct = `${((n > 1 ? at / (n - 1) : 0) * 100).toFixed(3)}%`;

  return (
    <div className="player" tabIndex={0} onKeyDown={onKeyDown}>
      <div className="player-stage" onClick={() => setPlaying((v) => !v)}>
        <img src={fileUrl(paths[at])} alt={basename(paths[at])} decoding="async" />
        {!playing && (
          <span className="player-overlay" aria-hidden>
            <Svg>
              <path d="M5 3.5v9l7.5-4.5z" fill="currentColor" />
            </Svg>
          </span>
        )}
      </div>

      <div
        className="timeline"
        ref={timeline}
        onPointerDown={(e) => {
          dragging.current = true;
          setPlaying(false);
          timeline.current!.setPointerCapture(e.pointerId);
          scrub(e.clientX);
        }}
        onPointerMove={(e) => dragging.current && scrub(e.clientX)}
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
      >
        <div className="tl-track" />
        <div className="tl-head" style={{ left: pct }} />
      </div>

      <div className="player-bar">
        <button className="pbtn" onClick={() => setPlaying((v) => !v)} title={playing ? tp.pause : tp.play}>
          <Svg>
            {playing ? (
              <g fill="currentColor">
                <rect x="4" y="3.5" width="2.6" height="9" rx="0.6" />
                <rect x="9.4" y="3.5" width="2.6" height="9" rx="0.6" />
              </g>
            ) : (
              <path d="M5 3.5v9l7.5-4.5z" fill="currentColor" />
            )}
          </Svg>
        </button>
        <button className="pbtn" onClick={() => step(-1)} title={tp.prevImage}>
          <Svg>
            <path d="M4 3.5v9M12 3.5v9L6 8z" stroke="currentColor" strokeWidth="1.4" fill="currentColor" strokeLinejoin="round" />
          </Svg>
        </button>
        <button className="pbtn" onClick={() => step(1)} title={tp.nextImage}>
          <Svg>
            <path d="M12 3.5v9M4 3.5v9L10 8z" stroke="currentColor" strokeWidth="1.4" fill="currentColor" strokeLinejoin="round" />
          </Svg>
        </button>
        <span className="player-time mono">
          {at + 1} <span className="muted">/ {n}</span>
        </span>
        <span className="player-name muted" title={paths[at]}>
          {basename(paths[at])}
        </span>
      </div>
    </div>
  );
}
