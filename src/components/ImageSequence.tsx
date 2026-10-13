import { useEffect, useRef, useState } from "react";
import { fileUrl, type SubjectPrompt } from "../lib/api";
import { basename } from "../lib/platform";
import { withMarks } from "../lib/prompt";
import { useI18n } from "../i18n";
import { PromptLayer } from "./PromptLayer";

/** Speed of the animated orbit preview. */
const PLAY_FPS = 12;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function Svg({ children }: { children: React.ReactNode }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
      {children}
    </svg>
  );
}

/** Preview of the images used as frames, in the order they are used (orbit
 *  order), styled like the video player. With `onPrompt`, marks the subject
 *  for SAM 2.1 on one of the images, as with video. */
export function ImageSequence({
  paths,
  prompt,
  onPrompt,
}: {
  paths: string[];
  prompt?: SubjectPrompt | null;
  onPrompt?: (p: SubjectPrompt | null) => void;
}) {
  const { t } = useI18n();
  const tp = t.player;
  const timeline = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [marking, setMarking] = useState(false);
  const [aspect, setAspect] = useState(16 / 9);
  const n = paths.length;
  const at = clamp(i, 0, n - 1);
  const name = basename(paths[at]);
  const editing = marking && !!onPrompt;
  const promptHere = !!prompt?.image && prompt.image === name;

  useEffect(() => {
    setI(0);
    setPlaying(false);
    setMarking(false);
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
      <div className="player-stage" onClick={() => !editing && setPlaying((v) => !v)}>
        <img
          src={fileUrl(paths[at])}
          alt={name}
          decoding="async"
          onLoad={(e) => e.currentTarget.naturalHeight && setAspect(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)}
        />
        {(editing || (promptHere && !playing)) && (
          <PromptLayer
            marks={promptHere ? (prompt ?? null) : null}
            aspect={aspect}
            editing={editing}
            onChange={(m) => {
              // Marking on another image starts from scratch.
              const base: SubjectPrompt = promptHere && prompt ? prompt : { time: null, image: name, box: null, points: [] };
              onPrompt!(withMarks(base, m));
            }}
          />
        )}
        {!playing && !editing && (
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
          {name}
        </span>
        {onPrompt && (
          <button
            className={`pbtn ${marking ? "on" : ""}`}
            onClick={() => {
              setPlaying(false);
              setMarking((m) => !m);
            }}
            title={marking ? tp.markDone : `${tp.mark} · ${tp.markHint}`}
          >
            <Svg>
              <g fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
                <circle cx="8" cy="8" r="4.5" />
                <path d="M8 1.5v3M8 11.5v3M1.5 8h3M11.5 8h3" />
              </g>
            </Svg>
          </button>
        )}
      </div>
    </div>
  );
}
