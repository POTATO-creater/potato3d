import { useEffect, useRef, useState, type RefObject } from "react";
import type { SubjectPrompt } from "../lib/api";
import { useI18n } from "../i18n";

interface Trim {
  start: number;
  end: number | null;
}

interface Props {
  src: string;
  /** Duração do ffprobe, usada até o vídeo carregar os metadados. */
  duration: number;
  fps: number;
  trim: Trim;
  onTrim: (t: Trim) => void;
  /** Bloqueia a edição do trecho (o vídeo continua navegável). */
  locked?: boolean;
  videoRef: RefObject<HTMLVideoElement | null>;
  /** Marcação do sujeito (SAM 2.1); com `onPrompt`, o player ganha o modo de marcar. */
  prompt?: SubjectPrompt | null;
  onPrompt?: (p: SubjectPrompt | null) => void;
}

const MIN_SPAN = 0.2;

const fmt = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

const Icon = {
  play: <path d="M5 3.5v9l7.5-4.5z" fill="currentColor" />,
  pause: (
    <g fill="currentColor">
      <rect x="4" y="3.5" width="2.6" height="9" rx="0.6" />
      <rect x="9.4" y="3.5" width="2.6" height="9" rx="0.6" />
    </g>
  ),
  prev: <path d="M4 3.5v9M12 3.5v9L6 8z" stroke="currentColor" strokeWidth="1.4" fill="currentColor" strokeLinejoin="round" />,
  next: <path d="M12 3.5v9M4 3.5v9L10 8z" stroke="currentColor" strokeWidth="1.4" fill="currentColor" strokeLinejoin="round" />,
  target: (
    <g fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
      <circle cx="8" cy="8" r="4.5" />
      <path d="M8 1.5v3M8 11.5v3M1.5 8h3M11.5 8h3" />
    </g>
  ),
  loop: (
    <path
      d="M3 7.5V7a3 3 0 0 1 3-3h6.5M10.5 2 12.5 4l-2 2M13 8.5V9a3 3 0 0 1-3 3H3.5M5.5 14l-2-2 2-2"
      stroke="currentColor"
      strokeWidth="1.4"
      fill="none"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  ),
};

function Svg({ children }: { children: React.ReactNode }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
      {children}
    </svg>
  );
}

/** Player compacto com linha do tempo que mostra e edita o trecho usado. */
/** Sobreposição da marcação, no retângulo que o vídeo ocupa (object-fit: contain). */
function PromptLayer({
  prompt,
  aspect,
  editing,
  onAdd,
}: {
  prompt: SubjectPrompt | null;
  aspect: number;
  editing: boolean;
  onAdd: (p: { box: [number, number, number, number] } | { point: { x: number; y: number; label: 0 | 1 } }) => void;
}) {
  const start = useRef<{ x: number; y: number } | null>(null);
  const [drag, setDrag] = useState<[number, number, number, number] | null>(null);
  const stage = 16 / 9;
  const style =
    aspect > stage
      ? { left: 0, width: "100%", top: `${((1 - stage / aspect) / 2) * 100}%`, height: `${(stage / aspect) * 100}%` }
      : { top: 0, height: "100%", left: `${((1 - aspect / stage) / 2) * 100}%`, width: `${(aspect / stage) * 100}%` };
  const at = (e: React.PointerEvent) => {
    const r = e.currentTarget.getBoundingClientRect();
    return { x: clamp((e.clientX - r.left) / r.width, 0, 1), y: clamp((e.clientY - r.top) / r.height, 0, 1) };
  };
  const rect = (a: { x: number; y: number }, b: { x: number; y: number }): [number, number, number, number] => [
    Math.min(a.x, b.x),
    Math.min(a.y, b.y),
    Math.max(a.x, b.x),
    Math.max(a.y, b.y),
  ];
  const box = drag ?? prompt?.box;
  return (
    <svg
      className={`prompt-layer ${editing ? "editing" : ""}`}
      style={style}
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      onClick={(e) => editing && e.stopPropagation()}
      onContextMenu={(e) => editing && e.preventDefault()}
      onPointerDown={(e) => {
        if (!editing) return;
        e.stopPropagation();
        e.currentTarget.setPointerCapture(e.pointerId);
        start.current = at(e);
      }}
      onPointerMove={(e) => {
        const s = start.current;
        if (!s) return;
        const p = at(e);
        if (drag || Math.hypot(p.x - s.x, p.y - s.y) > 0.02) setDrag(rect(s, p));
      }}
      onPointerUp={(e) => {
        const s = start.current;
        start.current = null;
        if (!s) return;
        const p = at(e);
        if (drag) {
          setDrag(null);
          const b = rect(s, p);
          if (b[2] - b[0] > 0.01 && b[3] - b[1] > 0.01) onAdd({ box: b });
        } else {
          onAdd({ point: { ...p, label: e.altKey || e.button === 2 ? 0 : 1 } });
        }
      }}
    >
      {box && (
        <rect
          className="prompt-box"
          x={box[0]}
          y={box[1]}
          width={box[2] - box[0]}
          height={box[3] - box[1]}
          vectorEffect="non-scaling-stroke"
        />
      )}
      {prompt?.points.map((p, i) => (
        <ellipse
          key={i}
          className={`prompt-point ${p.label ? "pos" : "neg"}`}
          cx={p.x}
          cy={p.y}
          rx={0.025}
          ry={0.025 * aspect}
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
}

export function VideoPlayer({ src, duration: probed, fps, trim, onTrim, locked, videoRef, prompt, onPrompt }: Props) {
  const { t } = useI18n();
  const tp = t.player;
  const timeline = useRef<HTMLDivElement>(null);
  const drag = useRef<"seek" | "start" | "end" | null>(null);
  const [duration, setDuration] = useState(probed);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loopTrim, setLoopTrim] = useState(true);
  const [marking, setMarking] = useState(false);
  const [aspect, setAspect] = useState(16 / 9);
  const editing = marking && !!onPrompt;
  // A marcação vale para um frame: aparece nele (e enquanto se marca).
  const promptHere = !!prompt && Math.abs(time - (prompt.time ?? 0)) < 1.5 / (fps || 30);

  const end = trim.end ?? duration;
  const pct = (t: number) => `${(clamp(t / (duration || 1), 0, 1) * 100).toFixed(3)}%`;

  useEffect(() => setDuration(probed), [probed, src]);

  // Playhead suave (timeupdate só dispara ~4x/s) e repetição do trecho.
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const v = videoRef.current;
      if (v) {
        if (loopTrim && !drag.current && v.currentTime >= end - 0.02) v.currentTime = trim.start;
        setTime(v.currentTime);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, loopTrim, end, trim.start, videoRef]);

  function seek(t: number) {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = clamp(t, 0, duration);
    setTime(v.currentTime);
  }

  function toggle() {
    const v = videoRef.current;
    if (!v || editing) return;
    if (v.paused) {
      if (loopTrim && (v.currentTime < trim.start || v.currentTime >= end - 0.02)) seek(trim.start);
      v.play();
    } else {
      v.pause();
    }
  }

  const step = (dir: 1 | -1) => {
    videoRef.current?.pause();
    seek((videoRef.current?.currentTime ?? 0) + dir / (fps || 30));
  };

  function timeAt(clientX: number) {
    const r = timeline.current!.getBoundingClientRect();
    return clamp((clientX - r.left) / r.width, 0, 1) * duration;
  }

  function onPointerDown(e: React.PointerEvent) {
    const handle = (e.target as HTMLElement).dataset.handle as "start" | "end" | undefined;
    drag.current = handle && !locked ? handle : "seek";
    timeline.current!.setPointerCapture(e.pointerId);
    onPointerMove(e);
  }

  function onPointerMove(e: React.PointerEvent) {
    const mode = drag.current;
    if (!mode) return;
    const t = timeAt(e.clientX);
    if (mode === "seek") {
      seek(t);
    } else if (mode === "start") {
      const s = clamp(t, 0, end - MIN_SPAN);
      onTrim({ ...trim, start: s });
      seek(s);
    } else {
      const en = clamp(t, trim.start + MIN_SPAN, duration);
      onTrim({ ...trim, end: en >= duration - 0.05 ? null : en });
      seek(en);
    }
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === " ") toggle();
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "ArrowRight") step(1);
    else return;
    e.preventDefault();
  }

  return (
    <div className="player" tabIndex={0} onKeyDown={onKeyDown}>
      <div className="player-stage" onClick={toggle}>
        <video
          ref={videoRef}
          src={src}
          muted
          playsInline
          preload="auto"
          onLoadedMetadata={(e) => {
            const v = e.currentTarget;
            if (Number.isFinite(v.duration)) setDuration(v.duration);
            if (v.videoWidth && v.videoHeight) setAspect(v.videoWidth / v.videoHeight);
          }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onSeeked={(e) => setTime(e.currentTarget.currentTime)}
        />
        {(editing || (promptHere && !playing)) && (
          <PromptLayer
            prompt={editing && !promptHere ? null : (prompt ?? null)}
            aspect={aspect}
            editing={editing}
            onAdd={(a) => {
              const t = videoRef.current?.currentTime ?? time;
              // Marcação em outro frame recomeça do zero.
              const base: SubjectPrompt = promptHere && prompt ? prompt : { time: t, box: null, points: [] };
              onPrompt!("box" in a ? { ...base, box: a.box } : { ...base, points: [...base.points, a.point] });
            }}
          />
        )}
        {!playing && !editing && (
          <span className="player-overlay" aria-hidden>
            <Svg>{Icon.play}</Svg>
          </span>
        )}
      </div>

      <div
        className={`timeline ${locked ? "locked" : ""}`}
        ref={timeline}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={() => (drag.current = null)}
        onPointerCancel={() => (drag.current = null)}
      >
        <div className="tl-track" />
        <div className="tl-range" style={{ left: pct(trim.start), width: `calc(${pct(end)} - ${pct(trim.start)})` }} />
        <div className="tl-handle" data-handle="start" style={{ left: pct(trim.start) }} title={tp.dragStart} />
        <div className="tl-handle" data-handle="end" style={{ left: pct(end) }} title={tp.dragEnd} />
        <div className="tl-head" style={{ left: pct(time) }} />
      </div>

      <div className="player-bar">
        <button className="pbtn" onClick={toggle} title={playing ? tp.pause : tp.play}>
          <Svg>{playing ? Icon.pause : Icon.play}</Svg>
        </button>
        <button className="pbtn" onClick={() => step(-1)} title={tp.prevFrame}>
          <Svg>{Icon.prev}</Svg>
        </button>
        <button className="pbtn" onClick={() => step(1)} title={tp.nextFrame}>
          <Svg>{Icon.next}</Svg>
        </button>
        <span className="player-time mono">
          {fmt(time)} <span className="muted">/ {fmt(duration)}</span>
        </span>
        {onPrompt && (
          <button
            className={`pbtn ${editing ? "on" : ""}`}
            onClick={() => {
              videoRef.current?.pause();
              setMarking((m) => !m);
            }}
            title={editing ? tp.markDone : `${tp.mark} · ${tp.markHint}`}
          >
            <Svg>{Icon.target}</Svg>
          </button>
        )}
        <button
          className={`pbtn ${loopTrim ? "on" : ""}`}
          onClick={() => setLoopTrim((v) => !v)}
          title={tp.loop}
        >
          <Svg>{Icon.loop}</Svg>
        </button>
      </div>
    </div>
  );
}
