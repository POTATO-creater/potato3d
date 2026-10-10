import { useEffect, useRef, useState, type RefObject } from "react";
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
export function VideoPlayer({ src, duration: probed, fps, trim, onTrim, locked, videoRef }: Props) {
  const { t } = useI18n();
  const tp = t.player;
  const timeline = useRef<HTMLDivElement>(null);
  const drag = useRef<"seek" | "start" | "end" | null>(null);
  const [duration, setDuration] = useState(probed);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loopTrim, setLoopTrim] = useState(true);

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
    if (!v) return;
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
          onLoadedMetadata={(e) => Number.isFinite(e.currentTarget.duration) && setDuration(e.currentTarget.duration)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onSeeked={(e) => setTime(e.currentTarget.currentTime)}
        />
        {!playing && (
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
