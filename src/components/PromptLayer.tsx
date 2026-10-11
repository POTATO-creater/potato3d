import { useEffect, useRef, useState } from "react";
import { eraseAt, type PromptMarks } from "../lib/prompt";

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/**
 * Camada de marcação do sujeito (SAM 2.1) sobre o quadro, no retângulo que a
 * imagem ocupa no palco 16:9 (object-fit: contain). Arrastar: caixa · clique:
 * ponto do sujeito · Alt+clique ou botão direito: fundo · Shift+clique: apaga.
 */
export function PromptLayer({
  marks,
  aspect,
  editing,
  onChange,
}: {
  marks: PromptMarks | null;
  /** largura / altura do quadro */
  aspect: number;
  editing: boolean;
  onChange: (m: PromptMarks) => void;
}) {
  const start = useRef<{ x: number; y: number } | null>(null);
  const [drag, setDrag] = useState<[number, number, number, number] | null>(null);
  const [erasing, setErasing] = useState(false);
  // O cursor de apagar acompanha o Shift mesmo sem mexer o mouse.
  useEffect(() => {
    if (!editing) return;
    const onKey = (e: KeyboardEvent) => setErasing(e.shiftKey);
    const onBlur = () => setErasing(false);
    addEventListener("keydown", onKey);
    addEventListener("keyup", onKey);
    addEventListener("blur", onBlur);
    return () => {
      removeEventListener("keydown", onKey);
      removeEventListener("keyup", onKey);
      removeEventListener("blur", onBlur);
    };
  }, [editing]);
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
  const cur: PromptMarks = marks ?? { box: null, points: [] };
  const box = drag ?? cur.box;
  return (
    <svg
      className={`prompt-layer ${editing ? "editing" : ""} ${editing && erasing ? "erasing" : ""}`}
      style={style}
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      onClick={(e) => editing && e.stopPropagation()}
      onContextMenu={(e) => editing && e.preventDefault()}
      onPointerEnter={(e) => setErasing(e.shiftKey)}
      onPointerDown={(e) => {
        if (!editing) return;
        e.stopPropagation();
        e.currentTarget.setPointerCapture(e.pointerId);
        start.current = at(e);
      }}
      onPointerMove={(e) => {
        setErasing(e.shiftKey);
        const s = start.current;
        if (!s || e.shiftKey) return;
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
          if (b[2] - b[0] > 0.01 && b[3] - b[1] > 0.01) onChange({ ...cur, box: b });
        } else if (e.shiftKey) {
          const next = eraseAt(cur, p.x, p.y, aspect);
          if (next !== cur) onChange(next);
        } else {
          onChange({ ...cur, points: [...cur.points, { ...p, label: e.altKey || e.button === 2 ? 0 : 1 }] });
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
      {cur.points.map((p, i) => (
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
