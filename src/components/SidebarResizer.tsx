import { useRef, useState } from "react";
import { useI18n } from "../i18n";

const KEY = "splatforge.sidebar.v1";
export const SIDEBAR_DEFAULT = 340;
const MIN = 280;
/** Espaço mínimo que sobra para o 3D. */
const MAIN_MIN = 420;

const clampWidth = (w: number) => Math.round(Math.max(MIN, Math.min(w, window.innerWidth - MAIN_MIN)));

function load() {
  try {
    const w = Number(localStorage.getItem(KEY));
    return Number.isFinite(w) && w >= MIN ? w : SIDEBAR_DEFAULT;
  } catch {
    return SIDEBAR_DEFAULT;
  }
}

function save(w: number) {
  try {
    localStorage.setItem(KEY, String(w));
  } catch {
    /* armazenamento indisponível */
  }
}

/** Largura da barra lateral (salva) e a alça na borda que a arrasta. */
export function useSidebarWidth() {
  const [width, setWidth] = useState(load);
  const drag = useRef<{ x: number; w: number } | null>(null);
  const { t } = useI18n();

  const handle = (
    <div
      className="sidebar-resize"
      role="separator"
      aria-orientation="vertical"
      title={t.app.resizeSidebar}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        drag.current = { x: e.clientX, w: clampWidth(width) };
        document.body.classList.add("resizing");
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (d) setWidth(clampWidth(d.w + e.clientX - d.x));
      }}
      onPointerUp={() => {
        drag.current = null;
        document.body.classList.remove("resizing");
        setWidth((w) => {
          save(w);
          return w;
        });
      }}
      onDoubleClick={() => {
        setWidth(SIDEBAR_DEFAULT);
        save(SIDEBAR_DEFAULT);
      }}
    />
  );
  return { width, handle };
}
