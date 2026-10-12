import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { fileUrl, type FrameInfo } from "../lib/api";
import { useI18n } from "../i18n";

const nameOf = (p: string) => p.split(/[\\/]/).pop() ?? p;
const stemOf = (p: string) => (p.split(/[\\/]/).pop() ?? p).replace(/\.[^.]+$/, "");
/** De onde o frame veio: o instante no vídeo ou a imagem de origem. */
const origin = (f: FrameInfo) => f.source ?? `${f.time.toFixed(2)} s`;

/** Recorte por CSS (luminância da máscara), no mesmo enquadramento da <img>
 *  (object-fit: cover, centralizada). */
function frameMaskStyle(f: FrameInfo, masksDir: { path: string; version: number } | null) {
  // Frame recusado pode não ter recorte: a máscara faltando o esconderia.
  if (!masksDir || f.rejected) return undefined;
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

/** Ícones dos botões redondos do modal: SVG centralizado pela geometria, e
 *  não glifos de texto, que cada fonte assenta numa altura. */
function ModalIcon({ d }: { d: string }) {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
      <path d={d} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
const ICON_PREV = "M11 3.5 5.5 9l5.5 5.5";
const ICON_NEXT = "M7 3.5 12.5 9 7 14.5";
const ICON_CLOSE = "M4.5 4.5l9 9M13.5 4.5l-9 9";

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
        {f.rejected && <span className="rejected-tag">{tf.rejected}</span>}
      </span>
      <button className="lightbox-nav prev" onClick={stop(() => go(index - 1))} disabled={index === 0} aria-label={tf.prev} title={`${tf.prev} (←)`}>
        <ModalIcon d={ICON_PREV} />
      </button>
      <button className="lightbox-nav next" onClick={stop(() => go(index + 1))} disabled={index === last} aria-label={tf.next} title={`${tf.next} (→)`}>
        <ModalIcon d={ICON_NEXT} />
      </button>
      <button className="lightbox-close" onClick={stop(onClose)} aria-label={tf.close} title={`${tf.close} (Esc)`}>
        <ModalIcon d={ICON_CLOSE} />
      </button>
    </div>
  );
}

export interface MenuItem {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  danger?: boolean;
}

/** Menu de contexto na posição do clique, mantido dentro da janela. Fecha com
 *  Esc, clique fora, rolagem ou perda de foco. */
function ContextMenu({ x, y, items, onClose }: { x: number; y: number; items: (MenuItem | "sep")[]; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: x, top: y });
  useLayoutEffect(() => {
    const r = ref.current!.getBoundingClientRect();
    setPos({ left: Math.min(x, innerWidth - r.width - 6), top: Math.min(y, innerHeight - r.height - 6) });
  }, [x, y]);
  useEffect(() => {
    const close = (e: Event) => {
      if (e.target instanceof Node && ref.current?.contains(e.target)) return;
      onClose();
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("pointerdown", close, true);
    window.addEventListener("scroll", close, true);
    window.addEventListener("blur", onClose);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", close, true);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("blur", onClose);
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose]);
  return (
    <div ref={ref} className="ctx-menu" role="menu" style={pos} onContextMenu={(e) => e.preventDefault()}>
      {items.map((it, i) =>
        it === "sep" ? (
          <hr key={i} />
        ) : (
          <button
            key={i}
            role="menuitem"
            className={it.danger ? "danger" : ""}
            disabled={it.disabled}
            onClick={() => {
              onClose();
              it.onClick();
            }}
          >
            {it.label}
          </button>
        ),
      )}
    </div>
  );
}

/** Miniaturas dos frames escolhidos, com a nitidez relativa de cada um.
 *  Frames bem abaixo da mediana costumam virar borrão no splat. Com recorte,
 *  a máscara é aplicada por CSS (luminância) sobre um xadrez. O botão direito
 *  recusa um frame (sai do recorte, das poses e do treino) ou o aceita de volta. */
export function FramesGrid({
  frames,
  version,
  masksDir,
  onReject,
  locked,
}: {
  frames: FrameInfo[];
  version: number;
  masksDir: { path: string; version: number } | null;
  /** Recusa (ou aceita de volta) frames pelo nome do arquivo. */
  onReject?: (names: string[], rejected: boolean) => void;
  /** Um job está rodando: os arquivos não podem mudar de lugar. */
  locked?: boolean;
}) {
  const { t } = useI18n();
  const tf = t.frames;
  const [open, setOpen] = useState<number | null>(null);
  const [menu, setMenu] = useState<{ x: number; y: number; index: number } | null>(null);
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

  const isWeak = (f: FrameInfo) => f.sharpness < median * 0.5;
  const rejected = frames.filter((f) => f.rejected);
  const weakAccepted = frames.filter((f) => isWeak(f) && !f.rejected);
  const low = weakAccepted.length;

  function menuItems(f: FrameInfo, index: number): (MenuItem | "sep")[] {
    const set = (list: FrameInfo[], value: boolean) => () => onReject?.(list.map((x) => nameOf(x.path)), value);
    const items: (MenuItem | "sep")[] = [
      f.rejected
        ? { label: tf.accept, onClick: set([f], false), disabled: locked }
        : { label: tf.reject, onClick: set([f], true), disabled: locked, danger: true },
      { label: tf.open, onClick: () => setOpen(index) },
    ];
    const bulk: MenuItem[] = [];
    if (weakAccepted.length > 0)
      bulk.push({ label: tf.rejectWeak(weakAccepted.length), onClick: set(weakAccepted, true), disabled: locked, danger: true });
    if (rejected.length > 0) bulk.push({ label: tf.acceptAll(rejected.length), onClick: set(rejected, false), disabled: locked });
    return bulk.length ? [...items, "sep", ...bulk] : items;
  }

  return (
    <div className="frames">
      <div className="frames-summary">
        <span>{tf.count(frames.length - rejected.length)}</span>
        {rejected.length > 0 && <span className="muted">{tf.rejectedCount(rejected.length)}</span>}
        {low > 0 ? <span className="warn">{tf.lowSharpness(low)}</span> : <span className="good">{tf.uniform}</span>}
        {onReject && <span className="muted frames-hint">{tf.rejectHint}</span>}
        {masksDir && (
          <button className={`btn-sm ${showCut ? "" : "ghost"}`} onClick={() => setShowCut((v) => !v)}>
            {showCut ? tf.showingCut : tf.showCut}
          </button>
        )}
      </div>
      <div className="frames-grid">
        {frames.map((f, i) => {
          const weak = isWeak(f);
          return (
            <button
              key={nameOf(f.path)}
              className={`thumb ${weak ? "weak" : ""} ${cut ? "cut" : ""} ${f.rejected ? "rejected" : ""} ${menu?.index === i ? "menu-open" : ""}`}
              onClick={() => setOpen(i)}
              onContextMenu={(e) => {
                if (!onReject) return;
                e.preventDefault();
                setMenu({ x: e.clientX, y: e.clientY, index: i });
              }}
            >
              <img src={fileUrl(f.path, version)} loading="lazy" decoding="async" alt={`frame ${i}`} style={cut ? frameMaskStyle(f, masksDir) : undefined} />
              <span className="thumb-meta mono">
                #{i} · {origin(f)}
              </span>
              {f.rejected && <span className="thumb-tag">{tf.rejected}</span>}
              <span className="thumb-bar" style={{ width: `${(f.sharpness / max) * 100}%` }} />
            </button>
          );
        })}
      </div>
      {menu && frames[menu.index] && (
        <ContextMenu x={menu.x} y={menu.y} items={menuItems(frames[menu.index], menu.index)} onClose={() => setMenu(null)} />
      )}
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
