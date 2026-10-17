import { useEffect, useRef, useState } from "react";
import { itemName, type BatchItem, type ItemStatus } from "../lib/batch";
import { useI18n, type Messages } from "../i18n";

interface Entry {
  item: BatchItem;
  status: ItemStatus;
}

function statusText(t: Messages, s: ItemStatus) {
  const tb = t.sidebar.batch;
  return s.kind === "running" ? `${t.stages[s.stage]} · ${Math.round(s.fraction * 100)}%` : tb.status[s.kind];
}

function StatusIcon({ status }: { status: ItemStatus }) {
  return <span className={`batch-status ${status.kind}`} aria-hidden />;
}

/** Inputs open in the batch: ‹ current › plus a list to jump to or remove one. */
export function BatchSelector({
  items,
  active,
  onSelect,
  onRemove,
  disabled,
}: {
  items: Entry[];
  active: string | null;
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
  /** Switching is not allowed right now (a retouch works on the shown item). */
  disabled?: boolean;
}) {
  const { t } = useI18n();
  const tb = t.sidebar.batch;
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const i = Math.max(0, items.findIndex((e) => e.item.id === active));
  const current = items[i];
  const done = items.filter((e) => e.status.kind === "done").length;

  useEffect(() => {
    if (!open) return;
    const close = (e: Event) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !root.current?.contains(e.target as Node)) setOpen(false);
    };
    addEventListener("pointerdown", close);
    addEventListener("keydown", close);
    return () => {
      removeEventListener("pointerdown", close);
      removeEventListener("keydown", close);
    };
  }, [open]);

  if (!current) return null;
  const go = (d: number) => onSelect(items[(i + d + items.length) % items.length].item.id);

  return (
    <div className="batch" ref={root}>
      <div className="batch-bar">
        <button className="batch-nav" onClick={() => go(-1)} disabled={disabled} aria-label={tb.prev} title={tb.prev}>
          ‹
        </button>
        <button
          className="batch-current"
          onClick={() => setOpen((o) => !o)}
          aria-haspopup="listbox"
          aria-expanded={open}
          title={itemName(current.item)}
        >
          <StatusIcon status={current.status} />
          <span className="batch-name">{itemName(current.item)}</span>
          <span className="batch-pos mono">
            {i + 1}/{items.length}
          </span>
          <span className="batch-caret" aria-hidden>
            ▾
          </span>
        </button>
        <button className="batch-nav" onClick={() => go(1)} disabled={disabled} aria-label={tb.next} title={tb.next}>
          ›
        </button>
      </div>
      {current.status.kind === "running" && (
        <div className="bar batch-progress">
          <div className="bar-fill" style={{ width: `${Math.round(current.status.fraction * 100)}%` }} />
        </div>
      )}

      {open && (
        <div className="batch-menu">
          <ul role="listbox" aria-label={tb.title}>
            {items.map(({ item, status }) => (
              <li key={item.id} className={`batch-row ${item.id === active ? "active" : ""}`}>
                <button
                  className="batch-pick"
                  role="option"
                  aria-selected={item.id === active}
                  disabled={disabled}
                  onClick={() => {
                    onSelect(item.id);
                    setOpen(false);
                  }}
                >
                  <StatusIcon status={status} />
                  <span className="batch-name" title={itemName(item)}>
                    {itemName(item)}
                  </span>
                  <span className={`batch-state ${status.kind}`}>{statusText(t, status)}</span>
                </button>
                <button
                  className="batch-remove"
                  onClick={() => onRemove(item.id)}
                  disabled={status.kind === "running"}
                  aria-label={tb.remove(itemName(item))}
                  title={tb.removeTitle}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
          <div className="batch-foot muted">{tb.summary(done, items.length)}</div>
        </div>
      )}
    </div>
  );
}
