import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="field">
      <span className="field-label">
        {label}
        {hint && <span className="field-hint">{hint}</span>}
      </span>
      {children}
    </label>
  );
}

export function NumField(props: {
  label: string;
  hint?: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  suffix?: string;
  disabled?: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <Field label={props.label} hint={props.hint}>
      <span className="input-wrap">
        <input
          type="number"
          className="input num"
          value={props.value}
          min={props.min}
          max={props.max}
          step={props.step ?? 1}
          disabled={props.disabled}
          onChange={(e) => {
            const v = e.currentTarget.valueAsNumber;
            if (!Number.isNaN(v)) props.onChange(v);
          }}
        />
        {props.suffix && <span className="input-suffix">{props.suffix}</span>}
      </span>
    </Field>
  );
}

export function Toggle(props: { label: string; hint?: string; checked: boolean; disabled?: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="toggle">
      <input
        type="checkbox"
        checked={props.checked}
        disabled={props.disabled}
        onChange={(e) => props.onChange(e.currentTarget.checked)}
      />
      <span className="toggle-track" aria-hidden />
      <span className="toggle-text">
        {props.label}
        {props.hint && <span className="field-hint">{props.hint}</span>}
      </span>
    </label>
  );
}

export function Segmented<T extends string>(props: {
  value: T;
  options: { value: T; label: string; disabled?: boolean; title?: string }[];
  onChange: (v: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="segmented" role="radiogroup">
      {props.options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={props.value === o.value}
          title={o.title}
          className={props.value === o.value ? "active" : ""}
          disabled={props.disabled || o.disabled}
          onClick={() => props.onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  const pct = ((props.value - props.min) / (props.max - props.min)) * 100;
  return (
    <label className="slider">
      <span className="slider-head">
        <span>{props.label}</span>
        <span className="mono">{props.value}</span>
      </span>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        value={props.value}
        style={{ "--fill": `${pct}%` } as React.CSSProperties}
        onChange={(e) => props.onChange(e.currentTarget.valueAsNumber)}
      />
    </label>
  );
}

export interface SelectOption<T> {
  value: T;
  label: string;
  /** Secondary line in the menu (not shown on the button). */
  hint?: string;
}

/** Text without accents and lowercased, for Select search. */
const fold = (t: string) => t.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();

/** App-styled dropdown replacing the native `<select>`. The menu lives in a
 *  fixed-position portal (not clipped by the scrolling sidebar) and opens
 *  upward when there's no room below. Keyboard: arrows, Home/End,
 *  Enter/Space and Esc. With `searchable`, the menu gets a search box that
 *  filters by label and secondary line (all typed words must match). */
export function Select<T extends string | number | null>(props: {
  value: T;
  options: SelectOption<T>[];
  onChange: (v: T) => void;
  disabled?: boolean;
  "aria-label"?: string;
  searchable?: boolean;
  searchPlaceholder?: string;
}) {
  const button = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [place, setPlace] = useState<{ left: number; width: number; top?: number; bottom?: number; maxHeight: number } | null>(null);
  const id = useId();
  const current = props.options.findIndex((o) => o.value === props.value);
  const selected = props.options[current];
  // Visible options (with their original index), filtered by the search.
  const terms = fold(query).split(/\s+/).filter(Boolean);
  const shown = props.options
    .map((o, i) => ({ o, i }))
    .filter(({ o }) => {
      if (!terms.length) return true;
      const text = fold(`${o.label} ${o.hint ?? ""}`);
      return terms.every((t) => text.includes(t));
    });

  const position = useCallback(() => {
    const r = button.current?.getBoundingClientRect();
    if (!r) return;
    const below = window.innerHeight - r.bottom - 8, above = r.top - 8;
    const up = below < 180 && above > below;
    setPlace({
      left: r.left,
      width: Math.max(r.width, props.searchable ? 260 : 0),
      ...(up ? { bottom: window.innerHeight - r.top + 4 } : { top: r.bottom + 4 }),
      maxHeight: Math.min(props.searchable ? 340 : 280, up ? above : below),
    });
  }, [props.searchable]);

  function show() {
    if (props.disabled) return;
    position();
    setQuery("");
    setActive(Math.max(current, 0));
    setOpen(true);
  }

  function choose(k: number) {
    setOpen(false);
    button.current?.focus();
    const pick = shown[k];
    if (pick && pick.i !== current) props.onChange(pick.o.value);
  }

  // Close on outside click; follow scroll and resize.
  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!button.current?.contains(t) && !menu.current?.contains(t)) setOpen(false);
    };
    document.addEventListener("pointerdown", away, true);
    window.addEventListener("resize", position);
    window.addEventListener("scroll", position, true);
    return () => {
      document.removeEventListener("pointerdown", away, true);
      window.removeEventListener("resize", position);
      window.removeEventListener("scroll", position, true);
    };
  }, [open, position]);

  useEffect(() => {
    if (open) list.current?.querySelector<HTMLElement>(`[data-index="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [open, active]);

  useEffect(() => {
    if (open && props.searchable) search.current?.focus();
  }, [open, props.searchable]);

  function onKeyDown(e: React.KeyboardEvent) {
    const last = shown.length - 1;
    const inSearch = e.target === search.current;
    const move = (i: number) => (open ? setActive(Math.min(Math.max(i, 0), Math.max(last, 0))) : show());
    switch (e.key) {
      case "ArrowDown":
        move(active + 1);
        break;
      case "ArrowUp":
        move(active - 1);
        break;
      case "Home":
        if (inSearch) return;
        move(0);
        break;
      case "End":
        if (inSearch) return;
        move(last);
        break;
      case "Enter":
        if (open) choose(active);
        else show();
        break;
      case " ":
        if (inSearch) return;
        if (open) choose(active);
        else show();
        break;
      case "Escape":
        if (!open) return;
        setOpen(false);
        button.current?.focus();
        break;
      case "Tab":
        setOpen(false);
        return;
      default:
        return;
    }
    e.preventDefault();
  }

  const options = (
    <ul
      ref={(el) => {
        list.current = el;
        if (!props.searchable) menu.current = el;
      }}
      id={id}
      role="listbox"
      className={props.searchable ? "select-list" : "select-menu"}
      style={props.searchable ? undefined : place ?? undefined}
      tabIndex={-1}
    >
      {shown.map(({ o, i }, k) => (
        <li
          key={String(o.value)}
          id={`${id}-${k}`}
          data-index={k}
          role="option"
          aria-selected={i === current}
          className={`select-option ${k === active ? "active" : ""} ${i === current ? "selected" : ""}`}
          onPointerEnter={() => setActive(k)}
          onClick={() => choose(k)}
        >
          <span className="select-option-text">
            {o.label}
            {o.hint && <span className="select-option-hint">{o.hint}</span>}
          </span>
          {i === current && (
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
              <path d="M2.5 6.2 5 8.5 9.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </li>
      ))}
      {!shown.length && <li className="select-empty">—</li>}
    </ul>
  );

  return (
    <>
      <button
        ref={button}
        type="button"
        className={`input select ${open ? "open" : ""}`}
        role="combobox"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={id}
        aria-activedescendant={open && shown.length ? `${id}-${active}` : undefined}
        aria-label={props["aria-label"]}
        disabled={props.disabled}
        onClick={() => (open ? setOpen(false) : show())}
        onKeyDown={onKeyDown}
      >
        <span className="select-value">{selected?.label ?? "—"}</span>
        <svg className="select-chevron" width="10" height="10" viewBox="0 0 10 10" aria-hidden>
          <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open &&
        place &&
        createPortal(
          props.searchable ? (
            <div ref={(el) => void (menu.current = el)} className="select-menu select-menu-search" style={place}>
              <input
                ref={search}
                className="input select-search"
                type="search"
                value={query}
                placeholder={props.searchPlaceholder}
                aria-label={props.searchPlaceholder}
                aria-controls={id}
                onChange={(e) => {
                  setQuery(e.currentTarget.value);
                  setActive(0);
                }}
                onKeyDown={onKeyDown}
              />
              {options}
            </div>
          ) : (
            options
          ),
          document.body,
        )}
    </>
  );
}
