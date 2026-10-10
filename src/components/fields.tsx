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
  /** Linha de apoio no menu (não aparece no botão). */
  hint?: string;
}

/** Lista suspensa no estilo do app, no lugar do `<select>` nativo. O menu vai
 *  num portal com posição fixa (não é cortado pela barra lateral rolável) e
 *  abre para cima quando falta espaço embaixo. Teclado: setas, Home/End,
 *  Enter/Espaço e Esc. */
export function Select<T extends string | number | null>(props: {
  value: T;
  options: SelectOption<T>[];
  onChange: (v: T) => void;
  disabled?: boolean;
  "aria-label"?: string;
}) {
  const button = useRef<HTMLButtonElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [place, setPlace] = useState<{ left: number; width: number; top?: number; bottom?: number; maxHeight: number } | null>(null);
  const id = useId();
  const current = props.options.findIndex((o) => o.value === props.value);
  const selected = props.options[current];

  const position = useCallback(() => {
    const r = button.current?.getBoundingClientRect();
    if (!r) return;
    const below = window.innerHeight - r.bottom - 8, above = r.top - 8;
    const up = below < 180 && above > below;
    setPlace({
      left: r.left,
      width: r.width,
      ...(up ? { bottom: window.innerHeight - r.top + 4 } : { top: r.bottom + 4 }),
      maxHeight: Math.min(280, up ? above : below),
    });
  }, []);

  function show() {
    if (props.disabled) return;
    position();
    setActive(Math.max(current, 0));
    setOpen(true);
  }

  function choose(i: number) {
    setOpen(false);
    button.current?.focus();
    if (props.options[i] && i !== current) props.onChange(props.options[i].value);
  }

  // Fecha ao clicar fora; acompanha rolagem e redimensionamento.
  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!button.current?.contains(t) && !list.current?.contains(t)) setOpen(false);
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

  function onKeyDown(e: React.KeyboardEvent) {
    const last = props.options.length - 1;
    const move = (i: number) => (open ? setActive(Math.min(Math.max(i, 0), last)) : show());
    switch (e.key) {
      case "ArrowDown":
        move(active + 1);
        break;
      case "ArrowUp":
        move(active - 1);
        break;
      case "Home":
        move(0);
        break;
      case "End":
        move(last);
        break;
      case "Enter":
      case " ":
        if (open) choose(active);
        else show();
        break;
      case "Escape":
        if (!open) return;
        setOpen(false);
        break;
      case "Tab":
        setOpen(false);
        return;
      default:
        return;
    }
    e.preventDefault();
  }

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
        aria-activedescendant={open ? `${id}-${active}` : undefined}
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
          <ul ref={list} id={id} role="listbox" className="select-menu" style={place} tabIndex={-1}>
            {props.options.map((o, i) => (
              <li
                key={String(o.value)}
                id={`${id}-${i}`}
                data-index={i}
                role="option"
                aria-selected={i === current}
                className={`select-option ${i === active ? "active" : ""} ${i === current ? "selected" : ""}`}
                onPointerEnter={() => setActive(i)}
                onClick={() => choose(i)}
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
          </ul>,
          document.body,
        )}
    </>
  );
}
