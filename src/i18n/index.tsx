// UI translations. Each language is a catalog with the same shape as pt-BR
// (TypeScript flags missing keys). Texts with parameters are functions;
// backend progress messages (`Msg`) live under `msg`.

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Msg } from "../lib/api";
import { CATALOGS, initialLocale, LOCALE_KEY as STORE_KEY, type Args, type Locale, type Messages } from "./catalogs";

export { CATALOGS, LOCALES, type Locale, type Messages } from "./catalogs";

/** Catalog text for a `Msg`; plain text passes through. */
export function formatMsg(t: Messages, m: Msg | string | null | undefined): string {
  if (!m) return "";
  if (typeof m === "string") return m;
  const fmt = (t.msg as Record<string, ((a: Args) => string) | undefined>)[m.key];
  if (!fmt) return m.key;
  const args: Args = {};
  for (const [k, v] of Object.entries(m.args ?? {})) args[k] = typeof v === "object" ? formatMsg(t, v) : v;
  return fmt(args);
}

interface I18n {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: Messages;
  /** Shortcut for `formatMsg(t, m)`. */
  tm: (m: Msg | string | null | undefined) => string;
}

const Ctx = createContext<I18n | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(initialLocale);
  const t = CATALOGS[locale];

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l);
    try {
      localStorage.setItem(STORE_KEY, l);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  const value = useMemo<I18n>(() => ({ locale, setLocale, t, tm: (m) => formatMsg(t, m) }), [locale, setLocale, t]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useI18n(): I18n {
  const v = useContext(Ctx);
  if (!v) throw new Error("useI18n used outside I18nProvider");
  return v;
}
