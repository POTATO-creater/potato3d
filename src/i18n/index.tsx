// Traduções da interface. Cada idioma é um catálogo com a mesma forma do
// pt-BR (o TypeScript acusa chave faltando). Textos com parâmetros são
// funções; as mensagens de progresso do backend (`Msg`) ficam em `msg`.

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Msg } from "../lib/api";
import { ptBR, type Args, type Messages } from "./pt-BR";
import { enUS } from "./en-US";
import { es } from "./es";

export type { Messages };
export type Locale = "pt-BR" | "en-US" | "es";

export const CATALOGS: Record<Locale, Messages> = { "pt-BR": ptBR, "en-US": enUS, es };
export const LOCALES = Object.keys(CATALOGS) as Locale[];

const STORE_KEY = "splatforge.locale.v1";

/** Idioma salvo ou, na primeira vez, o do sistema (inglês se não houver). */
function initialLocale(): Locale {
  try {
    const saved = localStorage.getItem(STORE_KEY);
    if (saved && saved in CATALOGS) return saved as Locale;
  } catch {
    /* armazenamento indisponível */
  }
  for (const lang of navigator.languages ?? [navigator.language]) {
    const l = lang.toLowerCase();
    if (l.startsWith("pt")) return "pt-BR";
    if (l.startsWith("es")) return "es";
    if (l.startsWith("en")) return "en-US";
  }
  return "en-US";
}

/** Texto de uma `Msg` no catálogo; texto puro passa direto (eventos gravados
 *  antes das mensagens traduzíveis, usados pelo `dev:mock`). */
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
  /** Atalho para `formatMsg(t, m)`. */
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
      /* ignora */
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
  if (!v) throw new Error("useI18n fora do I18nProvider");
  return v;
}
