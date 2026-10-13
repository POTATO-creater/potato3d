// Catalogs and language selection, without React: also used by the shared
// link page (web/src/share/).

import { ptBR, type Args, type Messages } from "./pt-BR";
import { enUS } from "./en-US";
import { es } from "./es";

export type { Args, Messages };
export type Locale = "pt-BR" | "en-US" | "es";

export const CATALOGS: Record<Locale, Messages> = { "pt-BR": ptBR, "en-US": enUS, es };
export const LOCALES = Object.keys(CATALOGS) as Locale[];

export const LOCALE_KEY = "splatforge.locale.v1";

/** Saved language or, the first time, the system's (English if none matches). */
export function initialLocale(): Locale {
  try {
    const saved = localStorage.getItem(LOCALE_KEY);
    if (saved && saved in CATALOGS) return saved as Locale;
  } catch {
    /* storage unavailable */
  }
  for (const lang of navigator.languages ?? [navigator.language]) {
    const l = lang.toLowerCase();
    if (l.startsWith("pt")) return "pt-BR";
    if (l.startsWith("es")) return "es";
    if (l.startsWith("en")) return "en-US";
  }
  return "en-US";
}
