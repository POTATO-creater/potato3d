import { CATALOGS, LOCALES, useI18n, type Locale } from "../i18n";
import { Select } from "./fields";

/** Switches the UI language (persisted in the browser/webview). */
export function LanguageSelect() {
  const { locale, setLocale, t } = useI18n();
  return (
    <span className="lang-select">
      <Select<Locale>
        aria-label={t.common.language}
        value={locale}
        onChange={setLocale}
        options={LOCALES.map((l) => ({ value: l, label: CATALOGS[l].language }))}
      />
    </span>
  );
}
