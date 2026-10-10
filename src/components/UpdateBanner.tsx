import { useEffect, useState } from "react";
import { host } from "../lib/api";
import { useI18n } from "../i18n";

/** Versão web: oferece trocar para a versão nova já baixada. Recarregar no
 *  meio de um processamento perderia o trabalho, então o botão espera. */
export function UpdateBanner({ busy }: { busy: boolean }) {
  const { t } = useI18n();
  const [available, setAvailable] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => host.onUpdate?.(() => setAvailable(true)), []);

  if (!available || dismissed || !host.applyUpdate) return null;
  return (
    <div className="banner info" role="status">
      <span>{t.update.available}</span>
      <span className="banner-actions">
        <button className="btn-sm" onClick={() => host.applyUpdate?.()} disabled={busy} title={busy ? t.update.busy : undefined}>
          {t.update.apply}
        </button>
        <button className="btn-sm ghost" onClick={() => setDismissed(true)}>
          {t.update.later}
        </button>
      </span>
    </div>
  );
}
