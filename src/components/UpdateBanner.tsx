import { useEffect, useState } from "react";
import { host } from "../lib/api";
import { useI18n } from "../i18n";

/** Web version: offers to switch to the already-downloaded new version.
 *  Reloading mid-job would lose the work, so the button waits. */
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
