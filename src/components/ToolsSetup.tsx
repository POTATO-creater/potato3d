import { useEffect, useState } from "react";
import { api, type InstallEvent, type InstallPlan, type Tools } from "../lib/api";
import { installCommand } from "../lib/platform";
import { useI18n } from "../i18n";

type Missing = { name: string; key: "ffmpeg" | "colmap" | "brush" | "model" }[];

interface PackageState {
  label: string;
  done: number;
  total: number;
  phase: "waiting" | "download" | "extract" | "done";
}

const mb = (n: number) => `${(n / 1e6).toFixed(0)} MB`;

/** First run: lists what's missing and, on Windows, installs it in one click. */
export function ToolsSetup({
  tools,
  missing,
  rmbg,
  onInstalled,
}: {
  tools: Tools;
  missing: Missing;
  /** RMBG-1.4 was chosen: include it in the download plan. */
  rmbg: boolean;
  onInstalled: () => void;
}) {
  const { t } = useI18n();
  const tt = t.setup;
  /** Models have translated labels; tools use the name from the backend. */
  const pkgLabel = (id: string, label: string) => tt.packages[id] ?? label;
  const [plan, setPlan] = useState<InstallPlan | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [pkgs, setPkgs] = useState<Record<string, PackageState>>({});
  const [error, setError] = useState<string | null>(null);

  const modelMissing = missing.some((m) => m.key === "model");
  // On macOS the app only downloads RMBG; the rest comes from Homebrew.
  const canInstall =
    (tools.os === "windows" && missing.some((m) => m.key !== "brush")) ||
    ((tools.os === "macos" || tools.os === "web") && rmbg && modelMissing);
  const macMasker = tools.os === "macos" && modelMissing && !rmbg;
  const brewTools = tools.os !== "windows" ? missing.filter((m) => m.key === "ffmpeg" || m.key === "colmap") : [];

  useEffect(() => {
    if (!canInstall) return;
    api.installPlan(rmbg).then(setPlan, (e) => setPlanError(String(e)));
  }, [canInstall, rmbg]);

  function onEvent(e: InstallEvent) {
    switch (e.type) {
      case "started":
        setPkgs((p) => ({ ...p, [e.id]: { label: e.label, done: 0, total: e.total, phase: "download" } }));
        break;
      case "progress":
        setPkgs((p) => ({ ...p, [e.id]: { ...p[e.id], done: e.done, total: e.total } }));
        break;
      case "extracting":
        setPkgs((p) => ({ ...p, [e.id]: { ...p[e.id], phase: "extract" } }));
        break;
      case "done":
        setPkgs((p) => ({ ...p, [e.id]: { ...p[e.id], phase: "done", done: p[e.id].total } }));
        break;
      case "finished":
        setRunning(false);
        onInstalled();
        break;
      case "failed":
        setRunning(false);
        setError(e.error);
        break;
      case "cancelled":
        setRunning(false);
        setPkgs({});
        break;
    }
  }

  async function install() {
    setError(null);
    setPkgs(
      Object.fromEntries(
        (plan?.packages ?? []).map((p) => [p.id, { label: p.label, done: 0, total: p.size, phase: "waiting" as const }]),
      ),
    );
    setRunning(true);
    try {
      await api.installTools(rmbg, onEvent);
    } catch (e) {
      setRunning(false);
      setError(String(e));
    }
  }

  return (
    <div className="setup">
      <div className="setup-head">
        <div>
          <strong>{tt.title}</strong>
          <span className="muted"> · {missing.map((m) => m.name).join(", ")}</span>
        </div>
        <button className="link" onClick={onInstalled} disabled={running}>
          {tt.recheck}
        </button>
      </div>

      {canInstall ? (
        <>
          {plan && plan.packages.length > 0 && !running && Object.keys(pkgs).length === 0 && (
            <div className="setup-plan">
              <ul>
                {plan.packages.map((p) => (
                  <li key={p.id}>
                    <span>{pkgLabel(p.id, p.label)}</span>
                    <span className="mono muted">{mb(p.size)}</span>
                  </li>
                ))}
              </ul>
              <p className="help">
                {tools.os === "windows" && (plan.cuda ? tt.nvidia : tt.noNvidia)}
                {plan.packages.some((p) => p.id === "rmbg") && tt.rmbgLicense}
                {tools.os === "web" ? tt.storedInBrowser : tt.storedLocally}
              </p>
              <button className="btn-primary setup-btn" onClick={install}>
                {tt.install(mb(plan.totalBytes))}
              </button>
            </div>
          )}
          {!plan && !planError && <p className="help">{tt.checking}</p>}
          {planError && <p className="help bad">{tt.planError(planError)}</p>}

          {Object.keys(pkgs).length > 0 && (
            <div className="setup-progress">
              {Object.entries(pkgs).map(([id, p]) => (
                <div key={id} className="setup-row">
                  <div className="setup-row-head">
                    <span>{pkgLabel(id, p.label)}</span>
                    <span className="mono muted">
                      {p.phase === "waiting" && tt.queued}
                      {p.phase === "download" && `${mb(p.done)} / ${mb(p.total)}`}
                      {p.phase === "extract" && tt.extracting}
                      {p.phase === "done" && tt.ready}
                    </span>
                  </div>
                  <div className="bar">
                    <div
                      className={`bar-fill ${p.phase === "extract" ? "indeterminate" : ""}`}
                      style={{ width: `${p.phase === "extract" ? 100 : Math.round((p.done / Math.max(p.total, 1)) * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
              {running && (
                <button className="btn-sm" onClick={() => api.cancelInstall()}>
                  {t.common.cancel}
                </button>
              )}
            </div>
          )}
          {tools.os !== "web" && missing.some((m) => m.key === "brush") && (
            <p className="help">
              {tt.brushDev} <code>pnpm setup:tools</code>.
            </p>
          )}
          {brewTools.length > 0 && (
            <div className="setup-cmds">
              {brewTools.map((m) => (
                <span key={m.name}>
                  {m.name} → <code>{installCommand(t, m.key, tools.os)}</code>
                </span>
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="setup-cmds">
          {missing
            .filter((m) => m.key !== "model")
            .map((m) => (
              <span key={m.name}>
                {m.name} → <code>{installCommand(t, m.key, tools.os)}</code>
              </span>
            ))}
          {macMasker && <span>{tt.visionNote}</span>}
        </div>
      )}
      {error && <pre className="setup-error">{error}</pre>}
    </div>
  );
}
