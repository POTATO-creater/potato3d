import { useEffect, useState } from "react";
import { api, host, type ExportOptions, type ExportStats, type Msg, type SplatInfo } from "../lib/api";
import { fmtBytes, fmtCount, fmtDuration } from "../lib/format";
import { basename, revealLabel, type Os } from "../lib/platform";
import { Field, NumField, Segmented, Select } from "./fields";
import { useI18n } from "../i18n";

const STORE_KEY = "splatforge.export.v1";
const DEFAULTS: ExportOptions = { format: "sog", decimate: 0, minOpacity: 0, shDegree: null };

function loadOptions(): ExportOptions {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    /* armazenamento indisponível */
  }
  return DEFAULTS;
}

interface Props {
  os: Os | undefined;
  projectDir: string;
  finalPly: string;
  lastExport: ExportStats | null;
  onExported: (stats: ExportStats) => void;
  /** O visualizador está mostrando a exportação (e não o splat do treino). */
  previewing: boolean;
  onPreview: (on: boolean) => void;
}

export function ExportPanel(p: Props) {
  const { t, tm } = useI18n();
  const te = t.export;
  const [opts, setOpts] = useState<ExportOptions>(loadOptions);
  const [info, setInfo] = useState<SplatInfo | null>(null);
  const [progress, setProgress] = useState<{ fraction: number; message: Msg } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const set = (patch: Partial<ExportOptions>) => setOpts((o) => ({ ...o, ...patch }));

  useEffect(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(opts));
    } catch {
      /* ignora */
    }
  }, [opts]);

  useEffect(() => {
    setInfo(null);
    api.splatInfo(p.finalPly).then(setInfo, () => setInfo(null));
  }, [p.finalPly]);

  // Grau escolhido acima do treinado vira "manter".
  const trained = info?.shDegree ?? 3;
  const shDegree = opts.shDegree != null && opts.shDegree < trained ? opts.shDegree : null;
  const estimate = info ? Math.round(info.count * (1 - opts.decimate)) : null;

  async function run() {
    setError(null);
    setProgress({ fraction: 0, message: { key: "stage.starting" } });
    try {
      const stats = await api.exportSplat(p.projectDir, { ...opts, shDegree }, setProgress);
      p.onExported(stats);
    } catch (e) {
      setError(String(e));
    } finally {
      setProgress(null);
    }
  }

  const last = p.lastExport;
  const busy = progress != null;

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>{te.title}</h2>
        {info && (
          <span className="muted mono small">
            {fmtCount(info.count)} · {fmtBytes(info.bytes)} · SH {info.shDegree}
          </span>
        )}
      </header>

      <fieldset className="export-fields" disabled={busy}>
        <Field
          label={te.format}
          hint={opts.format === "sog" ? te.formatHintSog : te.formatHintPly}
        >
          <Segmented
            value={opts.format}
            onChange={(format) => set({ format })}
            options={[
              { value: "sog", label: te.sog },
              { value: "ply", label: te.ply },
            ]}
          />
        </Field>
        <div className="grid2">
          <NumField
            label={te.reduce}
            hint={te.reduceHint}
            value={Math.round(opts.decimate * 100)}
            min={0}
            max={90}
            step={5}
            suffix="%"
            onChange={(v) => set({ decimate: Math.min(90, Math.max(0, v)) / 100 })}
          />
          <NumField
            label={te.minOpacity}
            hint={te.minOpacityHint}
            value={Math.round(opts.minOpacity * 1000) / 10}
            min={0}
            max={20}
            step={0.5}
            suffix="%"
            onChange={(v) => set({ minOpacity: Math.min(20, Math.max(0, v)) / 100 })}
          />
        </div>
        <Field label={te.shBands} hint={te.shBandsHint}>
          <Select<number | null>
            value={shDegree}
            onChange={(shDegree) => set({ shDegree })}
            options={[
              { value: null, label: te.keep(te.shLabels[trained]) },
              ...te.shLabels.slice(0, trained).map((label, d) => ({
                value: d,
                label: label[0].toUpperCase() + label.slice(1),
                hint: te.shHints[d],
              })),
            ]}
          />
        </Field>
      </fieldset>

      {busy ? (
        <div className="export-progress">
          <div className="bar">
            <div className="bar-fill" style={{ width: `${Math.round(progress.fraction * 100)}%` }} />
          </div>
          <span className="help">{tm(progress.message)}…</span>
        </div>
      ) : (
        <button className="btn-sm export-run" onClick={run}>
          {estimate != null && opts.decimate > 0 ? te.runApprox(fmtCount(estimate)) : te.run}
        </button>
      )}

      {error && <p className="help export-error">{error}</p>}

      {last && !busy && (
        <div className="export-result">
          <div className="file-name" title={last.path}>
            {basename(last.path)}
          </div>
          <div className="meta-row">
            <span>
              {fmtCount(last.inputCount)} → {fmtCount(last.outputCount)} splats
            </span>
            <span>
              {fmtBytes(last.inputBytes)} → {fmtBytes(last.outputBytes)}
            </span>
            <span className="good">{te.smaller((last.inputBytes / last.outputBytes).toFixed(1))}</span>
            <span>{fmtDuration(last.seconds)}</span>
          </div>
          {last.prunedCount < last.inputCount && (
            <p className="help">
              {te.pruned(fmtCount(last.inputCount - last.prunedCount), fmtCount(last.prunedCount - last.outputCount))}
            </p>
          )}
          <div className="trim-actions">
            <button className={`btn-sm ${p.previewing ? "on" : ""}`} onClick={() => p.onPreview(!p.previewing)}>
              {p.previewing ? te.viewOriginal : te.view3d}
            </button>
            <button className="btn-sm ghost" onClick={() => host.reveal(last.path)}>
              {revealLabel(t, p.os)}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
