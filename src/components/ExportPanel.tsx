import { useEffect, useRef, useState } from "react";
import { api, host, type ExportOptions, type ExportStats, type Msg, type SceneFrame, type SplatInfo } from "../lib/api";
import { fmtBytes, fmtCount, fmtDuration } from "../lib/format";
import { basename, revealLabel, type Os } from "../lib/platform";
import { Field, NumField, Segmented, Select, Toggle } from "./fields";
import { SharePanel } from "./SharePanel";
import { savedShare, shareKey } from "../lib/share";
import { useI18n } from "../i18n";

const STORE_KEY = "splatforge.export.v1";
const DEFAULTS: ExportOptions = { format: "sog", decimate: 0, minOpacity: 0, shDegree: null, meshTriangles: 50_000, meshTexture: true };
const MESH_TRIANGLES = [20_000, 50_000, 100_000, 200_000];

function loadOptions(): ExportOptions {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    /* storage unavailable */
  }
  return DEFAULTS;
}

interface Props {
  os: Os | undefined;
  projectDir: string;
  finalPly: string;
  lastExport: ExportStats | null;
  onExported: (stats: ExportStats) => void;
  /** The viewer is showing the export (not the trained splat). */
  previewing: boolean;
  onPreview: (on: boolean) => void;
  /** Video framing, passed along with the share link. */
  frame: SceneFrame | null;
}

export function ExportPanel(p: Props) {
  const { t, tm } = useI18n();
  const te = t.export;
  const [opts, setOpts] = useState<ExportOptions>(loadOptions);
  const [info, setInfo] = useState<SplatInfo | null>(null);
  const [progress, setProgress] = useState<{ fraction: number; message: Msg } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const set = (patch: Partial<ExportOptions>) => setOpts((o) => ({ ...o, ...patch }));
  const last = p.lastExport;
  // Open by default if this export has a valid link created on this device.
  const lastKey = last ? shareKey(last.path, last.outputBytes) : null;
  const [sharing, setSharing] = useState(false);
  useEffect(() => setSharing(!!lastKey && !!savedShare(lastKey)), [lastKey]);

  useEffect(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(opts));
    } catch {
      /* ignore */
    }
  }, [opts]);

  useEffect(() => {
    setInfo(null);
    api.splatInfo(p.finalPly).then(setInfo, () => setInfo(null));
  }, [p.finalPly]);

  // A degree above the trained one becomes "keep".
  // Mesh export runs on the web only for now.
  const meshAvailable = p.os === "web";
  const mesh = opts.format === "glb" && meshAvailable;
  const abort = useRef<AbortController | null>(null);
  const trained = info?.shDegree ?? 3;
  const shDegree = opts.shDegree != null && opts.shDegree < trained ? opts.shDegree : null;
  const estimate = info ? Math.round(info.count * (1 - opts.decimate)) : null;

  async function run() {
    setError(null);
    setProgress({ fraction: 0, message: { key: "stage.starting" } });
    abort.current = new AbortController();
    try {
      const format = opts.format === "glb" && !meshAvailable ? "sog" : opts.format;
      const stats = await api.exportSplat(p.projectDir, { ...opts, format, shDegree }, setProgress, abort.current.signal);
      p.onExported(stats);
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) setError(String(e));
    } finally {
      abort.current = null;
      setProgress(null);
    }
  }

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
          hint={mesh ? te.formatHintGlb : opts.format === "ply" ? te.formatHintPly : te.formatHintSog}
        >
          <Segmented
            value={opts.format === "glb" && !meshAvailable ? "sog" : opts.format}
            onChange={(format) => set({ format })}
            options={[
              { value: "sog", label: te.sog },
              { value: "ply", label: te.ply },
              ...(meshAvailable ? [{ value: "glb" as const, label: te.glb }] : []),
            ]}
          />
        </Field>
        {mesh ? (
          <>
            <Field label={te.meshTriangles} hint={te.meshTrianglesHint}>
              <Select<number>
                value={opts.meshTriangles ?? 50_000}
                onChange={(meshTriangles) => set({ meshTriangles })}
                options={MESH_TRIANGLES.map((n) => ({ value: n, label: fmtCount(n) }))}
              />
            </Field>
            <Toggle
              label={te.meshTexture}
              hint={te.meshTextureHint}
              checked={opts.meshTexture ?? true}
              onChange={(meshTexture) => set({ meshTexture })}
            />
          </>
        ) : (
          <>
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
          </>
        )}
      </fieldset>

      {busy ? (
        <div className="export-progress">
          <div className="bar">
            <div className="bar-fill" style={{ width: `${Math.round(progress.fraction * 100)}%` }} />
          </div>
          <div className="export-status">
            <span className="help">{tm(progress.message)}…</span>
            {mesh && (
              <button className="btn-sm ghost" onClick={() => abort.current?.abort()}>
                {te.cancel}
              </button>
            )}
          </div>
        </div>
      ) : (
        <button className="btn-sm export-run" onClick={run}>
          {mesh
            ? te.runMesh(fmtCount(opts.meshTriangles ?? 50_000))
            : estimate != null && opts.decimate > 0
              ? te.runApprox(fmtCount(estimate))
              : te.run}
        </button>
      )}

      {error && <p className="help export-error">{error}</p>}

      {last && !busy && (
        <div className="export-result">
          <div className="file-name" title={last.path}>
            {basename(last.path)}
          </div>
          {last.mesh ? (
            <div className="meta-row">
              <span>{te.meshResult(fmtCount(last.mesh.triangles), fmtCount(last.mesh.vertices))}</span>
              {last.mesh.texture ? <span>{te.meshTextureSize(last.mesh.texture)}</span> : null}
              <span>{fmtBytes(last.outputBytes)}</span>
              <span>{fmtDuration(last.seconds)}</span>
            </div>
          ) : (
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
          )}
          {!last.mesh && last.prunedCount < last.inputCount && (
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
          {!last.mesh && (
            <button className={`btn-sm export-share ${sharing ? "on" : ""}`} onClick={() => setSharing((v) => !v)}>
              {t.share.button}
            </button>
          )}
          {sharing && !last.mesh && (
            <SharePanel path={last.path} bytes={last.outputBytes} frame={p.frame} onClose={() => setSharing(false)} />
          )}
        </div>
      )}
    </section>
  );
}
