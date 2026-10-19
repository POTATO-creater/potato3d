import { useEffect, useRef, useState } from "react";
import { api, type MaskSource, type MaskSourceOf } from "../lib/api";
import { COMFY_CHANGED, loadComfy, loadParamValues, saveParamValues, workflowsOf } from "../lib/comfy/settings";
import { paramDefault, type ParamValue } from "../lib/comfy/workflow";
import { cancelMaskGen, clearMaskGen, generateMask, useMaskGens } from "../lib/maskgen";
import { useI18n } from "../i18n";
import { comfyErrorText, ComfySettings } from "./ComfySettings";
import { Field, Select } from "./fields";
import { ParamField } from "./RetouchPanel";

const WF_KEY = "splatforge.maskWorkflow.v1";

const sizeText = (w: number, h: number) => `${w}×${h}`;

/** The item's mask video or frames (method "external"): what is attached,
 *  attach another, or generate one with a ComfyUI mask workflow. */
export function MaskSourceBox({
  projectDir,
  source,
  sourceSize,
  locked,
  onChanged,
  onCurrent,
}: {
  projectDir: string | null;
  /** Null when the source isn't open (a package without it): masks are
   *  matched to the source, so nothing can be attached or generated. */
  source: MaskSourceOf | null;
  sourceSize: [number, number] | null;
  locked: boolean;
  /** The project's mask source changed (the masks must be cut again). */
  onChanged: (projectDir: string, src: MaskSource | null) => void;
  /** What is attached now (for the preview). */
  onCurrent?: (src: MaskSource | null) => void;
}) {
  const { t, locale } = useI18n();
  const tm = t.masks.source;
  const ms = api.maskSource;
  const [current, setCurrent] = useState<MaskSource | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [viewing, setViewing] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const gens = useMaskGens();
  const gen = projectDir ? gens.get(projectDir) : undefined;
  const generating = gen?.fraction != null;

  const last = gen?.last && "saved" in gen.last ? gen.last : null;
  const lastId = last?.saved.id;
  useEffect(() => {
    setError(null);
    if (!ms || !projectDir) return setCurrent(null);
    let live = true;
    ms.get(projectDir).then(
      (s) => live && setCurrent(s),
      () => live && setCurrent(null),
    );
    return () => void (live = false);
    // A finished generation saved a new one.
  }, [ms, projectDir, lastId]);

  const report = useRef(onCurrent);
  report.current = onCurrent;
  useEffect(() => report.current?.(current), [current]);

  // Workflows change in the Integrations dialog.
  const [, setComfyVersion] = useState(0);
  useEffect(() => {
    const onChange = () => setComfyVersion((v) => v + 1);
    window.addEventListener(COMFY_CHANGED, onChange);
    return () => window.removeEventListener(COMFY_CHANGED, onChange);
  }, []);
  const workflows = workflowsOf(loadComfy(), "mask");
  const [workflowId, setWorkflowId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(WF_KEY);
    } catch {
      return null;
    }
  });
  const workflow = workflows.find((w) => w.id === workflowId) ?? workflows[0] ?? null;
  const [values, setValues] = useState<Record<string, ParamValue>>({});
  useEffect(() => setValues(workflow ? loadParamValues(workflow.id) : {}), [workflow?.id]);
  const setParam = (id: string, v: ParamValue | undefined) => {
    if (!workflow) return;
    const next: Record<string, ParamValue | undefined> = { ...values, [id]: v };
    if (v === undefined) delete next[id];
    setValues(next as Record<string, ParamValue>);
    saveParamValues(workflow.id, next);
  };
  const chooseWorkflow = (id: string) => {
    setWorkflowId(id);
    try {
      localStorage.setItem(WF_KEY, id);
    } catch {
      /* storage unavailable */
    }
  };

  if (!ms || !projectDir) return null;
  const disabled = locked || busy || generating;

  const attach = async (files: File[]) => {
    setBusy(true);
    setError(null);
    try {
      const s = await ms.attach(projectDir, files, source!);
      clearMaskGen(projectDir);
      setCurrent(s);
      onChanged(projectDir, s);
    } catch (e) {
      setError(String((e as Error)?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  /** Object URL of the mask video, revoked by the caller. */
  const videoUrl = async () => {
    const blob = await ms.video(projectDir);
    return blob && URL.createObjectURL(blob);
  };
  const view = async () => setViewing(await videoUrl());
  const download = async () => {
    const url = await videoUrl();
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = `${projectDir.split("/").pop() || "mask"}_mask.mp4`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
  };

  const remove = async () => {
    await ms.remove(projectDir);
    clearMaskGen(projectDir);
    setCurrent(null);
    onChanged(projectDir, null);
  };

  const generate = () => {
    if (!workflow || !source) return;
    setError(null);
    const dir = projectDir;
    generateMask(dir, source, workflow, values, { ...tm, queued: t.retouch.queued, running: t.retouch.running }).then(
      ({ saved }) => onChanged(dir, saved),
      () => {}, // shown from the store
    );
  };

  // Warnings about what was attached or generated.
  const warnings: string[] = [];
  if (current) {
    if (last && last.saved.id === current.id && current.count < last.sent) warnings.push(tm.fewer(current.count, last.sent));
    if (sourceSize) {
      const [w, h] = sourceSize;
      if (Math.abs(current.width / current.height / (w / h) - 1) > 0.03) warnings.push(tm.aspect(sizeText(current.width, current.height), sizeText(w, h)));
    }
    if (!current.positions && source?.kind === "images" && current.count !== source.paths.length)
      warnings.push(tm.countDiffers(current.count, source.paths.length));
  }
  const failed = gen?.last && "error" in gen.last ? comfyErrorText(t, gen.last.error) : null;

  return (
    <div className="mask-source">
      {current ? (
        <p className="help">
          <strong>{(current.kind === "video" ? tm.video : tm.frames)(current.count, sizeText(current.width, current.height))}</strong>
          <br />
          <span className="muted">
            {tm.from(current.origin, new Date(current.createdAt).toLocaleString(locale, { dateStyle: "short", timeStyle: "short" }))}
          </span>
        </p>
      ) : (
        <p className="help">{tm.none}</p>
      )}
      {!source && <p className="help muted">{tm.needSource}</p>}
      <div className="mask-source-row">
        <button className="btn-sm" disabled={disabled || !source} title={tm.attachTitle} onClick={() => fileInput.current?.click()}>
          {tm.attach}
        </button>
        {current?.kind === "video" && (
          <>
            <button className="btn-sm ghost" onClick={() => void view()}>
              {tm.view}
            </button>
            <button className="btn-sm ghost" onClick={() => void download()}>
              {tm.download}
            </button>
          </>
        )}
        {current && (
          <button className="btn-sm ghost" disabled={disabled} onClick={() => void remove()}>
            {tm.remove}
          </button>
        )}
        <input
          ref={fileInput}
          type="file"
          multiple
          accept="video/*,image/png,image/jpeg,image/webp"
          hidden
          onChange={(e) => {
            const files = [...(e.currentTarget.files ?? [])];
            e.currentTarget.value = "";
            if (files.length) void attach(files);
          }}
        />
      </div>

      {workflows.length === 0 ? (
        <p className="help muted">
          {tm.noWorkflow}{" "}
          <button className="link" onClick={() => setSettingsOpen(true)}>
            {tm.openIntegrations}
          </button>
        </p>
      ) : (
        <div className="mask-source-gen">
          {workflows.length > 1 && (
            <Field label={tm.workflow}>
              <Select value={workflow!.id} disabled={disabled} options={workflows.map((w) => ({ value: w.id, label: w.name }))} onChange={chooseWorkflow} />
            </Field>
          )}
          {workflow && !!workflow.params?.length && (
            <div className="retouch-params">
              {workflow.params.map((p) => (
                <ParamField key={p.id} param={p} fallback={paramDefault(workflow.graph, p)} value={values[p.id]} disabled={disabled} onChange={(v) => setParam(p.id, v)} />
              ))}
            </div>
          )}
          {!generating && (
            <button className="btn" disabled={disabled || !source || !workflow} onClick={generate} title={workflow?.name}>
              {current ? tm.regenerate : tm.generate}
            </button>
          )}
          {source?.kind === "video" && (source.trim.start > 0 || source.trim.end != null) && <p className="help muted">{tm.trimNote}</p>}
          {generating && gen && (
            <div className="retouch-progress">
              <div className="bar">
                <div className="bar-fill" style={{ width: `${Math.round(gen.fraction! * 100)}%` }} />
              </div>
              <div className="retouch-progress-row">
                <span className="muted small">{gen.text}</span>
                <button className="link" onClick={() => cancelMaskGen(projectDir)}>
                  {tm.cancel}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {warnings.map((w) => (
        <p key={w} className="help warn">
          {w}
        </p>
      ))}
      {(error ?? failed) && <p className="bad small">{error ?? failed}</p>}
      {settingsOpen && <ComfySettings onClose={() => setSettingsOpen(false)} />}
      {viewing && current && (
        <MaskVideoDialog
          url={viewing}
          title={`${tm.viewTitle} · ${current.origin}`}
          size={[current.width, current.height]}
          onDownload={() => void download()}
          onClose={() => {
            URL.revokeObjectURL(viewing);
            setViewing(null);
          }}
        />
      )}
    </div>
  );
}

/** The mask video in a player, looping. */
function MaskVideoDialog({ url, title, size, onDownload, onClose }: { url: string; title: string; size: [number, number]; onDownload: () => void; onClose: () => void }) {
  const { t } = useI18n();
  const tm = t.masks.source;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal" role="dialog" aria-modal="true" aria-label={title} onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-card mask-video-view">
        <header className="modal-head">
          <h2>{title}</h2>
          <button className="modal-close" onClick={onClose} aria-label={tm.close} title={`${tm.close} (Esc)`}>
            <svg viewBox="0 0 18 18" aria-hidden>
              <path d="M4.5 4.5l9 9M13.5 4.5l-9 9" />
            </svg>
          </button>
        </header>
        <video src={url} style={{ aspectRatio: `${size[0]} / ${size[1]}` }} controls autoPlay loop muted playsInline />
        <div className="mask-source-row">
          <span className="muted small">{sizeText(size[0], size[1])}</span>
          <button className="btn-sm" onClick={onDownload}>
            {tm.download}
          </button>
        </div>
      </div>
    </div>
  );
}
