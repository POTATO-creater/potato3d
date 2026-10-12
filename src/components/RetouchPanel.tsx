import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { useI18n } from "../i18n";
import { api, fileUrl, host, type MaskOptions, type RetouchCameras, type RetouchClip, type RetouchApi, type TrainOptions } from "../lib/api";
import { ComfyClient } from "../lib/comfy/client";
import { progressFraction, type ComfyProgress } from "../lib/comfy/progress";
import { loadComfy, workflowsOf } from "../lib/comfy/settings";
import { DEFAULT_FPS, effectiveIO, withInput } from "../lib/comfy/workflow";
import { clipFrames, clipLens, DEFAULT_CLIP, nearestCamera, orbitFrame, presetPath, viewPath, type ClipPlan } from "../lib/retouch/trajectory";
import { alphaPng, createClipEncoder, videoInfo } from "../lib/retouch/video";
import { comfyErrorText } from "./ComfySettings";
import { NumField, Select } from "./fields";
import type { ColmapPose, ViewerHandle } from "./Viewer";

const DEFAULT_STEPS = 3000;
/** Taxa de aprendizado das posições: 10× menor que a do treino do zero. */
const LR_MEAN = 2e-6;
const WF_KEY = "splatforge.retouchWorkflow.v1";
/** Duração e inclinação escolhidas por último. */
const MOTION_KEY = "splatforge.retouchMotion.v1";
const DEFAULT_TILT = 30;
/** Abaixo disso uma volta inteira passa rápido demais para o modelo de vídeo. */
const ORBIT_MIN_SECONDS = 8;

const KINDS = ["view", "below", "above", "orbitBelow", "orbitAbove", "wave"] as const;
type Kind = (typeof KINDS)[number];
/** O que a trajetória guarda do clique: a vista mirada ou a foto do meio. A
 *  trajetória em si é refeita quando a duração ou a inclinação mudam. */
interface Draft {
  kind: Kind;
  pose: ColmapPose | null;
  /** Índice da foto mais próxima da vista no momento do clique. */
  nearest: number;
}

interface Motion {
  kind: Kind;
  seconds: number | null;
  tilt: number;
}

function loadMotion(): Motion {
  try {
    const raw = JSON.parse(localStorage.getItem(MOTION_KEY) ?? "null") as { kind?: unknown; seconds?: unknown; tilt?: unknown } | null;
    return {
      kind: KINDS.includes(raw?.kind as Kind) ? (raw!.kind as Kind) : "view",
      seconds: typeof raw?.seconds === "number" && raw.seconds > 0 ? raw.seconds : null,
      tilt: typeof raw?.tilt === "number" && raw.tilt > 0 ? raw.tilt : DEFAULT_TILT,
    };
  } catch {
    return { kind: "view", seconds: null, tilt: DEFAULT_TILT };
  }
}
interface Task {
  /** Clipe em andamento (ou "draft" ao gerar, "apply" no treino). */
  target: string;
  fraction: number;
  text: string;
  abort: AbortController;
}

const pad = (i: number) => String(i).padStart(5, "0");
const restoredOk = (c: RetouchClip) => !!c.restored;

export function RetouchPanel(props: {
  projectDir: string;
  viewer: RefObject<ViewerHandle | null>;
  /** Ajustes do treino normal, reaproveitados no ajuste. */
  train: TrainOptions;
  /** Recorte do projeto (null = treinou sem recorte). */
  masks: MaskOptions | null;
  /** Outro processamento usa a GPU. */
  disabled: boolean;
  onBusy: (busy: boolean) => void;
  /** Splat a mostrar no 3D no lugar do do projeto (prévias e comparação). */
  onViewerSplat: (s: { path: string; label: string; iter: number | null } | null) => void;
  /** O splat do projeto mudou (aplicar ou desfazer). */
  onSplatChanged: () => void;
  /** Linhas para a aba Log (Brush, montagem do dataset). */
  onLog: (line: string) => void;
}) {
  const { t, tm, locale } = useI18n();
  const tr = t.retouch;
  const rt = api.retouch as RetouchApi;
  const [cams, setCams] = useState<RetouchCameras | null>(null);
  const [clips, setClips] = useState<RetouchClip[]>([]);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [motion, setMotion] = useState(loadMotion);
  const [task, setTask] = useState<Task | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [applied, setApplied] = useState<{ clips: string[]; appliedAt: number; base: string } | null>(null);
  const [comparing, setComparing] = useState(false);
  const [steps, setSteps] = useState(DEFAULT_STEPS);
  const [reviewing, setReviewing] = useState<RetouchClip | null>(null);
  const [workflowId, setWorkflowId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(WF_KEY);
    } catch {
      return null;
    }
  });
  const importFor = useRef<string | null>(null);
  const importInput = useRef<HTMLInputElement>(null);
  const { projectDir, viewer, onBusy } = props;

  const reload = useCallback(async () => {
    const [list, state] = await Promise.all([rt.listClips(projectDir), api.loadProject(projectDir)]);
    setClips(list);
    setApplied(state.retouch ?? null);
  }, [projectDir, rt]);

  useEffect(() => {
    setCams(null);
    setDraft(null);
    setError(null);
    setComparing(false);
    void reload().catch((e) => setError(String(e)));
    rt.cameras(projectDir).then(setCams, (e) => setError(String(e)));
  }, [projectDir, reload, rt]);

  useEffect(() => onBusy(!!task), [task, onBusy]);
  // Ao sair (projeto trocado, painel some), apaga a trajetória do 3D.
  useEffect(() => () => viewer.current?.showPath(null), [viewer]);

  // Quadro do vídeo para a máscara de enquadramento do viewer.
  useEffect(() => {
    const v = viewer.current;
    if (!v || !cams) return;
    const { size, intrinsics } = clipLens(cams.lens, DEFAULT_CLIP.longSide);
    v.setFrameGuide({ size, fovY: 2 * Math.atan(size[1] / 2 / intrinsics.fy) });
    return () => v.setFrameGuide(null);
  }, [cams, viewer]);

  const restoreWorkflows = workflowsOf(loadComfy(), "restore");
  const workflow = restoreWorkflows.find((w) => w.id === workflowId) ?? restoreWorkflows[0] ?? null;
  const io = workflow ? effectiveIO(workflow.graph, workflow.io) : null;
  const fps = io?.fps ?? DEFAULT_FPS;
  const seconds = motion.seconds ?? Math.round((DEFAULT_CLIP.frames / fps) * 10) / 10;
  const frames = clipFrames(seconds, fps);
  const busy = !!task || props.disabled;

  const updateMotion = (patch: Partial<Motion>) => {
    const next = { ...motion, seconds, ...patch };
    setMotion(next);
    try {
      localStorage.setItem(MOTION_KEY, JSON.stringify(next));
    } catch {
      /* armazenamento indisponível */
    }
  };

  const run = async (target: string, fn: (task: Task, progress: (f: number, text: string) => void) => Promise<void>) => {
    const abort = new AbortController();
    const start: Task = { target, fraction: 0, text: "", abort };
    setTask(start);
    setError(null);
    try {
      await fn(start, (fraction, text) => setTask((cur) => (cur && cur.abort === abort ? { ...cur, fraction, text } : cur)));
    } catch (e) {
      if (!abort.signal.aborted) setError(comfyErrorText(t, e));
    } finally {
      setTask((cur) => (cur?.abort === abort ? null : cur));
    }
  };

  // ---------------------------------------------------------- trajetória --

  const plan = (kind: Kind = motion.kind) => {
    const v = viewer.current;
    if (!cams || !v) return;
    const pose = v.viewPose();
    if (kind === "view" && !pose) return;
    const nearest = pose ? nearestCamera(cams.cameras, pose, orbitFrame(cams.cameras).center) : 0;
    setDraft({ kind, pose, nearest });
  };

  // Trajetória do rascunho, refeita a cada mudança de duração ou inclinação.
  const tilt = motion.tilt;
  const draftPlan = useMemo((): ClipPlan | null => {
    if (!draft || !cams || (draft.kind === "view" && !draft.pose)) return null;
    const { cameras, lens } = cams;
    const m = cameras.length;
    const clip = { ...DEFAULT_CLIP, frames };
    switch (draft.kind) {
      case "view":
        // Um sexto da órbita de ponta a ponta; a vista mirada fica no meio.
        return viewPath(cameras, lens, draft.pose!, { ...clip, span: Math.max(6, Math.round(m / 6)), sweepDeg: 12 });
      case "below":
      case "above":
        // Arco de pouco menos de meia volta em volta da foto mais próxima.
        return presetPath(cameras, lens, { ...clip, center: draft.nearest, span: Math.round(m * 0.45), elevDeg: draft.kind === "below" ? -tilt : tilt });
      case "orbitBelow":
      case "orbitAbove":
      case "wave":
        // Volta inteira: começa e termina na foto mais próxima da vista.
        return presetPath(cameras, lens, {
          ...clip,
          center: draft.nearest + m / 2,
          span: m,
          elevDeg: draft.kind === "orbitAbove" ? tilt : -tilt,
          shape: draft.kind === "wave" ? "wave" : "arc",
        });
    }
  }, [draft, cams, frames, tilt]);

  useEffect(() => {
    if (draftPlan) viewer.current?.showPath(draftPlan.poses);
  }, [draftPlan, viewer]);

  const generate = () =>
    draft &&
    draftPlan &&
    run("draft", async ({ abort }, progress) => {
      const v = viewer.current!;
      const p = draftPlan;
      const kind = draft.kind;
      const clipFps = fps;
      const [w, h] = p.size;
      const n = p.poses.length;
      const enc = await createClipEncoder(w, h, clipFps);
      const files: Record<string, Blob> = {};
      try {
        await v.renderFrames({
          poses: p.poses,
          size: p.size,
          fy: p.intrinsics.fy,
          signal: abort.signal,
          onFrame: async (i, rgba) => {
            await enc.add(i, rgba);
            files[`alpha/${pad(i)}.png`] = await alphaPng(w, h, rgba);
            progress((i + 1) / (n + 1), tr.rendering(i + 1, n));
          },
        });
        progress(n / (n + 1), tr.encoding);
        files["reference.mp4"] = await enc.finish();
      } catch (e) {
        await enc.cancel();
        throw e;
      }
      const clip: RetouchClip = {
        id: `clip-${Date.now().toString(36)}`,
        label: tr.labels[kind](tilt),
        createdAt: Date.now(),
        size: p.size,
        fps: clipFps,
        intrinsics: p.intrinsics,
        poses: p.poses,
        reference: null,
        restored: null,
        restoredWith: null,
      };
      const saved = await rt.saveClip(projectDir, clip, files);
      setClips((cs) => [...cs, saved]);
      setDraft(null);
      v.showPath(null);
    });

  // ------------------------------------------------------------- ComfyUI --

  const checkFrames = async (clip: RetouchClip, video: Blob) => {
    const info = await videoInfo(video);
    if (info.frames !== clip.poses.length) throw new Error(tr.frameMismatch(info.frames, clip.poses.length));
  };

  const send = (clip: RetouchClip) =>
    workflow &&
    io &&
    run(clip.id, async ({ abort }, progress) => {
      const settings = loadComfy();
      const client = new ComfyClient(settings.url);
      progress(0, tr.uploading);
      const reference = await host.readBlob(clip.reference!);
      const name = await client.upload(reference, `gorbit_${clip.id}.mp4`, abort.signal);
      const titles = Object.fromEntries(Object.entries(workflow.graph).map(([id, n]) => [id, n._meta?.title || n.class_type]));
      const files = await client.run(
        withInput(workflow.graph, io, name, { frames: clip.poses.length, fps: clip.fps }),
        workflow.io.output.node,
        (p: ComfyProgress) => {
          const text =
            p.status === "queued"
              ? tr.queued(p.queue)
              : tr.running(p.node ? (titles[p.node] ?? p.node) : "", p.step ? `${p.step.value}/${p.step.max}` : "");
          progress(0.05 + 0.9 * progressFraction(p), text);
        },
        abort.signal,
      );
      progress(0.96, tr.downloading);
      const video = files.find((f) => /\.(mp4|webm|mov|mkv)$/i.test(f.filename)) ?? files[0];
      const restored = await client.download(video, abort.signal);
      await checkFrames(clip, restored);
      const saved = await rt.saveClip(projectDir, { ...clip, restoredWith: { workflow: workflow.name, at: Date.now() } }, { "restored.mp4": restored });
      setClips((cs) => cs.map((c) => (c.id === saved.id ? saved : c)));
    });

  const importRestored = (clip: RetouchClip, file: File) =>
    run(clip.id, async () => {
      await checkFrames(clip, file);
      const saved = await rt.saveClip(projectDir, { ...clip, restoredWith: { workflow: file.name, at: Date.now() } }, { "restored.mp4": file });
      setClips((cs) => cs.map((c) => (c.id === saved.id ? saved : c)));
    });

  const remove = async (clip: RetouchClip) => {
    await rt.deleteClip(projectDir, clip.id);
    setClips((cs) => cs.filter((c) => c.id !== clip.id));
  };

  // --------------------------------------------------------------- aplicar --

  const ready = clips.filter(restoredOk);
  const apply = () =>
    run("apply", async ({ abort }, progress) => {
      props.onViewerSplat(null);
      setComparing(false);
      const total = steps;
      const onAbort = () => void rt.cancel();
      abort.signal.addEventListener("abort", onAbort);
      try {
        await rt.apply(
          projectDir,
          ready.map((c) => c.id),
          { steps, lrMean: LR_MEAN, train: props.train, masks: props.masks },
          (e) => {
            if (e.kind === "log") props.onLog(`[retoque] ${e.line}`);
            else if (e.kind === "progress") progress(e.fraction, tm(e.message));
            else if (e.kind === "train" && e.event.kind === "step") progress(e.event.iter / total, tm({ key: "train.step", args: { iter: e.event.iter, total, splats: e.event.splats } }));
            else if (e.kind === "train" && e.event.kind === "snapshot") props.onViewerSplat({ path: e.event.path, label: "", iter: e.event.iter });
          },
        );
      } finally {
        abort.signal.removeEventListener("abort", onAbort);
        props.onViewerSplat(null);
      }
      await reload();
      props.onSplatChanged();
    });

  const undo = () =>
    run("apply", async () => {
      props.onViewerSplat(null);
      setComparing(false);
      await rt.undo(projectDir);
      await reload();
      props.onSplatChanged();
    });

  const toggleCompare = () => {
    const on = !comparing && !!applied;
    setComparing(on);
    props.onViewerSplat(on && applied ? { path: applied.base, label: tr.beforeLabel, iter: null } : null);
  };

  // ----------------------------------------------------------------- tela --

  const progressRow = (target: string) =>
    task?.target === target && (
      <div className="retouch-progress">
        <div className="bar">
          <div className="bar-fill" style={{ width: `${Math.round(task.fraction * 100)}%` }} />
        </div>
        <div className="retouch-progress-row">
          <span className="muted small">{task.text}</span>
          <button className="link" onClick={() => task.abort.abort()}>
            {tr.cancel}
          </button>
        </div>
      </div>
    );

  return (
    <section className="panel retouch">
      <header className="panel-head">
        <h2>{tr.title}</h2>
        <span className="panel-actions">
          <button className="link" onClick={() => viewer.current?.unlockAngle()} title={tr.unlockTitle}>
            {tr.unlock}
          </button>
        </span>
      </header>
      <p className="help retouch-intro">{tr.intro}</p>

      <h3 className="retouch-sub">{tr.motion}</h3>
      <div className="retouch-move">
        <Select
          value={motion.kind}
          aria-label={tr.motion}
          disabled={busy || !cams}
          options={KINDS.map((k) => ({ value: k, label: tr.moves[k].label, hint: tr.moves[k].hint }))}
          onChange={(kind) => {
            updateMotion({ kind });
            // Com uma trajetória aberta, troca o movimento mantendo a vista do clique.
            if (draft) setDraft({ ...draft, kind });
          }}
        />
        <button className="btn-sm" disabled={busy || !cams} onClick={() => plan()} title={tr.plotTitle}>
          {tr.plot}
        </button>
      </div>
      <div className="grid2 retouch-motion">
        <NumField
          label={tr.duration}
          hint={tr.durationHint(frames, fps)}
          value={seconds}
          min={0.5}
          max={30}
          step={0.5}
          suffix="s"
          disabled={busy}
          onChange={(v) => v > 0 && updateMotion({ seconds: v })}
        />
        <NumField label={tr.tilt} hint={tr.tiltHint} value={tilt} min={5} max={80} step={5} suffix="°" disabled={busy} onChange={(v) => v > 0 && updateMotion({ tilt: v })} />
      </div>
      {restoreWorkflows.length > 1 && (
        <div className="retouch-field">
          <span className="field-label">{tr.workflow}</span>
          <Select
            value={workflow?.id ?? null}
            disabled={busy}
            options={restoreWorkflows.map((w) => ({ value: w.id, label: w.name }))}
            onChange={(id) => {
              setWorkflowId(id);
              try {
                if (id) localStorage.setItem(WF_KEY, id);
              } catch {
                /* armazenamento indisponível */
              }
            }}
          />
        </div>
      )}

      {draft && draftPlan && (
        <div className="retouch-draft">
          <span className="small">
            {tr.labels[draft.kind](tilt)} · {tr.draft(draftPlan.poses.length, draftPlan.poses.length / fps, draftPlan.size[0], draftPlan.size[1])}
          </span>
          {(draft.kind === "orbitBelow" || draft.kind === "orbitAbove" || draft.kind === "wave") && seconds < ORBIT_MIN_SECONDS && (
            <span className="help warn">{tr.orbitFast(ORBIT_MIN_SECONDS)}</span>
          )}
          <div className="retouch-row">
            <button className="btn-sm" disabled={busy} onClick={generate}>
              {tr.generate}
            </button>
            <button className="btn-sm ghost" disabled={busy} onClick={() => viewer.current?.goTo(draftPlan.poses[draftPlan.poses.length >> 1])}>
              {tr.goToMiddle}
            </button>
            <button
              className="btn-sm ghost"
              disabled={busy}
              onClick={() => {
                setDraft(null);
                viewer.current?.showPath(null);
              }}
            >
              {tr.discard}
            </button>
          </div>
          {progressRow("draft")}
        </div>
      )}

      <h3 className="retouch-sub">{tr.clips}</h3>
      {clips.length === 0 && <p className="help">{tr.noClips}</p>}
      {clips.map((c) => (
        <div key={c.id} className="retouch-clip" onMouseEnter={() => !draft && viewer.current?.showPath(c.poses)} onMouseLeave={() => !draft && viewer.current?.showPath(null)}>
          <div className="retouch-clip-head">
            <span className="retouch-clip-name">{c.label}</span>
            <span className={`retouch-chip ${c.restored ? "on" : ""}`}>{c.restored ? tr.status.restored : tr.status.reference}</span>
          </div>
          <span className="muted small">
            {new Date(c.createdAt).toLocaleString(locale, { dateStyle: "short", timeStyle: "short" })} · {c.poses.length} frames · {c.size[0]}×{c.size[1]}
            {c.restoredWith && ` · ${tr.restoredWith(c.restoredWith.workflow)}`}
          </span>
          <div className="retouch-row">
            <button className="btn-sm" disabled={busy || !workflow || !c.reference} onClick={() => send(c)}>
              {c.restored ? tr.resend : tr.send}
            </button>
            <button className="btn-sm ghost" disabled={!c.reference} onClick={() => setReviewing(c)}>
              {tr.review}
            </button>
            <button
              className="btn-sm ghost"
              disabled={busy}
              title={tr.importTitle}
              onClick={() => {
                importFor.current = c.id;
                importInput.current?.click();
              }}
            >
              {tr.importRestored}
            </button>
            <button className="btn-sm ghost" disabled={busy} onClick={() => void remove(c)}>
              {tr.delete}
            </button>
          </div>
          {progressRow(c.id)}
        </div>
      ))}
      <input
        ref={importInput}
        type="file"
        accept="video/*"
        hidden
        onChange={(e) => {
          const f = e.currentTarget.files?.[0];
          e.currentTarget.value = "";
          const clip = clips.find((c) => c.id === importFor.current);
          if (f && clip) void importRestored(clip, f);
        }}
      />

      {!workflow && clips.length > 0 && <p className="help warn">{tr.noWorkflow}</p>}

      {clips.length > 0 && (
        <div className="retouch-apply">
          <NumField label={tr.steps} value={steps} min={500} max={20000} step={500} disabled={busy} onChange={setSteps} />
          <button className="btn-primary retouch-apply-btn" disabled={busy || ready.length === 0} onClick={apply}>
            {tr.apply(ready.length)}
          </button>
          <p className="help">{ready.length ? tr.applyHint : tr.needRestored}</p>
          {progressRow("apply")}
        </div>
      )}

      {applied && (
        <div className="retouch-applied">
          <span className="small good">
            {tr.applied(applied.clips.length, new Date(applied.appliedAt).toLocaleString(locale, { dateStyle: "short", timeStyle: "short" }))}
          </span>
          <div className="retouch-row">
            <button className={`btn-sm ${comparing ? "on" : ""}`} disabled={busy} onClick={toggleCompare}>
              {comparing ? tr.comparing : tr.compare}
            </button>
            <button className="btn-sm ghost" disabled={busy} onClick={undo}>
              {tr.undo}
            </button>
          </div>
        </div>
      )}

      {error && <p className="bad small retouch-error">{error}</p>}
      {reviewing && <ReviewDialog clip={reviewing} onClose={() => setReviewing(null)} />}
    </section>
  );
}

/** Referência e restaurado lado a lado, no mesmo frame. */
function ReviewDialog({ clip, onClose }: { clip: RetouchClip; onClose: () => void }) {
  const { t } = useI18n();
  const tr = t.retouch;
  const ref = useRef<HTMLVideoElement>(null);
  const res = useRef<HTMLVideoElement>(null);
  const n = clip.poses.length;
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [sizes, setSizes] = useState<{ ref?: string; res?: string }>({});
  const dims = (v: HTMLVideoElement) => `${v.videoWidth}×${v.videoHeight}`;

  // Mesmo índice de frame nos dois vídeos (o restaurado pode ter outro fps).
  const seek = useCallback(
    (i: number) => {
      for (const v of [ref.current, res.current]) {
        if (!v || !Number.isFinite(v.duration)) continue;
        v.currentTime = ((i + 0.5) / n) * v.duration;
      }
      setFrame(i);
    },
    [n],
  );

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setFrame((f) => {
      const next = (f + 1) % n;
      for (const v of [ref.current, res.current]) if (v && Number.isFinite(v.duration)) v.currentTime = ((next + 0.5) / n) * v.duration;
      return next;
    }), 1000 / clip.fps);
    return () => clearInterval(id);
  }, [playing, n, clip.fps]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal" role="dialog" aria-modal="true" aria-label={tr.reviewTitle} onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-card retouch-review">
        <header className="modal-head">
          <h2>{tr.reviewTitle}</h2>
          <button className="modal-close" onClick={onClose} aria-label={tr.close}>
            <svg viewBox="0 0 18 18" aria-hidden>
              <path d="M4.5 4.5l9 9M13.5 4.5l-9 9" />
            </svg>
          </button>
        </header>
        <div className={`retouch-review-videos ${clip.restored ? "" : "single"}`}>
          <figure>
            <div className="retouch-stage">
              <video ref={ref} src={fileUrl(clip.reference!)} style={{ aspectRatio: `${clip.size[0]} / ${clip.size[1]}` }} muted playsInline preload="auto" onLoadedMetadata={(e) => {
                  const d = dims(e.currentTarget);
                  seek(frame);
                  setSizes((z) => ({ ...z, ref: d }));
                }} />
            </div>
            <figcaption>
              {tr.referenceLabel}
              {sizes.ref && <span className="mono"> · {sizes.ref}</span>}
            </figcaption>
          </figure>
          {clip.restored && (
            <figure>
              <div className="retouch-stage">
                <video ref={res} src={fileUrl(clip.restored)} style={{ aspectRatio: `${clip.size[0]} / ${clip.size[1]}` }} muted playsInline preload="auto" onLoadedMetadata={(e) => {
                  const d = dims(e.currentTarget);
                  seek(frame);
                  setSizes((z) => ({ ...z, res: d }));
                }} />
              </div>
              <figcaption>
                {tr.restoredLabel}
                {sizes.res && <span className="mono"> · {sizes.res}</span>}
              </figcaption>
            </figure>
          )}
        </div>
        <div className="retouch-review-controls">
          <button className="btn-sm" onClick={() => setPlaying((p) => !p)}>
            {playing ? tr.pause : tr.play}
          </button>
          <input
            type="range"
            min={0}
            max={n - 1}
            value={frame}
            onChange={(e) => {
              setPlaying(false);
              seek(e.currentTarget.valueAsNumber);
            }}
          />
          <span className="mono small">{tr.frame(frame, n)}</span>
        </div>
      </div>
    </div>
  );
}
