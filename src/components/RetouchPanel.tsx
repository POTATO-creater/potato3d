import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject, type RefObject } from "react";
import { useI18n } from "../i18n";
import { api, fileUrl, host, type MaskOptions, type RetouchCameras, type RetouchClip, type RetouchApi, type TrainOptions } from "../lib/api";
import { ComfyClient } from "../lib/comfy/client";
import { progressFraction, type ComfyProgress } from "../lib/comfy/progress";
import { COMFY_CHANGED, loadComfy, loadParamValues, saveParamValues, workflowsOf } from "../lib/comfy/settings";
import { DEFAULT_FPS, paramDefault, withInput, withParams, workflowFps, type ParamValue, type WorkflowParam } from "../lib/comfy/workflow";
import { clipFrames, clipLens, DEFAULT_CLIP, nearestCamera, orbitDistance, orbitFrame, peakSpeed, poseStill, presetPath, recordedPath, recordedSeconds, tiltTurnPath, turnPath, viewPath, type ClipPlan, type RecordedSample, type Vec3 } from "../lib/retouch/trajectory";
import { DEFAULT_RESTORED_RATIO, restoredLimit } from "../lib/retouch/frames";
import { alphaPng, createClipEncoder, videoInfo } from "../lib/retouch/video";
import { comfyErrorText } from "./ComfySettings";
import { Field, NumField, Select, Toggle } from "./fields";
import type { ColmapPose, ViewerHandle } from "./Viewer";

const DEFAULT_STEPS = 3000;
/** Position learning rate: 10× lower than training from scratch. */
const LR_MEAN = 2e-6;
const WF_KEY = "splatforge.retouchWorkflow.v1";
/** Cap on restored frames in fine-tuning, as a fraction of the photos. */
const RATIO_KEY = "splatforge.retouchRestoredRatio.v1";
/** Last chosen duration and tilt. */
const MOTION_KEY = "splatforge.retouchMotion.v1";
const DEFAULT_TILT = 30;
/** Above this the camera moves too fast and the video model blurs
 *  (estimate; tune with experience). */
const MAX_SPEED = 30;
/** Recording limit, in seconds (the duration field's maximum). */
const MAX_RECORD = 30;
/** Most a single frame adds to the recording clock, in ms. Damping glides a
 *  fixed amount per frame, not per second: when the browser throttles frames
 *  (hitch, window in the background) each one would otherwise add its whole gap. */
const MAX_FRAME_MS = 100;

const KINDS = ["view", "orbit", "tiltOrbit", "below", "above", "orbitBelow", "orbitAbove", "wave", "record"] as const;
type Kind = (typeof KINDS)[number];
/** Moves that start from the current view: height comes from it (no tilt). */
const FROM_VIEW: Kind[] = ["view", "orbit", "tiltOrbit"];
/** What the trajectory keeps from the click: the aimed view or the middle
 *  photo. The trajectory itself is rebuilt when duration or tilt change. */
interface Draft {
  kind: Kind;
  pose: ColmapPose | null;
  /** Index of the photo closest to the view at click time. */
  nearest: number;
  /** Recorded motion: viewer camera samples and the seconds with movement. */
  recorded?: { samples: RecordedSample[]; seconds: number };
  /** Viewer anchor at click time: the moves turn around it. */
  pivot?: Vec3;
}

type Distance = "photos" | "view";

interface Motion {
  kind: Kind;
  seconds: number | null;
  tilt: number;
  /** Distance for arcs and loops: the photos' or the clicked view's. */
  distance: Distance;
  /** Long side of the reference video, in px. */
  longSide: number;
  /** Video aspect: "lens" (the photos' lens) or "W:H". */
  aspect: string;
  /** Reference video frame rate; null = the workflow's. */
  fps: number | null;
  /** Recorded motion: 0 keeps the recorded pace, 1 constant speed. */
  uniform: number;
  /** Recorded motion: 0 keeps the shake, 1 smooths the most. */
  stabilize: number;
  /** Recording: the clock only runs with a button pressed on the 3D view. */
  pressOnly: boolean;
  /** Recording: keep the set duration (the motion is fitted to it) instead
   *  of taking the recorded one. */
  fixed: boolean;
}

const FPS_OPTIONS = [12, 15, 16, 24, 25, 30, 48, 50, 60];

const ASPECTS = ["lens", "9:16", "2:3", "3:4", "1:1", "4:3", "3:2", "16:9"];
/** Width/height of a "W:H" aspect; undefined = the lens's. */
const aspectValue = (a: string) => {
  const [w, h] = a.split(":").map(Number);
  return w > 0 && h > 0 ? w / h : undefined;
};

/** Resolution options (long side); the short side follows the lens aspect,
 *  both rounded to multiples of 32. */
const LONG_SIDES = [512, 640, 768, 960, 1088, 1280, 1536, 1920];

function loadMotion(): Motion {
  try {
    return parseMotion(JSON.parse(localStorage.getItem(MOTION_KEY) ?? "null"));
  } catch {
    return parseMotion(null);
  }
}

/** Motion settings from storage or a package, with defaults for what is missing. */
function parseMotion(value: unknown): Motion {
  const raw = (value && typeof value === "object" ? value : null) as { kind?: unknown; seconds?: unknown; tilt?: unknown; distance?: unknown; longSide?: unknown; aspect?: unknown; fps?: unknown; uniform?: unknown; stabilize?: unknown; pressOnly?: unknown; fixed?: unknown } | null;
  return {
    uniform: typeof raw?.uniform === "number" ? Math.min(1, Math.max(0, raw.uniform)) : 0,
    stabilize: typeof raw?.stabilize === "number" ? Math.min(1, Math.max(0, raw.stabilize)) : 0,
    pressOnly: raw?.pressOnly === true,
    fixed: raw?.fixed === true,
    distance: raw?.distance === "view" ? "view" : "photos",
    longSide: LONG_SIDES.includes(raw?.longSide as number) ? (raw!.longSide as number) : DEFAULT_CLIP.longSide,
    fps: typeof raw?.fps === "number" && raw.fps > 0 ? raw.fps : null,
    aspect: ASPECTS.includes(raw?.aspect as string) ? (raw!.aspect as string) : "lens",
    kind: KINDS.includes(raw?.kind as Kind) ? (raw!.kind as Kind) : "view",
    seconds: typeof raw?.seconds === "number" && raw.seconds > 0 ? raw.seconds : null,
    tilt: typeof raw?.tilt === "number" && raw.tilt > 0 ? raw.tilt : DEFAULT_TILT,
  };
}

/** Panel state saved in a project package: the motion settings, the
 *  trajectory being planned (a recording included), the workflow with its
 *  field values and the fine-tuning settings. Clips are project files. */
export interface RetouchSession {
  motion: Motion;
  draft: Draft | null;
  steps: number;
  ratio: number;
  /** Restore workflow: found by id, else by name (another install). */
  workflow: { id: string; name: string; params: Record<string, ParamValue> } | null;
  /** Showing the splat before the retouch. */
  comparing: boolean;
}

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const numbers = <N extends number>(v: unknown, n: N) => (Array.isArray(v) && v.length === n && v.every(finite) ? v : null);
function parsePose(v: unknown): ColmapPose | null {
  const p = v as { qvec?: unknown; tvec?: unknown } | null;
  const qvec = numbers(p?.qvec, 4), tvec = numbers(p?.tvec, 3);
  return qvec && tvec ? { qvec: qvec as ColmapPose["qvec"], tvec: tvec as ColmapPose["tvec"] } : null;
}

function parseDraft(v: unknown): Draft | null {
  if (!v || typeof v !== "object") return null;
  const r = v as { kind?: unknown; pose?: unknown; nearest?: unknown; recorded?: { samples?: unknown; seconds?: unknown }; pivot?: unknown };
  if (!KINDS.includes(r.kind as Kind)) return null;
  const kind = r.kind as Kind;
  const pose = parsePose(r.pose);
  let recorded: Draft["recorded"];
  if (Array.isArray(r.recorded?.samples) && finite(r.recorded.seconds)) {
    const samples: RecordedSample[] = [];
    for (const x of r.recorded.samples) {
      const p = parsePose(x);
      const t = (x as { t?: unknown }).t;
      if (!p || !finite(t)) return null;
      samples.push({ t, ...p });
    }
    recorded = { samples, seconds: r.recorded.seconds };
  }
  if ((kind === "record" && !recorded) || (FROM_VIEW.includes(kind) && !pose)) return null;
  const pivot = numbers(r.pivot, 3) as Vec3 | null;
  return { kind, pose, nearest: finite(r.nearest) ? Math.max(0, Math.round(r.nearest)) : 0, recorded, pivot: pivot ?? undefined };
}

/** A package's retouch session (any app version); null if unusable. */
export function readRetouchSession(v: unknown): RetouchSession | null {
  if (!v || typeof v !== "object") return null;
  const r = v as Partial<Record<keyof RetouchSession, unknown>>;
  const w = r.workflow as { id?: unknown; name?: unknown; params?: unknown } | null | undefined;
  return {
    motion: parseMotion(r.motion),
    draft: parseDraft(r.draft),
    steps: finite(r.steps) && r.steps > 0 ? Math.round(r.steps) : DEFAULT_STEPS,
    ratio: finite(r.ratio) && r.ratio > 0 ? r.ratio : DEFAULT_RESTORED_RATIO,
    workflow:
      w && typeof w.id === "string" && typeof w.name === "string"
        ? { id: w.id, name: w.name, params: w.params && typeof w.params === "object" ? (w.params as Record<string, ParamValue>) : {} }
        : null,
    comparing: r.comparing === true,
  };
}

/** What the panel reads from storage on mount (motion, ratio, workflow and
 *  its fields) is set from the session, as if the user had just chosen it. */
function storeSession(s: RetouchSession) {
  const workflows = workflowsOf(loadComfy(), "restore");
  const wf = s.workflow && (workflows.find((w) => w.id === s.workflow!.id) ?? workflows.find((w) => w.name === s.workflow!.name));
  try {
    localStorage.setItem(MOTION_KEY, JSON.stringify(s.motion));
    localStorage.setItem(RATIO_KEY, String(s.ratio));
    if (wf) localStorage.setItem(WF_KEY, wf.id);
  } catch {
    /* storage unavailable */
  }
  if (wf) saveParamValues(wf.id, s.workflow!.params);
}
interface Task {
  /** Clip in progress (or "draft" while generating, "apply" while training). */
  target: string;
  fraction: number;
  text: string;
  abort: AbortController;
}

const pad = (i: number) => String(i).padStart(5, "0");
/** Readable aspect of a size (1080×1920 → 9:16). */
function lensRatio(w: number, h: number): string {
  const g = (a: number, b: number): number => (b ? g(b, a % b) : a);
  const d = g(Math.round(w), Math.round(h)) || 1;
  const [rw, rh] = [Math.round(w) / d, Math.round(h) / d];
  return rw <= 32 && rh <= 32 ? `${rw}:${rh}` : (w / h).toFixed(2);
}
const restoredOk = (c: RetouchClip) => !!c.restored;

export function RetouchPanel(props: {
  projectDir: string;
  viewer: RefObject<ViewerHandle | null>;
  /** Regular training settings, reused for fine-tuning. */
  train: TrainOptions;
  /** Project masking (null = trained without masks). */
  masks: MaskOptions | null;
  /** Another job is using the GPU. */
  disabled: boolean;
  onBusy: (busy: boolean) => void;
  /** Splat to show in 3D instead of the project's (previews and comparison). */
  onViewerSplat: (s: { path: string; label: string; iter: number | null } | null) => void;
  /** The project splat changed (apply or undo). */
  onSplatChanged: () => void;
  /** Lines for the Log tab (Brush, dataset assembly). */
  onLog: (line: string) => void;
  /** State to resume (from a package), read once on mount. */
  initial?: RetouchSession | null;
  /** `initial` has been applied. */
  onRestored?: () => void;
  /** Filled with a function returning the panel's state, to save it. */
  sessionRef?: MutableRefObject<(() => RetouchSession) | null>;
  /** The panel's state when it closes (another item shown), to resume it. */
  onLeave?: (session: RetouchSession) => void;
}) {
  const { t, tm, locale } = useI18n();
  const tr = t.retouch;
  const rt = api.retouch as RetouchApi;
  // Draft and comparison wait for the cameras; the rest is read from storage below.
  const pending = useRef(props.initial ?? null);
  const stored = useRef(false);
  if (!stored.current) {
    stored.current = true;
    if (pending.current) storeSession(pending.current);
  }
  const [cams, setCams] = useState<RetouchCameras | null>(null);
  const [clips, setClips] = useState<RetouchClip[]>([]);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [motion, setMotion] = useState(loadMotion);
  /** Clip under the mouse (its trajectory shows in 3D). */
  const [hovered, setHovered] = useState<string | null>(null);
  const [task, setTask] = useState<Task | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [applied, setApplied] = useState<{ clips: string[]; appliedAt: number; base: string } | null>(null);
  const [comparing, setComparing] = useState(false);
  const [steps, setSteps] = useState(() => props.initial?.steps ?? DEFAULT_STEPS);
  const [ratio, setRatio] = useState(() => {
    try {
      const v = Number(localStorage.getItem(RATIO_KEY));
      return v > 0 ? v : DEFAULT_RESTORED_RATIO;
    } catch {
      return DEFAULT_RESTORED_RATIO;
    }
  });
  const updateRatio = (v: number) => {
    setRatio(v);
    try {
      localStorage.setItem(RATIO_KEY, String(v));
    } catch {
      /* storage unavailable */
    }
  };
  const [reviewing, setReviewing] = useState<RetouchClip | null>(null);
  const [workflowId, setWorkflowId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(WF_KEY);
    } catch {
      return null;
    }
  });
  /** Recording the viewer camera; `elapsed` in seconds, for the button. */
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const samples = useRef<RecordedSample[]>([]);
  const importFor = useRef<string | null>(null);
  const importInput = useRef<HTMLInputElement>(null);
  const { projectDir, viewer, onBusy, onViewerSplat, onRestored } = props;

  const reload = useCallback(async () => {
    const [list, state] = await Promise.all([rt.listClips(projectDir), api.loadProject(projectDir)]);
    setClips(list);
    setApplied(state.retouch ?? null);
  }, [projectDir, rt]);

  useEffect(() => {
    setCams(null);
    setDraft(null);
    setRecording(false);
    setError(null);
    setComparing(false);
    void reload().catch((e) => setError(String(e)));
    rt.cameras(projectDir).then((c) => {
      setCams(c);
      const p = pending.current;
      if (!p) return;
      pending.current = null;
      if (p.draft) setDraft({ ...p.draft, nearest: Math.min(p.draft.nearest, Math.max(0, c.cameras.length - 1)) });
      if (p.comparing)
        void api.loadProject(projectDir).then((s) => {
          if (!s.retouch) return;
          setComparing(true);
          onViewerSplat({ path: s.retouch.base, label: tr.beforeLabel, iter: null });
        });
      onRestored?.();
    }, (e) => setError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectDir, reload, rt]);

  useEffect(() => onBusy(!!task), [task, onBusy]);
  // On exit (project switched, panel hidden), clear the trajectory from 3D.
  useEffect(() => () => viewer.current?.showPath(null), [viewer]);

  // Video frame for the viewer's framing mask.
  useEffect(() => {
    const v = viewer.current;
    if (!v || !cams) return;
    const { size, intrinsics } = clipLens(cams.lens, motion.longSide, aspectValue(motion.aspect));
    v.setFrameGuide({ size, fovY: 2 * Math.atan(size[1] / 2 / intrinsics.fy) });
    return () => v.setFrameGuide(null);
  }, [cams, viewer, motion.longSide, motion.aspect]);

  // Settings change in the Integrations dialog: reload on every change.
  const [, setComfyVersion] = useState(0);
  useEffect(() => {
    const onChange = () => setComfyVersion((v) => v + 1);
    window.addEventListener(COMFY_CHANGED, onChange);
    return () => window.removeEventListener(COMFY_CHANGED, onChange);
  }, []);
  const restoreWorkflows = workflowsOf(loadComfy(), "restore");
  const workflow = restoreWorkflows.find((w) => w.id === workflowId) ?? restoreWorkflows[0] ?? null;
  // Reference video FPS: the chosen one or, if none, the workflow's.
  const workflowRate = workflow ? workflowFps(workflow.graph) : DEFAULT_FPS;
  const fps = motion.fps ?? workflowRate;
  // Custom field values of the chosen workflow.
  const [paramValues, setParamValues] = useState<Record<string, ParamValue>>({});
  useEffect(() => setParamValues(workflow ? loadParamValues(workflow.id) : {}), [workflow?.id]);
  const setParam = (id: string, v: ParamValue | undefined) => {
    if (!workflow) return;
    const next: Record<string, ParamValue | undefined> = { ...paramValues, [id]: v };
    if (v === undefined) delete next[id];
    setParamValues(next as Record<string, ParamValue>);
    saveParamValues(workflow.id, next);
  };
  const seconds = motion.seconds ?? Math.round((DEFAULT_CLIP.frames / fps) * 10) / 10;
  const frames = clipFrames(seconds, fps);
  const busy = !!task || props.disabled || recording;

  const { sessionRef } = props;
  const session = (): RetouchSession => {
    if (pending.current) return pending.current;
    // Mid-recording, what has been recorded so far is the trajectory.
    const recordedSoFar = recording && recordedSeconds(samples.current);
    return {
      // Like Stop: the clip lasts what was recorded, unless the duration is fixed.
      motion: { ...motion, seconds: recordedSoFar && !motion.fixed ? Math.min(MAX_RECORD, Math.max(0.5, Math.round(recordedSoFar * 10) / 10)) : seconds },
      draft: recordedSoFar ? { kind: "record", pose: null, nearest: 0, recorded: { samples: [...samples.current], seconds: recordedSoFar } } : draft,
      steps,
      ratio,
      workflow: workflow ? { id: workflow.id, name: workflow.name, params: paramValues } : null,
      comparing,
    };
  };
  const sessionNow = useRef(session);
  sessionNow.current = session;
  useEffect(() => {
    if (!sessionRef) return;
    sessionRef.current = () => sessionNow.current();
    return () => void (sessionRef.current = null);
  }, [sessionRef]);
  const leave = useRef(props.onLeave);
  leave.current = props.onLeave;
  useEffect(() => () => leave.current?.(sessionNow.current()), []);

  const updateMotion = (patch: Partial<Motion>) => {
    const next = { ...motion, seconds, ...patch };
    setMotion(next);
    try {
      localStorage.setItem(MOTION_KEY, JSON.stringify(next));
    } catch {
      /* storage unavailable */
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

  // ---------------------------------------------------------- trajectory --

  const plan = (kind: Kind = motion.kind) => {
    const v = viewer.current;
    if (!cams || !v) return;
    const pose = v.viewPose();
    if (FROM_VIEW.includes(kind) && !pose) return;
    const pivot = v.anchorPoint() ?? undefined;
    const nearest = pose ? nearestCamera(cams.cameras, pose, pivot ?? orbitFrame(cams.cameras).center) : 0;
    setDraft({ kind, pose, nearest, pivot });
  };

  // Recording: samples the viewer camera every frame until Stop (or the limit).
  const startRecording = () => {
    if (!viewer.current?.viewPose()) return;
    samples.current = [];
    setDraft(null);
    setError(null);
    setElapsed(0);
    setRecording(true);
  };
  const stopRecording = () => {
    setRecording(false);
    const recorded = samples.current;
    samples.current = [];
    const s = recordedSeconds(recorded);
    if (!s) {
      setError(tr.recordStill);
      return;
    }
    // The clip lasts what was recorded (unless the duration is fixed); changing
    // the duration afterwards speeds up or slows down the same motion.
    if (!motion.fixed) updateMotion({ seconds: Math.min(MAX_RECORD, Math.max(0.5, Math.round(s * 10) / 10)) });
    setDraft({ kind: "record", pose: null, nearest: 0, recorded: { samples: recorded, seconds: s } });
  };
  const stopRef = useRef(stopRecording);
  stopRef.current = stopRecording;
  const pressOnly = motion.pressOnly;
  useEffect(() => {
    if (!recording) return;
    let raf = 0;
    let prev: { t: number; pose: ColmapPose } | null = null;
    // Recording clock, in ms: with `pressOnly` it stands still while no button
    // is pressed on the 3D view and the camera is at rest. It keeps running
    // while the camera still glides after the release (damping), otherwise the
    // video would jump to where the camera stopped.
    let clock = 0;
    let shown = 0;
    const tick = (t: number) => {
      const v = viewer.current;
      const pose = v?.viewPose();
      if (pose) {
        const runs = !pressOnly || v!.pointerDown() || (prev !== null && !poseStill(prev.pose, pose));
        if (prev && runs) clock += Math.min(t - prev.t, MAX_FRAME_MS);
        // Paused frames are left out: the next sample continues the same time.
        if (!prev || runs) samples.current.push({ t: clock, ...pose });
        prev = { t, pose };
      }
      const s = clock / 1000;
      // Button text at 10 Hz: no need to redraw the panel every frame.
      if (s - shown >= 0.1) {
        shown = s;
        setElapsed(s);
      }
      if (s >= MAX_RECORD) stopRef.current();
      else raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [recording, viewer, pressOnly]);

  // Draft trajectory, rebuilt whenever duration or tilt changes.
  const tilt = motion.tilt;
  const distance = motion.distance;
  const longSide = motion.longSide;
  const aspect = aspectValue(motion.aspect);
  const uniform = motion.uniform;
  const stabilize = motion.stabilize;
  const draftPlan = useMemo((): ClipPlan | null => {
    if (!draft || !cams || (FROM_VIEW.includes(draft.kind) && !draft.pose)) return null;
    const { cameras, lens } = cams;
    const m = cameras.length;
    const pivot = draft.pivot;
    const clip = { ...DEFAULT_CLIP, frames, longSide, aspect, pivot };
    // Arcs and loops at the photos' distance or the clicked view's.
    const radius = distance === "view" && draft.pose ? orbitDistance(cameras, draft.pose, pivot) : undefined;
    switch (draft.kind) {
      case "record":
        return draft.recorded
          ? recordedPath(draft.recorded.samples, lens, { ...clip, uniform, stabilize, center: orbitFrame(cameras).center })
          : null;
      case "view":
        // One sixth of the orbit end to end; the aimed view sits in the middle.
        return viewPath(cameras, lens, draft.pose!, { ...clip, span: Math.max(6, Math.round(m / 6)), sweepDeg: 12 });
      case "orbit":
        // Full loop at the view's height, starting at it; distance is the chosen
        // one (the view's or the photos').
        return turnPath(cameras, lens, draft.pose!, {
          ...clip,
          radius: distance === "photos" ? orbitDistance(cameras, cameras[draft.nearest], pivot) : undefined,
        });
      case "tiltOrbit":
        // Straight climb from the photos' height (well-covered reference frame)
        // to the view, a loop at its tilt, then back down.
        return tiltTurnPath(cameras, lens, draft.pose!, {
          ...clip,
          radius: distance === "photos" ? orbitDistance(cameras, cameras[draft.nearest], pivot) : undefined,
        });
      case "below":
      case "above":
        // Arc of one third of the orbit around the nearest photo.
        return presetPath(cameras, lens, { ...clip, radius, center: draft.nearest, span: Math.round(m / 3), elevDeg: draft.kind === "below" ? -tilt : tilt });
      case "orbitBelow":
      case "orbitAbove":
      case "wave":
        // Full loop: starts and ends at the photo nearest the view. The ramp is
        // shorter than for arcs so the loop spends more time at the new height.
        // The wave needs no ramp: the sine already passes smoothly through the
        // point where end meets start (a ramp there would create a step).
        return presetPath(cameras, lens, {
          ...clip,
          radius,
          ramp: draft.kind === "wave" ? 0 : 0.25,
          center: draft.nearest + m / 2,
          span: m,
          elevDeg: draft.kind === "orbitAbove" ? tilt : -tilt,
          shape: draft.kind === "wave" ? "wave" : "arc",
        });
    }
  }, [draft, cams, frames, tilt, distance, longSide, aspect, uniform, stabilize]);
  // Recorded motions may zoom: the peak counts travel too (presets keep
  // their distance, so rotation alone measures them).
  const speed = draftPlan && cams ? peakSpeed(draftPlan.poses, fps, 3, draft?.kind === "record" ? orbitFrame(cams.cameras).center : undefined) : 0;

  // Trajectory in 3D: the draft's or, with no draft, the hovered clip's.
  // Derived from state, not mouseenter/mouseleave: a card that disappears under
  // the cursor (deleted clip) fires no mouseleave and would leave it stuck.
  const hoveredClip = clips.find((c) => c.id === hovered) ?? null;
  const shown = draftPlan
    ? { poses: draftPlan.poses, size: draftPlan.size, fy: draftPlan.intrinsics.fy }
    : hoveredClip
      ? { poses: hoveredClip.poses, size: hoveredClip.size, fy: hoveredClip.intrinsics.fy }
      : null;
  const shownPoses = shown?.poses ?? null;
  const shownSize = shown?.size.join("x");
  const shownFy = shown?.fy;
  useEffect(() => {
    const [w, h] = shownSize ? shownSize.split("x").map(Number) : [0, 0];
    viewer.current?.showPath(shownPoses, shownFy ? { size: [w, h], fy: shownFy } : undefined);
  }, [shownPoses, shownSize, shownFy, viewer]);

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
        label: tr.labels[kind](tilt) + (draft.pivot ? tr.anchored : ""),
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
    });

  // ------------------------------------------------------------- ComfyUI --

  /** Checks the restored video and returns its size. Frame count must match;
   *  so must the aspect (size may change: fine-tuning scales the intrinsics),
   *  otherwise frames no longer correspond to poses. */
  const checkRestored = async (clip: RetouchClip, video: Blob): Promise<[number, number]> => {
    const info = await videoInfo(video);
    if (info.frames !== clip.poses.length) throw new Error(tr.frameMismatch(info.frames, clip.poses.length));
    const want = clip.size[0] / clip.size[1];
    const got = info.width / info.height;
    if (Math.abs(got / want - 1) > 0.02) throw new Error(tr.aspectMismatch(`${info.width}×${info.height}`, `${clip.size[0]}×${clip.size[1]}`));
    return [info.width, info.height];
  };

  const send = (clip: RetouchClip) =>
    workflow &&
    run(clip.id, async ({ abort }, progress) => {
      const settings = loadComfy();
      const client = new ComfyClient(settings.url);
      progress(0, tr.uploading);
      const reference = await host.readBlob(clip.reference!);
      const name = await client.upload(reference, `gorbit_${clip.id}.mp4`, abort.signal);
      const titles = Object.fromEntries(Object.entries(workflow.graph).map(([id, n]) => [id, n._meta?.title || n.class_type]));
      const files = await client.run(
        withParams(withInput(workflow.graph, workflow.io, name), workflow.params ?? [], paramValues),
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
      const restoredSize = await checkRestored(clip, restored);
      const saved = await rt.saveClip(projectDir, { ...clip, restoredSize, restoredWith: { workflow: workflow.name, at: Date.now() } }, { "restored.mp4": restored });
      setClips((cs) => cs.map((c) => (c.id === saved.id ? saved : c)));
    });

  const importRestored = (clip: RetouchClip, file: File) =>
    run(clip.id, async () => {
      const restoredSize = await checkRestored(clip, file);
      const saved = await rt.saveClip(projectDir, { ...clip, restoredSize, restoredWith: { workflow: file.name, at: Date.now() } }, { "restored.mp4": file });
      setClips((cs) => cs.map((c) => (c.id === saved.id ? saved : c)));
    });

  const remove = async (clip: RetouchClip) => {
    await rt.deleteClip(projectDir, clip.id);
    setClips((cs) => cs.filter((c) => c.id !== clip.id));
  };

  // ----------------------------------------------------------------- apply --

  const ready = clips.filter(restoredOk);
  const restoredTotal = ready.reduce((n, c) => n + c.poses.length, 0);
  const photos = cams?.cameras.length ?? 0;
  const restoredMax = photos ? restoredLimit(photos, ratio) : restoredTotal;
  const restoredUsed = Math.min(restoredTotal, Math.max(restoredMax, ready.length));
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
          { steps, lrMean: LR_MEAN, train: props.train, masks: props.masks, maxRestored: restoredMax },
          (e) => {
            if (e.kind === "log") props.onLog(`[retouch] ${e.line}`);
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

  // -------------------------------------------------------------------- UI --

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
            // With a trajectory open, switch the move but keep the clicked view
            // (a recording has no clicked view, and a preset has no recording).
            if (draft) setDraft(kind === "record" || draft.kind === "record" ? null : { ...draft, kind });
          }}
        />
        {motion.kind === "record" ? (
          <button
            className={`btn-sm ${recording ? "on" : ""}`}
            disabled={recording ? false : busy || !cams}
            onClick={recording ? stopRecording : startRecording}
            title={tr.recordTitle}
          >
            {recording ? tr.stop(elapsed) : tr.record}
          </button>
        ) : (
          <button className="btn-sm" disabled={busy || !cams} onClick={() => plan()} title={tr.plotTitle}>
            {tr.plot}
          </button>
        )}
      </div>
      {recording && <p className="help retouch-recording">{tr.recording(MAX_RECORD)}</p>}
      <div className="grid2 retouch-motion">
        <NumField
          label={tr.duration}
          hint={draft?.recorded ? tr.recordedHint(frames, draft.recorded.seconds, seconds) : tr.durationHint(frames)}
          value={seconds}
          min={0.5}
          max={30}
          step={0.5}
          suffix="s"
          disabled={busy}
          onChange={(v) => v > 0 && updateMotion({ seconds: v })}
        />
        <Field label={tr.fps} hint={tr.fpsHint(workflowRate)}>
          <Select
            value={fps}
            aria-label={tr.fps}
            disabled={busy}
            options={[...new Set([...FPS_OPTIONS, workflowRate])]
              .sort((x, y) => x - y)
              .map((r) => ({ value: r, label: `${r} fps`, hint: r === workflowRate ? tr.fpsWorkflow : undefined }))}
            onChange={(r) => updateMotion({ fps: r === workflowRate ? null : r })}
          />
        </Field>
      </div>
      {motion.kind === "record" ? (
        <>
          <div className="grid2">
            <NumField
              label={tr.uniform}
              hint={tr.uniformHint}
              value={Math.round(uniform * 100)}
              min={0}
              max={100}
              step={10}
              suffix="%"
              disabled={busy}
              onChange={(v) => updateMotion({ uniform: Math.min(1, Math.max(0, v / 100)) })}
            />
            <NumField
              label={tr.stabilize}
              hint={tr.stabilizeHint}
              value={Math.round(stabilize * 100)}
              min={0}
              max={100}
              step={10}
              suffix="%"
              disabled={busy}
              onChange={(v) => updateMotion({ stabilize: Math.min(1, Math.max(0, v / 100)) })}
            />
          </div>
          <Toggle label={tr.pressOnly} hint={tr.pressOnlyHint} checked={motion.pressOnly} disabled={busy} onChange={(v) => updateMotion({ pressOnly: v })} />
          <Toggle label={tr.fixed} hint={tr.fixedHint(seconds)} checked={motion.fixed} disabled={busy} onChange={(v) => updateMotion({ fixed: v })} />
        </>
      ) : (
        <div className="grid2">
          <NumField
            label={tr.tilt}
            hint={tr.tiltHint}
            value={tilt}
            min={5}
            max={80}
            step={5}
            suffix="°"
            disabled={busy || FROM_VIEW.includes(motion.kind)}
            onChange={(v) => v > 0 && updateMotion({ tilt: v })}
          />
          <Field label={tr.distance} hint={motion.kind === "view" ? tr.distanceViewHint : tr.distanceHint}>
            <Select
              value={motion.kind === "view" ? "view" : distance}
              aria-label={tr.distance}
              disabled={busy || motion.kind === "view"}
              options={[
                { value: "photos" as const, label: tr.distances.photos },
                { value: "view" as const, label: tr.distances.view },
              ]}
              onChange={(d) => updateMotion({ distance: d })}
            />
          </Field>
        </div>
      )}
      <div className="grid2">
        <Field label={tr.aspect} hint={tr.aspectHint}>
          <Select
            value={motion.aspect}
            aria-label={tr.aspect}
            disabled={busy || !cams}
            options={ASPECTS.map((a) => ({ value: a, label: a === "lens" ? cams ? tr.aspectLens(lensRatio(cams.lens.width, cams.lens.height)) : tr.aspectLens("") : a }))}
            onChange={(a) => updateMotion({ aspect: a })}
          />
        </Field>
        <Field label={tr.resolution} hint={tr.resolutionHint}>
          <Select
            value={longSide}
            aria-label={tr.resolution}
            disabled={busy || !cams}
            options={LONG_SIDES.map((ls) => {
              const [w, h] = cams ? clipLens(cams.lens, ls, aspect).size : [ls, ls];
              return { value: ls, label: `${w}×${h}` };
            })}
            onChange={(ls) => updateMotion({ longSide: ls })}
          />
        </Field>
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
                /* storage unavailable */
              }
            }}
          />
        </div>
      )}

      {draft && draftPlan && (
        <div className="retouch-draft">
          <span className="small">
            {tr.labels[draft.kind](tilt)}
            {draft.pivot && tr.anchored} · {tr.draft(draftPlan.poses.length, draftPlan.poses.length / fps, draftPlan.size[0], draftPlan.size[1], Math.round(speed))}
          </span>
          {speed > MAX_SPEED && <span className="help warn">{tr.tooFast(Math.round(speed), Math.ceil((seconds * speed) / MAX_SPEED), MAX_SPEED)}</span>}
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
              onClick={() => setDraft(null)}
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
        <div key={c.id} className="retouch-clip" onMouseEnter={() => setHovered(c.id)} onMouseLeave={() => setHovered((h) => (h === c.id ? null : h))}>
          <div className="retouch-clip-head">
            <span className="retouch-clip-name">{c.label}</span>
            <span className={`retouch-chip ${c.restored ? "on" : ""}`}>{c.restored ? tr.status.restored : tr.status.reference}</span>
          </div>
          <span className="muted small">
            {new Date(c.createdAt).toLocaleString(locale, { dateStyle: "short", timeStyle: "short" })} · {c.poses.length} frames · {c.size[0]}×{c.size[1]}
            {c.restoredWith && ` · ${tr.restoredWith(c.restoredWith.workflow)}`}
            {c.restoredSize && (c.restoredSize[0] !== c.size[0] || c.restoredSize[1] !== c.size[1]) && ` · ${tr.restoredSize(`${c.restoredSize[0]}×${c.restoredSize[1]}`)}`}
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
      {workflow && !!workflow.params?.length && (
        <>
          <h3 className="retouch-sub">{tr.workflowParams(workflow.name)}</h3>
          <div className="retouch-params">
            {workflow.params.map((p) => (
              <ParamField
                key={p.id}
                param={p}
                fallback={paramDefault(workflow.graph, p)}
                value={paramValues[p.id]}
                disabled={busy}
                onChange={(v) => setParam(p.id, v)}
              />
            ))}
          </div>
        </>
      )}

      {clips.length > 0 && (
        <div className="retouch-apply">
          <div className="grid2">
            <NumField label={tr.steps} value={steps} min={500} max={20000} step={500} disabled={busy} onChange={setSteps} />
            <NumField
              label={tr.restoredRatio}
              hint={tr.restoredRatioHint(restoredUsed, restoredTotal, photos)}
              value={Math.round(ratio * 100)}
              min={5}
              max={300}
              step={5}
              suffix="%"
              disabled={busy}
              onChange={(v) => v > 0 && updateRatio(v / 100)}
            />
          </div>
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

/** Editor for a custom workflow field, typed like the original value.
 *  `fallback` is the workflow's value (default); undefined = field is gone. */
export function ParamField(props: {
  param: WorkflowParam;
  fallback: ParamValue | undefined;
  value: ParamValue | undefined;
  disabled: boolean;
  onChange: (v: ParamValue | undefined) => void;
}) {
  const { t } = useI18n();
  const tr = t.retouch;
  const { param, fallback } = props;
  const name = param.label.trim() || param.field;
  if (fallback === undefined) return <p className="help warn">{tr.paramMissing(name)}</p>;
  // Saved value of another type (the workflow changed): use the default.
  const value = props.value !== undefined && typeof props.value === typeof fallback ? props.value : fallback;
  const changed = value !== fallback;
  const reset = changed && (
    <button className="link small" disabled={props.disabled} onClick={() => props.onChange(undefined)}>
      {tr.paramReset}
    </button>
  );
  if (typeof fallback === "boolean") {
    return (
      <div className="retouch-param">
        <Toggle label={name} checked={value as boolean} disabled={props.disabled} onChange={(v) => props.onChange(v)} />
        {reset}
      </div>
    );
  }
  const long = typeof fallback === "string" && (fallback.length > 60 || fallback.includes("\n"));
  return (
    <div className="retouch-param">
      <div className="retouch-param-head">
        <span className="field-label">{name}</span>
        {reset}
      </div>
      {typeof fallback === "number" ? (
        <input
          className="input num"
          type="number"
          step="any"
          value={value as number}
          disabled={props.disabled}
          onChange={(e) => {
            const v = e.currentTarget.valueAsNumber;
            if (!Number.isNaN(v)) props.onChange(v);
          }}
        />
      ) : long ? (
        <textarea className="input retouch-param-text" rows={4} value={value as string} disabled={props.disabled} onChange={(e) => props.onChange(e.currentTarget.value)} />
      ) : (
        <input className="input" value={value as string} disabled={props.disabled} onChange={(e) => props.onChange(e.currentTarget.value)} />
      )}
    </div>
  );
}

/** Reference and restored side by side, on the same frame. */
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

  // Same frame index in both videos (the restored one may have another fps).
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
