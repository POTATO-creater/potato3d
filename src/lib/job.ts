import { STAGES, type FrameInfo, type JobEvent, type MaskStats, type ModelStats, type Msg, type ProjectState, type Stage } from "./api";

export type StageStatus = "idle" | "queued" | "running" | "done" | "error" | "kept" | "skipped";

export interface StageState {
  status: StageStatus;
  fraction: number;
  /** Traduzida na hora de exibir (src/i18n). */
  message: Msg | null;
  seconds?: number;
}

export interface TrainProgress {
  iter: number;
  splats: number;
  /** Passos por segundo, média móvel. */
  rate: number | null;
  lastAt: number;
}

export interface JobState {
  running: boolean;
  stages: Record<Stage, StageState>;
  frames: FrameInfo[];
  masks: MaskStats | null;
  /** Pasta das máscaras + versão (cache-buster das miniaturas). */
  masksDir: { path: string; version: number } | null;
  poses: ModelStats | null;
  train: TrainProgress | null;
  evals: { iter: number; psnr: number; ssim: number }[];
  /** .ply mais recente a exibir (snapshot durante o treino, final depois). */
  splat: { path: string; iter: number | null; version: number } | null;
  finalPly: string | null;
  error: { stage: Stage | null; message: string } | null;
  totalSeconds: number | null;
}

const idle = (): StageState => ({ status: "idle", fraction: 0, message: null });

export const initialJob: JobState = {
  running: false,
  stages: { frames: idle(), masks: idle(), poses: idle(), train: idle() },
  frames: [],
  masks: null,
  masksDir: null,
  poses: null,
  train: null,
  evals: [],
  splat: null,
  finalPly: null,
  error: null,
  totalSeconds: null,
};

export type JobAction =
  /** `masksForPoses`: o recorte entra nas poses (refazê-lo invalida as poses). */
  | { type: "start"; from: Stage; to: Stage; masksForPoses: boolean }
  | { type: "event"; event: JobEvent }
  | { type: "load"; project: ProjectState }
  /** Frames recusados ou aceitos de volta (`Api.setFramesRejected`). */
  | { type: "framesEdited"; project: ProjectState };

/** Etapas cujo resultado um job de `from` a `to` descarta, as mesmas regras
 *  dos backends: as que ele roda e as seguintes que dependiam delas. */
export function discardedBy(from: Stage, to: Stage, masksForPoses: boolean): Set<Stage> {
  const f = STAGES.indexOf(from);
  const t = Math.max(f, STAGES.indexOf(to));
  const runs = (st: Stage) => STAGES.indexOf(st) >= f && STAGES.indexOf(st) <= t;
  const out = new Set<Stage>();
  if (runs("frames")) STAGES.forEach((st) => out.add(st));
  if (runs("masks")) out.add("masks");
  // Recortar de novo sem usar o recorte nas poses as reaproveita.
  if ((runs("poses") && from !== "masks") || (runs("masks") && masksForPoses)) out.add("poses");
  if (runs("train") || out.has("poses")) out.add("train");
  return out;
}

export function jobReducer(s: JobState, a: JobAction): JobState {
  switch (a.type) {
    case "load": {
      const p = a.project;
      const kept = (has: boolean): StageState =>
        has ? { status: "kept", fraction: 1, message: { key: "stage.saved" } } : idle();
      return {
        ...initialJob,
        stages: {
          frames: kept(p.frames.length > 0),
          masks: kept(!!p.masks),
          poses: kept(!!p.poses),
          train: kept(!!p.finalPly),
        },
        frames: p.frames,
        masks: p.masks,
        masksDir: p.masks && p.masksDir ? { path: p.masksDir, version: Date.now() } : null,
        poses: p.poses,
        evals: p.lastPsnr != null ? [{ iter: 0, psnr: p.lastPsnr, ssim: 0 }] : [],
        finalPly: p.finalPly,
        splat: p.finalPly ? { path: p.finalPly, iter: null, version: Date.now() } : null,
      };
    }

    case "start": {
      const from = STAGES.indexOf(a.from);
      const to = Math.max(from, STAGES.indexOf(a.to));
      const gone = discardedBy(a.from, a.to, a.masksForPoses);
      const stages = { ...s.stages };
      STAGES.forEach((st, i) => {
        if (i >= from && i <= to) stages[st] = { status: "queued", fraction: 0, message: { key: "stage.queued" } };
        else if (gone.has(st)) stages[st] = idle();
      });
      return {
        ...s,
        running: true,
        stages,
        error: null,
        totalSeconds: null,
        frames: gone.has("frames") ? [] : s.frames,
        masks: gone.has("masks") ? null : s.masks,
        masksDir: gone.has("masks") ? null : s.masksDir,
        poses: gone.has("poses") ? null : s.poses,
        ...(gone.has("train") ? { train: null, evals: [], finalPly: null } : {}),
      };
    }

    case "framesEdited": {
      const p = a.project;
      let next: JobState = { ...s, frames: p.frames };
      const outdated = (st: Stage, patch: Partial<JobState>) => {
        next = patchStage({ ...next, ...patch }, st, { status: "idle", fraction: 0, message: { key: "stage.framesChanged" }, seconds: undefined });
      };
      if (!p.masks && s.masks) outdated("masks", { masks: null, masksDir: null });
      if (!p.poses && s.poses) outdated("poses", { poses: null });
      return next;
    }

    case "event":
      return applyEvent(s, a.event);
  }
}

function patchStage(s: JobState, stage: Stage, patch: Partial<StageState>): JobState {
  return { ...s, stages: { ...s.stages, [stage]: { ...s.stages[stage], ...patch } } };
}

function applyEvent(s: JobState, e: JobEvent): JobState {
  switch (e.type) {
    case "stageStarted":
      return patchStage(s, e.stage, { status: "running", fraction: 0, message: { key: "stage.starting" } });
    case "progress":
      return patchStage(s, e.stage, { status: "running", fraction: e.fraction, message: e.message });
    case "log":
      return s;
    case "framesReady":
      return { ...s, frames: e.frames };
    case "masksReady":
      return { ...s, masks: e.stats, masksDir: { path: e.dir, version: Date.now() } };
    case "stageSkipped": {
      // Poses reaproveitadas continuam valendo; recorte desativado some.
      const next = patchStage(s, e.stage, { status: "skipped", fraction: 1, message: e.reason });
      return e.stage === "masks" ? { ...next, masks: null, masksDir: null } : next;
    }
    case "posesReady":
      return { ...s, poses: e.stats };
    case "stageDone":
      return patchStage(s, e.stage, { status: "done", fraction: 1, seconds: e.seconds });
    case "finished":
      return {
        ...s,
        running: false,
        totalSeconds: e.seconds,
        // Sem ply: o job parou antes do treino.
        ...(e.ply ? { finalPly: e.ply, splat: { path: e.ply, iter: null, version: Date.now() } } : {}),
      };
    case "failed": {
      const next = { ...s, running: false, error: { stage: e.stage, message: e.error } };
      return clearQueued(e.stage ? patchStage(next, e.stage, { status: "error", message: { key: "stage.failed" } }) : next);
    }
    case "cancelled": {
      const running = STAGES.find((st) => s.stages[st].status === "running");
      const next = { ...s, running: false };
      return clearQueued(running ? patchStage(next, running, { status: "idle", message: { key: "stage.cancelled" } }) : next);
    }
    case "train": {
      const t = e.event;
      if (t.kind === "step") {
        const now = performance.now();
        const prev = s.train;
        let rate = prev?.rate ?? null;
        if (prev && t.iter > prev.iter) {
          const inst = (t.iter - prev.iter) / ((now - prev.lastAt) / 1000);
          rate = rate == null ? inst : rate * 0.6 + inst * 0.4;
        }
        return { ...s, train: { iter: t.iter, splats: t.splats, rate, lastAt: now } };
      }
      if (t.kind === "eval") return { ...s, evals: [...s.evals, { iter: t.iter, psnr: t.psnr, ssim: t.ssim }] };
      return { ...s, splat: { path: t.path, iter: t.iter, version: Date.now() } };
    }
  }
}

function clearQueued(s: JobState): JobState {
  const stages = { ...s.stages };
  for (const st of STAGES) if (stages[st].status === "queued") stages[st] = idle();
  return { ...s, stages };
}
