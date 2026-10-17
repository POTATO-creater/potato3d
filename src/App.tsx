import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import {
  api,
  DEFAULT_CHROMA,
  host,
  IMAGE_EXT,
  PACKAGE_EXT,
  type ExportOptions,
  type ImageSet,
  type JobEvent,
  STAGES,
  type MaskOptions,
  type Settings,
  type Stage,
  type Tools,
} from "./lib/api";
import { discardedBy, initialJob } from "./lib/job";
import { emptyWorkspace, itemName, itemStatus, newItem, workspaceReducer, type BatchItem, type BulkExport } from "./lib/batch";
import { groupSources, VIDEO_EXT } from "./lib/sources";
import { fmtDuration } from "./lib/format";
import { revealLabel, stageBlurbs } from "./lib/platform";
import { ToolsSetup } from "./components/ToolsSetup";
import { PRESETS, presetById, type PresetId } from "./lib/presets";
import { Sidebar } from "./components/Sidebar";
import { BatchSelector } from "./components/BatchSelector";
import { Viewer, type ViewerHandle } from "./components/Viewer";
import { RetouchPanel } from "./components/RetouchPanel";
import { StageStrip } from "./components/StageStrip";
import { FrameLightbox, FramesGrid, LogView } from "./components/FramesGrid";
import { ExportPanel } from "./components/ExportPanel";
import { maskMethods } from "./components/MasksPanel";
import { GitHubLink } from "./components/GitHubLink";
import { LanguageSelect } from "./components/LanguageSelect";
import { ComfySettingsButton } from "./components/ComfySettings";
import { UpdateBanner } from "./components/UpdateBanner";
import { useSidebarWidth } from "./components/SidebarResizer";
import { promptInTrim } from "./lib/prompt";
import { basename } from "./lib/platform";
import { I18nProvider, useI18n } from "./i18n";
import "./styles.css";

const STORE_KEY = "splatforge.settings.v3";
const MASKS_KEY = "splatforge.masks.v1";

function loadMasks(): MaskOptions {
  const base: MaskOptions = {
    enabled: false,
    forPoses: false,
    method: "auto",
    chroma: DEFAULT_CHROMA,
    despill: true,
    despillStrength: 100,
  };
  try {
    const raw = localStorage.getItem(MASKS_KEY);
    // Merge with the defaults: older versions had no method or chroma, and
    // despill lived inside chroma.
    if (raw) {
      const { chroma: { despill, ...chroma } = {} as never, ...saved } = JSON.parse(raw);
      // The SAM 2.1 marking belongs to the previous session's video: drop it.
      return { ...base, despill: despill ?? base.despill, ...saved, chroma: { ...DEFAULT_CHROMA, ...chroma }, prompt: null };
    }
  } catch {
    /* storage unavailable */
  }
  return base;
}

type Tab = "viewer" | "frames" | "log";

function loadStored(): { preset: PresetId | "custom"; settings: Settings } {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) {
      const saved = JSON.parse(raw);
      // A named preset follows its current definition (new tweaks reach users who
      // already picked it); only "custom" keeps the saved values.
      const p = PRESETS.find((x) => x.id === saved.preset);
      return p ? { preset: p.id, settings: p.settings } : saved;
    }
  } catch {
    /* storage unavailable: use the default */
  }
  return { preset: "balanced", settings: presetById("balanced").settings };
}

/** UI settings stored in the project package (.gorbit). */
interface Session {
  settings: Settings;
  masks: MaskOptions;
  trim: { start: number; end: number | null };
}

/** A package's session merged with current defaults: it may come from another
 *  app version (with fewer fields) or from the other app (desktop ↔ web), with
 *  a masking method that does not exist here. */
function readSession(raw: unknown, current: Session, os: Tools["os"] | undefined): Session | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Partial<Session>;
  const s = r.settings;
  const m = r.masks;
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  return {
    settings: s
      ? {
          frames: { ...current.settings.frames, ...s.frames },
          poses: { ...current.settings.poses, ...s.poses },
          train: { ...current.settings.train, ...s.train },
        }
      : current.settings,
    masks: m
      ? {
          ...current.masks,
          ...m,
          method: maskMethods(os).includes(m.method) ? m.method : current.masks.method,
          chroma: { ...DEFAULT_CHROMA, ...m.chroma },
          prompt: m.prompt ?? null,
        }
      : current.masks,
    trim: { start: num(r.trim?.start) ?? 0, end: num(r.trim?.end) },
  };
}

function matchPreset(s: Settings): PresetId | "custom" {
  const json = JSON.stringify(s);
  return PRESETS.find((p) => JSON.stringify(p.settings) === json)?.id ?? "custom";
}

/** Does AI masking have what it needs on this system? */
function maskToolReady(tools: Tools, m: MaskOptions) {
  // SAM 2.1 (web only) downloads its models during masking.
  if (m.method === "sam2") return tools.os === "web";
  // On the web the other AI masker is RMBG.
  if (m.method === "rmbg" || tools.os === "web") return !!tools.rmbgModel;
  if (tools.os !== "windows") return !!tools.masker;
  return m.method === "general" ? !!tools.birefnetModel : !!tools.modnetModel;
}

export default function App() {
  return (
    <I18nProvider>
      <AppView />
    </I18nProvider>
  );
}

function AppView() {
  const { t } = useI18n();
  const stored = useRef(loadStored()).current;
  const [tools, setTools] = useState<Tools | null>(null);
  /** Inputs open in the window: one per video, frames folder or package. */
  const [ws, wsDispatch] = useReducer(workspaceReducer, emptyWorkspace);
  const wsRef = useRef(ws);
  wsRef.current = ws;
  const item = ws.items.find((it) => it.id === ws.active) ?? null;
  const activeId = item?.id ?? null;
  const activeRef = useRef(activeId);
  activeRef.current = activeId;
  /** Inputs that could not be added (no images in a folder, unreadable package…). */
  const [addError, setAddError] = useState<string | null>(null);
  const [preset, setPreset] = useState<PresetId | "custom">(stored.preset);
  /** Shared by every item in the batch. */
  const [settings, setSettings] = useState<Settings>(stored.settings);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  /** Masking for new items (the last one edited); each item then has its own. */
  const [defaultMasks, setDefaultMasks] = useState<MaskOptions>(loadMasks);
  const defaultMasksRef = useRef(defaultMasks);
  defaultMasksRef.current = defaultMasks;
  const sidebar = useSidebarWidth();
  const [tab, setTab] = useState<Tab>("viewer");
  /** Frame opened by clicking a camera in 3D (index into `job.frames`). */
  const [camFrame, setCamFrame] = useState<number | null>(null);
  /** Shows the export in 3D instead of the trained splat. */
  const [previewExport, setPreviewExport] = useState<{ version: number } | null>(null);
  const viewerRef = useRef<ViewerHandle>(null);
  /** Retouch in progress (uses the GPU: the pipeline waits). */
  const [retouchBusy, setRetouchBusy] = useState(false);
  /** Splat the retouch asks to show: fine-tuning previews or the previous one. */
  const [retouchSplat, setRetouchSplat] = useState<{ path: string; label: string; iter: number | null; version: number } | null>(null);
  /** "Process all" in progress: items already tried in this run, and whether to stop. */
  const batchRun = useRef<{ stop: boolean; tried: Set<string> } | null>(null);
  const [batchRunning, setBatchRunning] = useState(false);
  /** "Export all" progress and results. */
  const [bulk, setBulk] = useState<BulkExport | null>(null);
  const bulkAbort = useRef<AbortController | null>(null);
  /** Resolves the promise of `run` when its job ends, by item. */
  const jobEnded = useRef(new Map<string, () => void>());
  /** Project folder of each item, claimed synchronously so two inputs added
   *  together never share one. */
  const claimedDirs = useRef(new Map<string, string>());

  // The active item, or empty values before anything is opened.
  const job = item?.job ?? initialJob;
  const videoPath = item?.videoPath ?? null;
  const video = item?.video ?? null;
  const imageSet = item?.imageSet ?? null;
  const trim = item?.trim ?? { start: 0, end: null };
  const projectDir = item?.projectDir ?? null;
  const maskOpts = item?.masks ?? defaultMasks;
  const fromStage = item?.fromStage ?? "frames";
  const frame = item?.frame ?? null;
  const lastExport = item?.lastExport ?? null;
  const framesVersion = item?.framesVersion ?? 0;
  const pack = item?.pack ?? null;
  const packBusy = pack?.kind === "saving" || pack?.kind === "opening";
  const anyRunning = ws.items.some((it) => it.job.running);

  const get = (id: string) => wsRef.current.items.find((it) => it.id === id) ?? null;
  const patch = useCallback(
    (id: string, p: Partial<BatchItem> | ((it: BatchItem) => Partial<BatchItem>)) => wsDispatch({ type: "patch", id, patch: p }),
    [],
  );

  // Logs live in a ref (per item) and only re-render every 250 ms.
  const logs = useRef(new Map<string, string[]>());
  const [logLines, setLogLines] = useState<string[]>([]);
  const logTimer = useRef<number | null>(null);
  const showLogs = useCallback(() => {
    const id = activeRef.current;
    setLogLines(id ? [...(logs.current.get(id) ?? [])] : []);
  }, []);
  const pushLog = useCallback(
    (id: string, line: string) => {
      let lines = logs.current.get(id);
      if (!lines) logs.current.set(id, (lines = []));
      lines.push(line);
      if (lines.length > 5000) lines.splice(0, lines.length - 4000);
      if (id === activeRef.current && logTimer.current == null) {
        logTimer.current = window.setTimeout(() => {
          logTimer.current = null;
          showLogs();
        }, 250);
      }
    },
    [showLogs],
  );

  useEffect(() => {
    api.detectTools().then((t) => {
      setTools(t);
      // Per-OS layout tweaks (e.g. room for the macOS window buttons).
      document.documentElement.dataset.os = t.os;
    });
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({ preset, settings }));
    } catch {
      /* ignore */
    }
  }, [preset, settings]);

  useEffect(() => {
    try {
      localStorage.setItem(MASKS_KEY, JSON.stringify(defaultMasks));
    } catch {
      /* ignore */
    }
  }, [defaultMasks]);

  // The web host keeps only the latest few projects in browser storage:
  // the batch's must survive.
  const projectDirs = ws.items.map((it) => it.projectDir ?? "").join("\n");
  useEffect(() => {
    api.keepProjects?.(projectDirs.split("\n").filter(Boolean));
  }, [projectDirs]);

  // Switching items: the view shows the other item's results.
  useEffect(() => {
    setPreviewExport(null);
    setCamFrame(null);
    setRetouchSplat(null);
    showLogs();
  }, [activeId, showLogs]);

  const refreshFrame = useCallback(
    async (id: string) => {
      const dir = wsRef.current.items.find((it) => it.id === id)?.projectDir;
      if (!dir) return;
      const frame = await api.sceneFrame(dir).catch(() => null);
      patch(id, { frame });
    },
    [patch],
  );

  /** Another item already using `dir` (same name in another folder) gets a suffix. */
  function claimDir(id: string, dir: string) {
    const taken = new Set([...claimedDirs.current].filter(([k]) => k !== id).map(([, d]) => d));
    let out = dir;
    for (let n = 2; taken.has(out); n++) out = `${dir} (${n})`;
    claimedDirs.current.set(id, out);
    return out;
  }

  const openProject = useCallback(
    async (id: string, dir: string) => {
      claimedDirs.current.set(id, dir);
      patch(id, { projectDir: dir });
      const project = await api.loadProject(dir);
      // Frame and splat arrive together: the Viewer aligns the splat to the
      // current frame when it loads, and the previous project's won't do.
      const frame = project.poses ? await api.sceneFrame(dir).catch(() => null) : null;
      wsDispatch({ type: "job", id, action: { type: "load", project } });
      patch(id, {
        frame,
        lastExport: project.lastExport ?? null,
        framesVersion: Date.now(),
        fromStage: project.poses ? "train" : project.frames.length ? "poses" : "frames",
      });
      if (id === activeRef.current) setPreviewExport(null);
    },
    [patch],
  );

  function addItem(p: Partial<BatchItem>, select: boolean) {
    const it = newItem(defaultMasksRef.current, p);
    wsDispatch({ type: "add", item: it, select });
    logs.current.set(it.id, []);
    return it.id;
  }

  /** `dir`: project already chosen (opened package); otherwise the video's default. */
  async function addVideo(path: string, select: boolean, dir?: string, id?: string) {
    const same = wsRef.current.items.find((it) => it.videoPath === path);
    if (same && !id) {
      if (select) wsDispatch({ type: "select", id: same.id });
      return;
    }
    id ??= addItem({ videoPath: path }, select);
    patch(id, { videoPath: path });
    try {
      patch(id, { video: await api.probeVideo(path) });
    } catch (e) {
      patch(id, { sourceError: String(e) });
    }
    await openProject(id, dir ?? claimDir(id, await api.defaultProjectDir(path)));
  }

  /** Returns false if no image was usable. */
  async function addImages(paths: string[], select: boolean, dir?: string, id?: string) {
    let set: ImageSet;
    try {
      set = await api.probeImages(paths);
    } catch (e) {
      setAddError(t.app.imagesError(`${basename(paths[0] ?? "")}: ${e}`));
      return false;
    }
    const same = wsRef.current.items.find((it) => it.imageSet?.dir === set.dir);
    if (same && !id) {
      if (select) wsDispatch({ type: "select", id: same.id });
      return true;
    }
    id ??= addItem({ imageSet: set }, select);
    patch(id, { imageSet: set });
    await openProject(id, dir ?? claimDir(id, await api.defaultProjectDir(set.dir)));
    return true;
  }

  /** Project saved as .gorbit: extracts into a new project and adds it with the
   *  source and settings it was saved with, like a cached project. */
  async function addPackage(path: string, select: boolean) {
    const alone = wsRef.current.items.length === 0;
    const id = addItem({ label: basename(path), pack: { kind: "opening", fraction: 0 } }, select);
    try {
      const r = await api.unpackProject(path, (fraction) => patch(id, { pack: { kind: "opening", fraction } }));
      claimedDirs.current.set(id, r.projectDir);
      const restored = r.video
        ? (await addVideo(r.video, select, r.projectDir, id), true)
        : r.images
          ? await addImages(r.images, select, r.projectDir, id)
          : false;
      // No source: finished stages still hold (only frames can't be redone).
      if (!restored) await openProject(id, r.projectDir);
      const it = get(id);
      const s = readSession(r.session, { settings: settingsRef.current, masks: it?.masks ?? defaultMasksRef.current, trim: it?.trim ?? { start: 0, end: null } }, tools?.os);
      if (s) {
        // Quality is shared by the batch: a package only sets it when it is
        // the first thing opened.
        if (alone && !wsRef.current.items.some((x) => x.job.running)) {
          setSettings(s.settings);
          setPreset(matchPreset(s.settings));
        }
        patch(id, { masks: s.masks, trim: s.trim });
      }
      patch(id, { pack: null });
    } catch (e) {
      removeItem(id);
      setAddError(t.sidebar.packError(String(e)));
    }
  }

  /** Each video, package and frames folder becomes an item of the batch. */
  async function openPaths(paths: string[]) {
    const inputs = groupSources(paths, IMAGE_EXT, PACKAGE_EXT);
    if (!inputs.length) return;
    setAddError(null);
    // The first new item is shown; the others wait in the selector.
    let first = true;
    for (const s of inputs) {
      if (s.kind === "package") await addPackage(s.path, first);
      else if (s.kind === "video") await addVideo(s.path, first);
      else if (!(await addImages(s.paths, first))) continue;
      first = false;
    }
  }
  const openPathsRef = useRef(openPaths);
  openPathsRef.current = openPaths;

  // Drag and drop videos, images, folders or packages anywhere in the window:
  // they join the batch, even while it is being processed.
  useEffect(() => host.onDrop((paths) => void openPathsRef.current(paths)), []);

  function removeItem(id: string) {
    const it = get(id);
    if (!it || it.job.running) return;
    logs.current.delete(id);
    claimedDirs.current.delete(id);
    wsDispatch({ type: "remove", id });
  }

  function selectItem(id: string) {
    // The retouch panel works on the shown project.
    if (!retouchBusy) wsDispatch({ type: "select", id });
  }

  async function pickFiles() {
    openPaths(await host.pickFiles([...VIDEO_EXT, ...IMAGE_EXT, PACKAGE_EXT]));
  }

  async function pickPackage() {
    openPaths(await host.pickFiles([PACKAGE_EXT]));
  }

  /** Saves the active project as .gorbit, even mid-job: only finished stages
   *  the job is not redoing are included (the reducer has already dropped those). */
  async function savePackage() {
    if (!item || !projectDir || !job.frames.length || packBusy) return;
    const id = item.id;
    const dest = await host.pickPackageDest(projectDir);
    if (!dest) return;
    patch(id, { pack: { kind: "saving", fraction: 0 } });
    try {
      const stats = await api.packProject(
        {
          projectDir,
          dest,
          include: { masks: !!job.masks, poses: !!job.poses, splat: !!job.finalPly },
          source: videoPath ? { kind: "video", path: videoPath } : imageSet ? { kind: "images", paths: imageSet.paths } : null,
          session: { settings, masks: maskOpts, trim } satisfies Session,
        },
        (fraction) => patch(id, { pack: { kind: "saving", fraction } }),
      );
      patch(id, { pack: { kind: "saved", name: basename(stats.path), bytes: stats.bytes } });
    } catch (e) {
      patch(id, { pack: { kind: "error", message: String(e) } });
    }
  }

  async function pickFramesFolder() {
    try {
      const paths = await host.pickFramesFolder();
      if (paths.length) openPaths(paths);
    } catch (e) {
      setAddError(t.app.imagesError(String(e)));
    }
  }

  async function pickProject() {
    if (!item) return;
    const id = item.id;
    const p = await host.pickFolder(projectDir ?? undefined);
    if (p) openProject(id, p);
  }

  function onEvent(id: string, e: JobEvent) {
    wsDispatch({ type: "job", id, action: { type: "event", event: e } });
    switch (e.type) {
      case "log":
        pushLog(id, `[${e.stage}] ${e.line}`);
        break;
      case "stageStarted":
        pushLog(id, `── ${e.stage} ──`);
        break;
      case "framesReady":
        patch(id, { framesVersion: Date.now() });
        break;
      case "posesReady":
        refreshFrame(id);
        break;
      case "failed":
        pushLog(id, t.app.logError(e.error));
        break;
    }
    if (e.type === "finished" || e.type === "failed" || e.type === "cancelled") {
      jobEnded.current.get(id)?.();
      jobEnded.current.delete(id);
    }
  }

  /** Tools missing for an item (its masking decides which model it needs). */
  function missingTools(it: BatchItem | null) {
    if (!tools) return [];
    const m = it?.masks ?? defaultMasks;
    return (
      [
        // Images don't need video decoding.
        { name: "ffmpeg", key: "ffmpeg", ok: (!!tools.ffmpeg && !!tools.ffprobe) || !!it?.imageSet },
        { name: "COLMAP", key: "colmap", ok: !!tools.colmap },
        { name: "Brush", key: "brush", ok: !!tools.brush },
        // Masking is only required when enabled.
        {
          name: tools.os === "windows" || m.method === "rmbg" ? t.app.maskModel : t.app.masker,
          key: "model",
          ok: !m.enabled || m.method === "chroma" || maskToolReady(tools, m),
        },
      ] as const
    ).filter((t) => !t.ok);
  }
  type Missing = ReturnType<typeof missingTools>;
  // Tool each stage needs (keys of `missing`).
  const toolStage: Record<Missing[number]["key"], Stage> = { ffmpeg: "frames", model: "masks", colmap: "poses", brush: "train" };
  const missingFor = (list: Missing, from: Stage, to: Stage) =>
    list.filter((m) => {
      const i = STAGES.indexOf(toolStage[m.key]);
      return i >= STAGES.indexOf(from) && i <= STAGES.indexOf(to);
    });
  /** Only the frames stage reads the source; the others use what's in the project. */
  const startableItem = (it: BatchItem, st: Stage) =>
    !!it.projectDir && (!!it.video || !!it.imageSet || (st !== "frames" && it.job.frames.length > 0));
  const itemPackBusy = (it: BatchItem) => it.pack?.kind === "saving" || it.pack?.kind === "opening";
  /** Waiting for "Process all": not trained yet and ready to start. */
  const pendingItem = (it: BatchItem) =>
    !it.job.finalPly &&
    !it.job.running &&
    !itemPackBusy(it) &&
    startableItem(it, it.fromStage) &&
    missingFor(missingTools(it), it.fromStage, "train").length === 0;

  /** Runs stages `from` through `to` (the same one to run just one) of an item;
   *  resolves when its job ends. */
  async function run(id: string, from: Stage, to: Stage = "train") {
    const it = get(id);
    // The source is only read by the frames stage (a package may lack it).
    const sourcePath = it?.videoPath ?? it?.imageSet?.dir ?? null;
    if (!it?.projectDir || (!sourcePath && from === "frames") || itemPackBusy(it)) return;
    const { masks, trim } = it;
    const s = settingsRef.current;
    const masksForPoses = masks.enabled && masks.forPoses;
    const gone = discardedBy(from, to, masksForPoses);
    if (gone.has("poses")) patch(id, { frame: null });
    wsDispatch({ type: "job", id, action: { type: "start", from, to, masksForPoses } });
    if (gone.has("train")) {
      // New training invalidates the export (the backend forgets it too).
      patch(id, { lastExport: null });
      if (id === activeRef.current) setPreviewExport(null);
    }
    if (id === activeRef.current) setTab(from === "frames" || to === "masks" ? "frames" : "viewer");
    const ended = new Promise<void>((resolve) => jobEnded.current.set(id, resolve));
    try {
      await api.startJob(
        {
          videoPath: sourcePath ?? "",
          images: it.imageSet?.paths ?? null,
          projectDir: it.projectDir,
          fromStage: from,
          toStage: to,
          frames: { ...s.frames, trimStart: trim.start, trimEnd: trim.end },
          // A marking outside the trim range (changed afterwards) no longer applies.
          masks: promptInTrim(masks.prompt, trim) ? masks : { ...masks, prompt: null },
          poses: s.poses,
          train: s.train,
        },
        (e) => onEvent(id, e),
      );
    } catch (e) {
      onEvent(id, { type: "failed", stage: null, error: String(e) });
    }
    await ended;
  }

  /** Processes, one after another, every item not trained yet (from its own
   *  start stage). Items added meanwhile join the queue; a failure moves on. */
  async function runAll() {
    if (batchRun.current || wsRef.current.items.some((it) => it.job.running)) return;
    const b = { stop: false, tried: new Set<string>() };
    batchRun.current = b;
    setBatchRunning(true);
    try {
      for (;;) {
        const next = b.stop ? null : wsRef.current.items.find((it) => !b.tried.has(it.id) && pendingItem(it));
        if (!next) break;
        b.tried.add(next.id);
        await run(next.id, next.fromStage);
      }
    } finally {
      batchRun.current = null;
      setBatchRunning(false);
    }
  }

  function cancel() {
    if (batchRun.current) batchRun.current.stop = true;
    api.cancelJob();
  }

  /** Cancels the item being processed; the queue goes on. */
  function skip() {
    api.cancelJob();
  }

  /** Items "Export all" covers: trained and not being reprocessed. */
  const exportable = (it: BatchItem) => !!it.projectDir && !!it.job.finalPly && !it.job.running;

  /** Exports every trained item in turn with the same options. Each item's
   *  SH degree is capped at the one it was trained with; a failure moves on. */
  async function exportAll(options: ExportOptions) {
    if (bulk?.running) return;
    const targets = wsRef.current.items.filter(exportable);
    if (!targets.length) return;
    const ctl = new AbortController();
    bulkAbort.current = ctl;
    const results: BulkExport["results"] = [];
    const base = { running: true, total: targets.length, results, cancelled: false };
    setBulk({ ...base, index: 0, name: itemName(targets[0]), progress: null });
    for (const [index, it] of targets.entries()) {
      if (ctl.signal.aborted) break;
      const name = itemName(it);
      setBulk({ ...base, results: [...results], index, name, progress: null });
      try {
        const info = await api.splatInfo(it.job.finalPly!).catch(() => null);
        const shDegree = options.shDegree != null && info && options.shDegree < info.shDegree ? options.shDegree : null;
        const stats = await api.exportSplat(
          it.projectDir!,
          { ...options, shDegree },
          (progress) => setBulk((b) => b && { ...b, progress }),
          ctl.signal,
        );
        patch(it.id, { lastExport: stats });
        results.push({ id: it.id, name, stats });
      } catch (e) {
        if (ctl.signal.aborted) break;
        results.push({ id: it.id, name, error: String(e) });
      }
    }
    bulkAbort.current = null;
    setBulk((b) => b && { ...b, running: false, progress: null, results: [...results], cancelled: ctl.signal.aborted });
  }

  /** Rejects or restores frames (grid context menu). */
  async function setRejected(names: string[], rejected: boolean) {
    if (!item || !projectDir || job.running || packBusy || !names.length) return;
    const id = item.id;
    try {
      const project = await api.setFramesRejected(projectDir, names, rejected);
      wsDispatch({ type: "job", id, action: { type: "framesEdited", project } });
      // The start stage can't stay on one whose prerequisite is no longer valid.
      patch(id, (it) => {
        const need: Stage | null = it.masks.enabled && !project.masks ? "masks" : !project.poses ? "poses" : null;
        return need && STAGES.indexOf(it.fromStage) > STAGES.indexOf(need) ? { fromStage: need } : {};
      });
    } catch (e) {
      onEvent(id, { type: "failed", stage: null, error: String(e) });
    }
  }

  function setMasks(m: MaskOptions) {
    // The next items added start from the last masking edited.
    setDefaultMasks({ ...m, prompt: null });
    if (!item) return;
    patch(item.id, (it) => ({
      masks: m,
      // Disabling masking must not leave "Masks" selected as the start stage.
      ...(!m.enabled && it.fromStage === "masks" ? { fromStage: "poses" as Stage } : {}),
    }));
  }

  /** Copies the active item's masking to the others (each keeps its own SAM 2.1 marking). */
  function applyMasksToAll() {
    for (const it of ws.items)
      if (it.id !== activeId && !it.job.running)
        patch(it.id, (x) => ({
          masks: { ...maskOpts, prompt: x.masks.prompt ?? null },
          ...(!maskOpts.enabled && x.fromStage === "masks" ? { fromStage: "poses" as Stage } : {}),
        }));
  }

  // After frames, switch to the 3D view when training starts.
  useEffect(() => {
    if (job.stages.train.status === "running" && tab === "frames") setTab("viewer");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.stages.train.status]);

  /** COLMAP camera → frame: the image name (00012.jpg) is the index. */
  function pickCamera(camera: number) {
    const n = frame?.frameIndices[camera];
    if (n == null) return;
    const i = job.frames.findIndex((f) => Number(basename(f.path).replace(/\.[^.]+$/, "")) === n);
    setCamFrame(i >= 0 ? i : job.frames[n] ? n : null);
  }

  const viewerSplat = retouchSplat
    ? { path: retouchSplat.path, iter: retouchSplat.iter, version: retouchSplat.version, label: retouchSplat.label || undefined }
    : previewExport && lastExport
      ? { path: lastExport.path, iter: null, version: previewExport.version, label: t.app.exported(basename(lastExport.path)) }
      : job.splat;

  const missing = missingTools(item);
  const accepted = job.frames.filter((f) => !f.rejected).length;
  const available: Record<Stage, boolean> = {
    frames: true,
    masks: accepted > 0,
    poses: accepted > 0 && (!maskOpts.enabled || !maskOpts.forPoses || !!job.masks),
    train: !!job.poses && (!maskOpts.enabled || !!job.masks),
  };
  const startable = (st: Stage) => !!item && startableItem(item, st);
  /** Why each stage can't run on its own; null = it can. */
  const blocked = Object.fromEntries(
    STAGES.map((st) => {
      const tool = missingFor(missing, st, st)[0];
      const reason = !startable(st)
        ? t.stageStrip.needsSource
        : st === "masks" && !maskOpts.enabled
          ? t.sidebar.enableMasks
          : !available[st]
            ? st === "train"
              ? t.sidebar.needsPoses
              : t.sidebar.needsFrames
            : tool
              ? t.stageStrip.needsTool(tool.name)
              : null;
      return [st, reason];
    }),
  ) as Record<Stage, string | null>;

  const blurbs = stageBlurbs(t, tools?.os);
  const sourcePath = videoPath ?? imageSet?.dir ?? null;
  const pending = ws.items.filter(pendingItem).length;
  const queued = (it: BatchItem) => batchRunning && !batchRun.current?.tried.has(it.id) && pendingItem(it);

  return (
    <div className="app" style={{ "--sidebar-w": `${sidebar.width}px` } as React.CSSProperties}>
      {sidebar.handle}
      <Sidebar
        os={tools?.os}
        batch={
          ws.items.length > 1 && (
            <BatchSelector
              items={ws.items.map((it) => ({ item: it, status: itemStatus(it, queued(it)) }))}
              active={activeId}
              onSelect={selectItem}
              onRemove={removeItem}
              disabled={retouchBusy}
            />
          )
        }
        hasItems={ws.items.length > 0}
        videoPath={videoPath}
        video={video}
        imageSet={imageSet}
        sourceKey={activeId}
        trim={trim}
        onTrim={(trim) => item && patch(item.id, { trim })}
        onPickFiles={pickFiles}
        onPickFolder={pickFramesFolder}
        projectDir={projectDir}
        onPickProject={pickProject}
        onRevealProject={() => projectDir && host.reveal(job.finalPly ?? projectDir)}
        onSavePackage={projectDir && job.frames.length ? savePackage : undefined}
        onOpenPackage={pickPackage}
        packageStatus={pack}
        preset={preset}
        onPreset={(id) => {
          setPreset(id);
          setSettings(presetById(id).settings);
        }}
        masks={maskOpts}
        onMasks={setMasks}
        onMasksToAll={ws.items.length > 1 ? applyMasksToAll : undefined}
        settings={settings}
        onSettings={(s) => {
          setSettings(s);
          setPreset(matchPreset(s));
        }}
        fromStage={fromStage}
        onFromStage={(fromStage) => item && patch(item.id, { fromStage })}
        available={available}
        running={job.running}
        anyRunning={anyRunning}
        // Reprocessing would discard a splat "Export all" may be reading.
        canRun={startable(fromStage) && !packBusy && !retouchBusy && !anyRunning && !bulk?.running && missingFor(missing, fromStage, "train").length === 0}
        onRun={() => item && run(item.id, fromStage)}
        onCancel={cancel}
        queue={
          ws.items.length > 1
            ? { pending, running: batchRunning, onRunAll: runAll, onSkip: skip, canRunAll: pending > 0 && !anyRunning && !retouchBusy }
            : undefined
        }
        exportPanel={
          item &&
          projectDir &&
          job.finalPly &&
          !job.running && (
            <>
            <ExportPanel
              // One per item: an export keeps going if another item is shown.
              key={item.id}
              os={tools?.os}
              projectDir={projectDir}
              finalPly={job.finalPly}
              lastExport={lastExport}
              onExported={(stats) => {
                patch(item.id, { lastExport: stats });
                if (activeRef.current !== item.id) return;
                // Show the result right away to compare with the original.
                setPreviewExport({ version: Date.now() });
                setTab("viewer");
              }}
              frame={frame}
              previewing={!!previewExport}
              onPreview={(on) => {
                setPreviewExport(on ? { version: Date.now() } : null);
                if (on) setTab("viewer");
              }}
              bulk={
                ws.items.length > 1
                  ? {
                      count: ws.items.filter(exportable).length,
                      state: bulk,
                      onRun: exportAll,
                      onCancel: () => bulkAbort.current?.abort(),
                      onClose: () => setBulk(null),
                    }
                  : undefined
              }
            />
            {api.retouch && job.poses && frame && (
              <RetouchPanel
                key={item.id}
                projectDir={projectDir}
                viewer={viewerRef}
                train={settings.train}
                masks={maskOpts.enabled && job.masks ? maskOpts : null}
                disabled={anyRunning || packBusy}
                onBusy={setRetouchBusy}
                onViewerSplat={(sp) => {
                  setRetouchSplat(sp && { ...sp, version: Date.now() });
                  if (sp) {
                    setPreviewExport(null);
                    setTab("viewer");
                  }
                }}
                onLog={(line) => pushLog(item.id, line)}
                onSplatChanged={async () => {
                  const project = await api.loadProject(projectDir);
                  wsDispatch({ type: "job", id: item.id, action: { type: "load", project } });
                  patch(item.id, { lastExport: project.lastExport ?? null });
                  setPreviewExport(null);
                }}
              />
            )}
            </>
          )
        }
      />

      <main className="main">
        <div className="topbar" data-tauri-drag-region>
          <nav className="tabs">
            <button className={tab === "viewer" ? "active" : ""} onClick={() => setTab("viewer")}>
              {t.app.tabs.viewer}
            </button>
            <button className={tab === "frames" ? "active" : ""} onClick={() => setTab("frames")}>
              {t.app.tabs.frames}
              {job.frames.length ? ` (${accepted})` : ""}
            </button>
            <button className={tab === "log" ? "active" : ""} onClick={() => setTab("log")}>
              {t.app.tabs.log}
            </button>
          </nav>
          {job.totalSeconds != null && (
            <span className="done-note">{t.app.readyIn(fmtDuration(job.totalSeconds))}</span>
          )}
          {job.finalPly && (
            <button className="btn-sm" onClick={() => host.reveal(job.finalPly!)}>
              {revealLabel(t, tools?.os, ".ply")}
            </button>
          )}
          {api.retouch && <ComfySettingsButton />}
          <LanguageSelect />
          <GitHubLink />
        </div>

        <UpdateBanner busy={anyRunning} />
        {tools && missing.length > 0 && (
          <ToolsSetup
            tools={tools}
            missing={[...missing]}
            rmbg={maskOpts.enabled && maskOpts.method === "rmbg"}
            onInstalled={() => api.detectTools().then(setTools)}
          />
        )}
        {addError && <div className="banner error">{addError}</div>}
        {item?.sourceError && <div className="banner error">{t.app.videoError(item.sourceError)}</div>}
        {job.error && (
          <div className="banner error">
            <strong>{job.error.stage ? t.app.failedAt(t.stages[job.error.stage]) : t.app.failed}</strong>
            <pre>{job.error.message}</pre>
          </div>
        )}

        <div className="content">
          <div className={`pane ${tab === "viewer" ? "" : "hidden"}`}>
            <Viewer
              ref={viewerRef}
              splat={viewerSplat}
              frame={frame}
              onPickCamera={job.frames.length ? pickCamera : undefined}
              placeholder={
                <div className="placeholder">
                  <h1>{t.app.hero}</h1>
                  <ol>
                    <li>
                      <strong>{t.stages.frames}</strong>: {blurbs.frames}
                    </li>
                    <li>
                      <strong>{t.stages.masks}</strong> ({t.app.optional}): {blurbs.masks}
                    </li>
                    <li>
                      <strong>{t.stages.poses}</strong>: {blurbs.poses}
                    </li>
                    <li>
                      <strong>{t.stages.train}</strong>: {blurbs.train}
                    </li>
                  </ol>
                  {!sourcePath && <p className="muted">{t.app.dropToStart}</p>}
                </div>
              }
            />
          </div>
          {camFrame != null && tab === "viewer" && job.frames[camFrame] && (
            <FrameLightbox
              frames={job.frames}
              index={camFrame}
              onIndex={setCamFrame}
              onClose={() => setCamFrame(null)}
              version={framesVersion}
              masksDir={job.masksDir}
            />
          )}
          {tab === "frames" && (
            <div className="pane scroll">
              <FramesGrid
                frames={job.frames}
                version={framesVersion}
                masksDir={job.masksDir}
                onReject={projectDir ? setRejected : undefined}
                locked={job.running || packBusy}
              />
            </div>
          )}
          {tab === "log" && (
            <div className="pane scroll">
              <LogView lines={logLines} />
            </div>
          )}
        </div>

        <StageStrip
          job={job}
          totalSteps={settings.train.totalSteps}
          tools={tools}
          masks={maskOpts}
          images={!!imageSet}
          blocked={blocked}
          busy={anyRunning || retouchBusy || !!bulk?.running}
          onRunStage={(st) => item && run(item.id, st, st)}
          onCancel={cancel}
        />
      </main>
    </div>
  );
}
