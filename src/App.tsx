import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import {
  api,
  DEFAULT_CHROMA,
  host,
  type ExportStats,
  type JobEvent,
  type MaskOptions,
  type SceneFrame,
  type Settings,
  type Stage,
  type Tools,
  type VideoInfo,
} from "./lib/api";
import { initialJob, jobReducer } from "./lib/job";
import { fmtDuration } from "./lib/format";
import { revealLabel, stageBlurbs } from "./lib/platform";
import { ToolsSetup } from "./components/ToolsSetup";
import { PRESETS, presetById, type PresetId } from "./lib/presets";
import { Sidebar } from "./components/Sidebar";
import { Viewer } from "./components/Viewer";
import { StageStrip } from "./components/StageStrip";
import { FramesGrid, LogView } from "./components/FramesGrid";
import { ExportPanel } from "./components/ExportPanel";
import { LanguageSelect } from "./components/LanguageSelect";
import { UpdateBanner } from "./components/UpdateBanner";
import { basename } from "./lib/platform";
import { I18nProvider, useI18n } from "./i18n";
import "./styles.css";

const VIDEO_EXT = ["mp4", "mov", "m4v", "webm", "mkv", "avi"];
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
    // Mescla com o padrão: versões antigas não tinham método nem chroma, e o
    // despill ficava dentro do chroma.
    if (raw) {
      const { chroma: { despill, ...chroma } = {} as never, ...saved } = JSON.parse(raw);
      return { ...base, despill: despill ?? base.despill, ...saved, chroma: { ...DEFAULT_CHROMA, ...chroma } };
    }
  } catch {
    /* armazenamento indisponível */
  }
  return base;
}

type Tab = "viewer" | "frames" | "log";

function loadStored(): { preset: PresetId | "custom"; settings: Settings } {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    /* armazenamento indisponível: usa o padrão */
  }
  return { preset: "balanced", settings: presetById("balanced").settings };
}

function matchPreset(s: Settings): PresetId | "custom" {
  const json = JSON.stringify(s);
  return PRESETS.find((p) => JSON.stringify(p.settings) === json)?.id ?? "custom";
}

/** O recorte por IA tem o que precisa neste sistema? */
function maskToolReady(tools: Tools, m: MaskOptions) {
  // No web o único recorte por IA é o RMBG.
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
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [video, setVideo] = useState<VideoInfo | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [trim, setTrim] = useState<{ start: number; end: number | null }>({ start: 0, end: null });
  const [projectDir, setProjectDir] = useState<string | null>(null);
  const [preset, setPreset] = useState<PresetId | "custom">(stored.preset);
  const [settings, setSettings] = useState<Settings>(stored.settings);
  const [maskOpts, setMaskOpts] = useState<MaskOptions>(loadMasks);
  const [fromStage, setFromStage] = useState<Stage>("frames");
  const [frame, setFrame] = useState<SceneFrame | null>(null);
  const [tab, setTab] = useState<Tab>("viewer");
  const [job, dispatch] = useReducer(jobReducer, initialJob);
  const [framesVersion, setFramesVersion] = useState(0);
  const [lastExport, setLastExport] = useState<ExportStats | null>(null);
  /** Mostra a exportação no 3D em vez do splat do treino. */
  const [previewExport, setPreviewExport] = useState<{ version: number } | null>(null);

  // Logs ficam num ref e só re-renderizam a cada 250 ms.
  const logs = useRef<string[]>([]);
  const [logLines, setLogLines] = useState<string[]>([]);
  const logTimer = useRef<number | null>(null);
  const pushLog = useCallback((line: string) => {
    logs.current.push(line);
    if (logs.current.length > 5000) logs.current.splice(0, logs.current.length - 4000);
    if (logTimer.current == null) {
      logTimer.current = window.setTimeout(() => {
        logTimer.current = null;
        setLogLines([...logs.current]);
      }, 250);
    }
  }, []);

  useEffect(() => {
    api.detectTools().then((t) => {
      setTools(t);
      // Ajustes de layout por sistema (ex.: espaço dos botões da janela no macOS).
      document.documentElement.dataset.os = t.os;
    });
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({ preset, settings }));
    } catch {
      /* ignora */
    }
  }, [preset, settings]);

  useEffect(() => {
    try {
      localStorage.setItem(MASKS_KEY, JSON.stringify(maskOpts));
    } catch {
      /* ignora */
    }
  }, [maskOpts]);

  const refreshFrame = useCallback(async (dir: string) => {
    try {
      setFrame(await api.sceneFrame(dir));
    } catch {
      setFrame(null);
    }
  }, []);

  const openProject = useCallback(
    async (dir: string) => {
      setProjectDir(dir);
      const project = await api.loadProject(dir);
      dispatch({ type: "load", project });
      setLastExport(project.lastExport ?? null);
      setPreviewExport(null);
      setFramesVersion(Date.now());
      setFromStage(project.poses ? "train" : project.frames.length ? "poses" : "frames");
      if (project.poses) await refreshFrame(dir);
      else setFrame(null);
    },
    [refreshFrame],
  );

  const chooseVideo = useCallback(
    async (path: string) => {
      setVideoPath(path);
      setVideo(null);
      setVideoError(null);
      setTrim({ start: 0, end: null });
      logs.current = [];
      setLogLines([]);
      try {
        setVideo(await api.probeVideo(path));
      } catch (e) {
        setVideoError(String(e));
      }
      await openProject(await api.defaultProjectDir(path));
    },
    [openProject],
  );

  // Arrastar e soltar um vídeo em qualquer lugar da janela.
  useEffect(
    () =>
      host.onDrop((paths) => {
        if (job.running) return;
        const p = paths.find((x) => VIDEO_EXT.includes(x.split(".").pop()?.toLowerCase() ?? ""));
        if (p) chooseVideo(p);
      }),
    [chooseVideo, job.running],
  );

  async function pickVideo() {
    const p = await host.pickVideo(VIDEO_EXT);
    if (p) chooseVideo(p);
  }

  async function pickProject() {
    const p = await host.pickFolder(projectDir ?? undefined);
    if (p) openProject(p);
  }

  function onEvent(e: JobEvent) {
    dispatch({ type: "event", event: e });
    switch (e.type) {
      case "log":
        pushLog(`[${e.stage}] ${e.line}`);
        break;
      case "stageStarted":
        pushLog(`── ${e.stage} ──`);
        break;
      case "framesReady":
        setFramesVersion(Date.now());
        break;
      case "posesReady":
        if (projectDir) refreshFrame(projectDir);
        break;
      case "failed":
        pushLog(t.app.logError(e.error));
        break;
    }
  }

  async function run() {
    if (!videoPath || !projectDir) return;
    if (fromStage !== "train") setFrame(null);
    dispatch({ type: "start", from: fromStage });
    // Um novo treino invalida a exportação (o backend também a esquece).
    setLastExport(null);
    setPreviewExport(null);
    setTab(fromStage === "frames" ? "frames" : "viewer");
    try {
      await api.startJob(
        {
          videoPath,
          projectDir,
          fromStage,
          frames: { ...settings.frames, trimStart: trim.start, trimEnd: trim.end },
          masks: maskOpts,
          poses: settings.poses,
          train: settings.train,
        },
        onEvent,
      );
    } catch (e) {
      dispatch({ type: "event", event: { type: "failed", stage: null, error: String(e) } });
    }
  }

  // Ao terminar os frames, leva a atenção para o 3D quando o treino começa.
  useEffect(() => {
    if (job.stages.train.status === "running" && tab === "frames") setTab("viewer");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.stages.train.status]);

  const viewerSplat =
    previewExport && lastExport
      ? { path: lastExport.path, iter: null, version: previewExport.version, label: t.app.exported(basename(lastExport.path)) }
      : job.splat;

  const missing = tools
    ? (
        [
          { name: "ffmpeg", key: "ffmpeg", ok: !!tools.ffmpeg && !!tools.ffprobe },
          { name: "COLMAP", key: "colmap", ok: !!tools.colmap },
          { name: "Brush", key: "brush", ok: !!tools.brush },
          // O recorte só é exigido quando ativado.
          {
            name: tools.os === "windows" || maskOpts.method === "rmbg" ? t.app.maskModel : t.app.masker,
            key: "model",
            ok: !maskOpts.enabled || maskOpts.method === "chroma" || maskToolReady(tools, maskOpts),
          },
        ] as const
      ).filter((t) => !t.ok)
    : [];

  const available: Record<Stage, boolean> = {
    frames: true,
    masks: job.frames.length > 0,
    poses: job.frames.length > 0 && (!maskOpts.enabled || !maskOpts.forPoses || !!job.masks),
    train: !!job.poses && (!maskOpts.enabled || !!job.masks),
  };

  const blurbs = stageBlurbs(t, tools?.os);

  return (
    <div className="app">
      <Sidebar
        os={tools?.os}
        videoPath={videoPath}
        video={video}
        trim={trim}
        onTrim={setTrim}
        onPickVideo={pickVideo}
        projectDir={projectDir}
        onPickProject={pickProject}
        onRevealProject={() => projectDir && host.reveal(job.finalPly ?? projectDir)}
        preset={preset}
        onPreset={(id) => {
          setPreset(id);
          setSettings(presetById(id).settings);
        }}
        masks={maskOpts}
        onMasks={(m) => {
          setMaskOpts(m);
          // Desativar o recorte não deixa "Recorte" selecionado como ponto de partida.
          if (!m.enabled && fromStage === "masks") setFromStage("poses");
        }}
        settings={settings}
        onSettings={(s) => {
          setSettings(s);
          setPreset(matchPreset(s));
        }}
        fromStage={fromStage}
        onFromStage={setFromStage}
        available={available}
        running={job.running}
        canRun={!!videoPath && !!projectDir && !!video && missing.length === 0}
        onRun={run}
        onCancel={() => api.cancelJob()}
        exportPanel={
          projectDir &&
          job.finalPly &&
          !job.running && (
            <ExportPanel
              os={tools?.os}
              projectDir={projectDir}
              finalPly={job.finalPly}
              lastExport={lastExport}
              onExported={(stats) => {
                setLastExport(stats);
                // Mostra o resultado na hora para comparar com o original.
                setPreviewExport({ version: Date.now() });
                setTab("viewer");
              }}
              previewing={!!previewExport}
              onPreview={(on) => {
                setPreviewExport(on ? { version: Date.now() } : null);
                if (on) setTab("viewer");
              }}
            />
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
              {job.frames.length ? ` (${job.frames.length})` : ""}
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
          <LanguageSelect />
        </div>

        <UpdateBanner busy={job.running} />
        {tools && missing.length > 0 && (
          <ToolsSetup
            tools={tools}
            missing={[...missing]}
            rmbg={maskOpts.enabled && maskOpts.method === "rmbg"}
            onInstalled={() => api.detectTools().then(setTools)}
          />
        )}
        {videoError && <div className="banner error">{t.app.videoError(videoError)}</div>}
        {job.error && (
          <div className="banner error">
            <strong>{job.error.stage ? t.app.failedAt(t.stages[job.error.stage]) : t.app.failed}</strong>
            <pre>{job.error.message}</pre>
          </div>
        )}

        <div className="content">
          <div className={`pane ${tab === "viewer" ? "" : "hidden"}`}>
            <Viewer
              splat={viewerSplat}
              frame={frame}
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
                  {!videoPath && <p className="muted">{t.app.dropToStart}</p>}
                </div>
              }
            />
          </div>
          {tab === "frames" && (
            <div className="pane scroll">
              <FramesGrid frames={job.frames} version={framesVersion} masksDir={job.masksDir} />
            </div>
          )}
          {tab === "log" && (
            <div className="pane scroll">
              <LogView lines={logLines} />
            </div>
          )}
        </div>

        <StageStrip job={job} totalSteps={settings.train.totalSteps} tools={tools} masks={maskOpts} />
      </main>
    </div>
  );
}
