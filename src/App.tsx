import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import {
  api,
  DEFAULT_CHROMA,
  host,
  IMAGE_EXT,
  PACKAGE_EXT,
  type ExportStats,
  type ImageSet,
  type JobEvent,
  STAGES,
  type MaskOptions,
  type SceneFrame,
  type Settings,
  type Stage,
  type Tools,
  type VideoInfo,
} from "./lib/api";
import { discardedBy, initialJob, jobReducer } from "./lib/job";
import { fmtDuration } from "./lib/format";
import { revealLabel, stageBlurbs } from "./lib/platform";
import { ToolsSetup } from "./components/ToolsSetup";
import { PRESETS, presetById, type PresetId } from "./lib/presets";
import { Sidebar, type PackageStatus } from "./components/Sidebar";
import { Viewer } from "./components/Viewer";
import { StageStrip } from "./components/StageStrip";
import { FrameLightbox, FramesGrid, LogView } from "./components/FramesGrid";
import { ExportPanel } from "./components/ExportPanel";
import { maskMethods } from "./components/MasksPanel";
import { LanguageSelect } from "./components/LanguageSelect";
import { ComfySettingsButton } from "./components/ComfySettings";
import { UpdateBanner } from "./components/UpdateBanner";
import { useSidebarWidth } from "./components/SidebarResizer";
import { promptInTrim } from "./lib/prompt";
import { basename } from "./lib/platform";
import { I18nProvider, useI18n } from "./i18n";
import "./styles.css";

const VIDEO_EXT = ["mp4", "mov", "m4v", "webm", "mkv", "avi"];
const extOf = (p: string) => p.split(".").pop()?.toLowerCase() ?? "";
const isVideo = (p: string) => VIDEO_EXT.includes(extOf(p));
const isPackage = (p: string) => extOf(p) === PACKAGE_EXT;
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
      // A marcação do SAM 2.1 é do vídeo da sessão anterior: não serve mais.
      return { ...base, despill: despill ?? base.despill, ...saved, chroma: { ...DEFAULT_CHROMA, ...chroma }, prompt: null };
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
    if (raw) {
      const saved = JSON.parse(raw);
      // Preset nomeado segue a definição atual (ajustes novos chegam a quem já
      // o escolheu); só o "personalizado" guarda os valores salvos.
      const p = PRESETS.find((x) => x.id === saved.preset);
      return p ? { preset: p.id, settings: p.settings } : saved;
    }
  } catch {
    /* armazenamento indisponível: usa o padrão */
  }
  return { preset: "balanced", settings: presetById("balanced").settings };
}

/** Ajustes da interface que vão no pacote do projeto (.gorbit). */
interface Session {
  settings: Settings;
  masks: MaskOptions;
  trim: { start: number; end: number | null };
}

/** Sessão de um pacote, mesclada com os padrões atuais: pode vir de outra
 *  versão do app (com campos a menos) ou do outro app (desktop ↔ web), com um
 *  método de recorte que não existe aqui. */
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

/** O recorte por IA tem o que precisa neste sistema? */
function maskToolReady(tools: Tools, m: MaskOptions) {
  // O SAM 2.1 (só no web) baixa os modelos durante o recorte.
  if (m.method === "sam2") return tools.os === "web";
  // No web o outro recorte por IA é o RMBG.
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
  /** Frames já extraídos, no lugar do vídeo. */
  const [imageSet, setImageSet] = useState<ImageSet | null>(null);
  const [sourceError, setSourceError] = useState<{ kind: "video" | "images"; message: string } | null>(null);
  const [trim, setTrim] = useState<{ start: number; end: number | null }>({ start: 0, end: null });
  const [projectDir, setProjectDir] = useState<string | null>(null);
  const [preset, setPreset] = useState<PresetId | "custom">(stored.preset);
  const [settings, setSettings] = useState<Settings>(stored.settings);
  const [maskOpts, setMaskOpts] = useState<MaskOptions>(loadMasks);
  const sidebar = useSidebarWidth();
  const [fromStage, setFromStage] = useState<Stage>("frames");
  const [frame, setFrame] = useState<SceneFrame | null>(null);
  const [tab, setTab] = useState<Tab>("viewer");
  const [job, dispatch] = useReducer(jobReducer, initialJob);
  const [framesVersion, setFramesVersion] = useState(0);
  /** Frame aberto ao clicar numa câmera do 3D (índice em `job.frames`). */
  const [camFrame, setCamFrame] = useState<number | null>(null);
  const [lastExport, setLastExport] = useState<ExportStats | null>(null);
  /** Mostra a exportação no 3D em vez do splat do treino. */
  const [previewExport, setPreviewExport] = useState<{ version: number } | null>(null);
  const [pack, setPack] = useState<PackageStatus | null>(null);
  const packBusy = pack?.kind === "saving" || pack?.kind === "opening";

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
      // Frame e splat chegam juntos: o Viewer alinha o splat ao frame vigente
      // quando ele carrega, e o do projeto anterior não serve.
      const frame = project.poses ? await api.sceneFrame(dir).catch(() => null) : null;
      dispatch({ type: "load", project });
      setFrame(frame);
      setLastExport(project.lastExport ?? null);
      setPreviewExport(null);
      setFramesVersion(Date.now());
      setFromStage(project.poses ? "train" : project.frames.length ? "poses" : "frames");
    },
    [],
  );

  /** `dir`: projeto já escolhido (pacote aberto); senão o padrão do vídeo. */
  const chooseVideo = useCallback(
    async (path: string, dir?: string) => {
      setVideoPath(path);
      setVideo(null);
      setImageSet(null);
      setSourceError(null);
      setTrim({ start: 0, end: null });
      setMaskOpts((m) => (m.prompt ? { ...m, prompt: null } : m));
      logs.current = [];
      setLogLines([]);
      try {
        setVideo(await api.probeVideo(path));
      } catch (e) {
        setSourceError({ kind: "video", message: String(e) });
      }
      await openProject(dir ?? (await api.defaultProjectDir(path)));
    },
    [openProject],
  );

  /** Devolve false se nenhuma imagem serviu (a entrada anterior fica). */
  const chooseImages = useCallback(
    async (paths: string[], dir?: string) => {
      setSourceError(null);
      let set: ImageSet;
      try {
        set = await api.probeImages(paths);
      } catch (e) {
        // Mantém a entrada anterior: nada aqui serve como frame.
        setSourceError({ kind: "images", message: String(e) });
        return false;
      }
      setVideoPath(null);
      setVideo(null);
      setImageSet(set);
      setTrim({ start: 0, end: null });
      setMaskOpts((m) => (m.prompt ? { ...m, prompt: null } : m));
      logs.current = [];
      setLogLines([]);
      await openProject(dir ?? (await api.defaultProjectDir(set.dir)));
      return true;
    },
    [openProject],
  );

  const session: Session = { settings, masks: maskOpts, trim };
  const sessionRef = useRef(session);
  sessionRef.current = session;

  /** Projeto salvo em .gorbit: extrai num projeto novo e o abre com a fonte e
   *  os ajustes de quando foi salvo, como um projeto em cache. */
  const openPackage = useCallback(
    async (path: string) => {
      setPack({ kind: "opening", fraction: 0 });
      try {
        const r = await api.unpackProject(path, (fraction) => setPack({ kind: "opening", fraction }));
        const restored = r.video ? (await chooseVideo(r.video, r.projectDir), true) : r.images ? await chooseImages(r.images, r.projectDir) : false;
        if (!restored) {
          // Sem a fonte: as etapas prontas continuam valendo (só os frames não refazem).
          setVideoPath(null);
          setVideo(null);
          setImageSet(null);
          logs.current = [];
          setLogLines([]);
          await openProject(r.projectDir);
        }
        const s = readSession(r.session, sessionRef.current, tools?.os);
        if (s) {
          setSettings(s.settings);
          setPreset(matchPreset(s.settings));
          setMaskOpts(s.masks);
          setTrim(s.trim);
        }
        setPack(null);
      } catch (e) {
        setPack({ kind: "error", message: String(e) });
      }
    },
    [chooseVideo, chooseImages, openProject, tools?.os],
  );

  /** Um pacote, um vídeo, ou então imagens e pastas de frames. */
  const openPaths = useCallback(
    (paths: string[]) => {
      const pkg = paths.find(isPackage);
      const v = paths.find(isVideo);
      if (pkg) openPackage(pkg);
      else if (v) chooseVideo(v);
      else if (paths.length) chooseImages(paths);
    },
    [chooseVideo, chooseImages, openPackage],
  );

  // Arrastar e soltar um vídeo, imagens, uma pasta ou um pacote em qualquer lugar da janela.
  useEffect(
    () =>
      host.onDrop((paths) => {
        if (!job.running && !packBusy) openPaths(paths);
      }),
    [openPaths, job.running, packBusy],
  );

  async function pickFiles() {
    openPaths(await host.pickFiles([...VIDEO_EXT, ...IMAGE_EXT, PACKAGE_EXT]));
  }

  async function pickPackage() {
    const [p] = await host.pickFiles([PACKAGE_EXT]);
    if (p) openPaths([p]);
  }

  /** Salva o projeto em .gorbit, inclusive no meio de um job: vão só as
   *  etapas prontas que ele não está refazendo (o reducer já as descartou). */
  async function savePackage() {
    if (!projectDir || !job.frames.length || packBusy) return;
    const dest = await host.pickPackageDest(projectDir);
    if (!dest) return;
    setPack({ kind: "saving", fraction: 0 });
    try {
      const stats = await api.packProject(
        {
          projectDir,
          dest,
          include: { masks: !!job.masks, poses: !!job.poses, splat: !!job.finalPly },
          source: videoPath ? { kind: "video", path: videoPath } : imageSet ? { kind: "images", paths: imageSet.paths } : null,
          session,
        },
        (fraction) => setPack({ kind: "saving", fraction }),
      );
      setPack({ kind: "saved", name: basename(stats.path), bytes: stats.bytes });
    } catch (e) {
      setPack({ kind: "error", message: String(e) });
    }
  }

  async function pickFramesFolder() {
    try {
      const paths = await host.pickFramesFolder();
      if (paths.length) chooseImages(paths);
    } catch (e) {
      setSourceError({ kind: "images", message: String(e) });
    }
  }

  /** O vídeo ou a pasta das imagens. */
  const sourcePath = videoPath ?? imageSet?.dir ?? null;

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

  /** Roda as etapas de `from` a `to` (a mesma, para rodar só uma). */
  async function run(from: Stage, to: Stage = "train") {
    // A fonte só é lida na etapa de frames (um pacote pode vir sem ela).
    if (!projectDir || (!sourcePath && from === "frames") || packBusy) return;
    const masksForPoses = maskOpts.enabled && maskOpts.forPoses;
    const gone = discardedBy(from, to, masksForPoses);
    if (gone.has("poses")) setFrame(null);
    dispatch({ type: "start", from, to, masksForPoses });
    if (gone.has("train")) {
      // Um novo treino invalida a exportação (o backend também a esquece).
      setLastExport(null);
      setPreviewExport(null);
    }
    setTab(from === "frames" || to === "masks" ? "frames" : "viewer");
    try {
      await api.startJob(
        {
          videoPath: sourcePath ?? "",
          images: imageSet?.paths ?? null,
          projectDir,
          fromStage: from,
          toStage: to,
          frames: { ...settings.frames, trimStart: trim.start, trimEnd: trim.end },
          // Marcação fora do trecho (o trecho mudou depois) não vale mais.
          masks: promptInTrim(maskOpts.prompt, trim) ? maskOpts : { ...maskOpts, prompt: null },
          poses: settings.poses,
          train: settings.train,
        },
        onEvent,
      );
    } catch (e) {
      dispatch({ type: "event", event: { type: "failed", stage: null, error: String(e) } });
    }
  }

  /** Recusa ou aceita de volta frames (menu de contexto da grade). */
  async function setRejected(names: string[], rejected: boolean) {
    if (!projectDir || job.running || packBusy || !names.length) return;
    try {
      const project = await api.setFramesRejected(projectDir, names, rejected);
      dispatch({ type: "framesEdited", project });
      // A partida não pode ficar numa etapa cujo pré-requisito deixou de valer.
      const need: Stage | null = maskOpts.enabled && !project.masks ? "masks" : !project.poses ? "poses" : null;
      if (need && STAGES.indexOf(fromStage) > STAGES.indexOf(need)) setFromStage(need);
    } catch (e) {
      dispatch({ type: "event", event: { type: "failed", stage: null, error: String(e) } });
    }
  }

  // Ao terminar os frames, leva a atenção para o 3D quando o treino começa.
  useEffect(() => {
    if (job.stages.train.status === "running" && tab === "frames") setTab("viewer");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.stages.train.status]);

  /** Câmera do COLMAP → frame: o nome da imagem (00012.jpg) é o índice. */
  function pickCamera(camera: number) {
    const n = frame?.frameIndices[camera];
    if (n == null) return;
    const i = job.frames.findIndex((f) => Number(basename(f.path).replace(/\.[^.]+$/, "")) === n);
    setCamFrame(i >= 0 ? i : job.frames[n] ? n : null);
  }

  const viewerSplat =
    previewExport && lastExport
      ? { path: lastExport.path, iter: null, version: previewExport.version, label: t.app.exported(basename(lastExport.path)) }
      : job.splat;

  const missing = tools
    ? (
        [
          // Imagens dispensam a decodificação de vídeo.
          { name: "ffmpeg", key: "ffmpeg", ok: (!!tools.ffmpeg && !!tools.ffprobe) || !!imageSet },
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

  const accepted = job.frames.filter((f) => !f.rejected).length;
  const available: Record<Stage, boolean> = {
    frames: true,
    masks: accepted > 0,
    poses: accepted > 0 && (!maskOpts.enabled || !maskOpts.forPoses || !!job.masks),
    train: !!job.poses && (!maskOpts.enabled || !!job.masks),
  };

  // Ferramenta de que cada etapa precisa (chaves de `missing`).
  const toolStage: Record<(typeof missing)[number]["key"], Stage> = { ffmpeg: "frames", model: "masks", colmap: "poses", brush: "train" };
  const missingFor = (from: Stage, to: Stage) =>
    missing.filter((m) => {
      const i = STAGES.indexOf(toolStage[m.key]);
      return i >= STAGES.indexOf(from) && i <= STAGES.indexOf(to);
    });
  const hasSource = !!projectDir && (!!video || !!imageSet);
  /** Só a etapa de frames lê a fonte; as outras usam o que já está no projeto. */
  const startable = (st: Stage) => hasSource || (!!projectDir && st !== "frames" && job.frames.length > 0);
  /** Por que cada etapa não pode rodar sozinha; null = pode. */
  const blocked = Object.fromEntries(
    STAGES.map((st) => {
      const tool = missingFor(st, st)[0];
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

  return (
    <div className="app" style={{ "--sidebar-w": `${sidebar.width}px` } as React.CSSProperties}>
      {sidebar.handle}
      <Sidebar
        os={tools?.os}
        videoPath={videoPath}
        video={video}
        imageSet={imageSet}
        trim={trim}
        onTrim={setTrim}
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
        canRun={startable(fromStage) && !packBusy && missingFor(fromStage, "train").length === 0}
        onRun={() => run(fromStage)}
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
              frame={frame}
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
        {sourceError && (
          <div className="banner error">
            {sourceError.kind === "video" ? t.app.videoError(sourceError.message) : t.app.imagesError(sourceError.message)}
          </div>
        )}
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
          onRunStage={(st) => run(st, st)}
          onCancel={() => api.cancelJob()}
        />
      </main>
    </div>
  );
}
