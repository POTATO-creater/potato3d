import { useRef, type ReactNode } from "react";
import { fileUrl, type ImageSet, type MaskOptions, type Settings, type Stage, type VideoInfo } from "../lib/api";
import { PRESETS, type PresetId } from "../lib/presets";
import { Field, NumField, Segmented, Select, Toggle } from "./fields";
import { VideoPlayer } from "./VideoPlayer";
import { ImageSequence } from "./ImageSequence";
import { MasksPanel } from "./MasksPanel";
import type { PreviewSource } from "./MaskPreview";
import { basename, fileManager, type Os } from "../lib/platform";
import { useI18n } from "../i18n";

interface Props {
  os: Os | undefined;
  videoPath: string | null;
  video: VideoInfo | null;
  /** Frames já extraídos, no lugar do vídeo. */
  imageSet: ImageSet | null;
  trim: { start: number; end: number | null };
  onTrim: (t: { start: number; end: number | null }) => void;
  /** Um vídeo ou várias imagens. */
  onPickFiles: () => void;
  /** Uma pasta de frames. */
  onPickFolder: () => void;
  projectDir: string | null;
  onPickProject: () => void;
  onRevealProject: () => void;
  preset: PresetId | "custom";
  onPreset: (p: PresetId) => void;
  settings: Settings;
  onSettings: (s: Settings) => void;
  masks: MaskOptions;
  onMasks: (m: MaskOptions) => void;
  fromStage: Stage;
  onFromStage: (s: Stage) => void;
  available: Record<Stage, boolean>;
  running: boolean;
  canRun: boolean;
  onRun: () => void;
  onCancel: () => void;
  /** Painel de exportação (só aparece com um splat treinado). */
  exportPanel?: ReactNode;
}

const fmtTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

export function Sidebar(p: Props) {
  const { t } = useI18n();
  const ts = t.sidebar;
  const videoEl = useRef<HTMLVideoElement>(null);
  // No web os projetos ficam no armazenamento do navegador e as poses saem
  // do SfM próprio (sem as opções do COLMAP).
  const web = p.os === "web";
  const s = p.settings;
  const set = <K extends keyof Settings>(k: K, patch: Partial<Settings[K]>) =>
    p.onSettings({ ...s, [k]: { ...s[k], ...patch } });
  const locked = p.running;
  // Prévia do recorte: o quadro marcado para o SAM 2.1, senão o início do trecho.
  const markedAt = p.masks.method === "sam2" ? p.masks.prompt?.time : null;
  const previewSource: PreviewSource | null = p.videoPath
    ? { kind: "video", src: fileUrl(p.videoPath), time: markedAt ?? p.trim.start }
    : p.imageSet
      ? { kind: "image", src: fileUrl(p.imageSet.paths[0]) }
      : null;
  const end = p.trim.end ?? p.video?.duration ?? 0;
  const span = Math.max(0, end - p.trim.start);

  return (
    <aside className="sidebar">
      <div className="brand" data-tauri-drag-region>
        <span className="brand-mark" aria-hidden />
        Gausify Orbit
      </div>

      <div className="sidebar-scroll">
        <section className="panel">
          <header className="panel-head">
            <h2>{p.imageSet ? ts.images : p.videoPath ? ts.video : ts.source}</h2>
            <span className="panel-actions">
              <button className="link" onClick={p.onPickFiles} disabled={locked}>
                {p.videoPath || p.imageSet ? ts.changeVideo : ts.chooseVideo}
              </button>
              <button className="link" onClick={p.onPickFolder} disabled={locked} title={ts.chooseFolderTitle}>
                {ts.chooseFolder}
              </button>
            </span>
          </header>

          {p.videoPath ? (
            <>
              <VideoPlayer
                videoRef={videoEl}
                src={fileUrl(p.videoPath)}
                duration={p.video?.duration ?? 0}
                fps={p.video?.fps ?? 30}
                trim={p.trim}
                onTrim={p.onTrim}
                locked={locked}
                {...(p.os === "web" && p.masks.enabled && p.masks.method === "sam2"
                  ? { prompt: p.masks.prompt ?? null, onPrompt: locked ? undefined : (prompt) => p.onMasks({ ...p.masks, prompt }) }
                  : {})}
              />
              <div className="file-name" title={p.videoPath}>
                {basename(p.videoPath)}
              </div>
              {p.video && (
                <div className="meta-row">
                  <span>
                    {p.video.width}×{p.video.height}
                  </span>
                  <span>{p.video.fps.toFixed(p.video.fps % 1 ? 2 : 0)} fps</span>
                  <span>{fmtTime(p.video.duration)}</span>
                  {p.video.codec && <span>{p.video.codec}</span>}
                </div>
              )}
              <div className="trim">
                <div className="trim-row">
                  <span className="trim-label">{ts.trim}</span>
                  <span className="mono">
                    {fmtTime(p.trim.start)} → {fmtTime(end)}
                  </span>
                  <span className="muted mono">({span.toFixed(1)} s)</span>
                </div>
                <div className="trim-actions">
                  <button
                    className="btn-sm"
                    disabled={locked}
                    onClick={() => {
                      const t = videoEl.current?.currentTime ?? 0;
                      p.onTrim({ start: t, end: p.trim.end != null && p.trim.end <= t ? null : p.trim.end });
                    }}
                  >
                    {ts.startHere}
                  </button>
                  <button
                    className="btn-sm"
                    disabled={locked}
                    onClick={() => {
                      const t = videoEl.current?.currentTime ?? 0;
                      if (t > p.trim.start) p.onTrim({ ...p.trim, end: t });
                    }}
                  >
                    {ts.endHere}
                  </button>
                  <button
                    className="btn-sm ghost"
                    disabled={locked || (p.trim.start === 0 && p.trim.end == null)}
                    onClick={() => p.onTrim({ start: 0, end: null })}
                  >
                    {ts.wholeVideo}
                  </button>
                </div>
                <p className="help">{ts.trimHelp}</p>
              </div>
            </>
          ) : p.imageSet ? (
            <>
              <ImageSequence paths={p.imageSet.paths} />
              <div className="file-name" title={p.imageSet.dir}>
                {basename(p.imageSet.dir)}
              </div>
              <div className="meta-row">
                <span>{ts.imageCount(p.imageSet.paths.length)}</span>
                <span>
                  {p.imageSet.width}×{p.imageSet.height}
                </span>
              </div>
              <p className="help">{ts.imagesHelp}</p>
            </>
          ) : (
            <button className="dropzone" onClick={p.onPickFiles}>
              <strong>{ts.dropTitle}</strong>
              <span>{ts.dropHint}</span>
            </button>
          )}
        </section>

        <section className="panel">
          <header className="panel-head">
            <h2>{ts.project}</h2>
            <span className="panel-actions">
              {!web && (
                <button className="link" onClick={p.onRevealProject} disabled={!p.projectDir}>
                  {fileManager(t, p.os)}
                </button>
              )}
              <button
                className="link"
                onClick={p.onPickProject}
                disabled={locked || !(p.videoPath || p.imageSet)}
                title={web ? ts.importTitle : undefined}
              >
                {web ? ts.importProject : ts.changeProject}
              </button>
            </span>
          </header>
          <div className="path" title={p.projectDir ?? ""}>
            {p.projectDir ?? ts.pickVideoFirst}
          </div>
        </section>

        <MasksPanel os={p.os} masks={p.masks} onMasks={p.onMasks} locked={locked} preview={previewSource} />

        <section className="panel">
          <header className="panel-head">
            <h2>{ts.quality}</h2>
            {p.preset === "custom" && <span className="badge">{ts.custom}</span>}
          </header>
          <div className="presets">
            {PRESETS.map((pr) => (
              <button
                key={pr.id}
                className={`preset ${p.preset === pr.id ? "active" : ""}`}
                disabled={locked}
                onClick={() => p.onPreset(pr.id)}
              >
                <span className="preset-label">{t.presets[pr.id].label}</span>
                <span className="preset-hint">{t.presets[pr.id].hint}</span>
              </button>
            ))}
          </div>

          <details className="advanced">
            <summary>{ts.advanced}</summary>
            <fieldset disabled={locked}>
              <h3>{ts.frames}</h3>
              <div className="grid2">
                <NumField label={ts.frames} value={s.frames.targetFrames} min={20} max={600} step={10}
                  onChange={(v) => set("frames", { targetFrames: v })} />
                <NumField label={ts.maxSide} value={s.frames.maxSide} min={480} max={3840} step={160} suffix="px"
                  onChange={(v) => set("frames", { maxSide: v })} />
                <NumField label={ts.candidates} hint={ts.candidatesHint} value={s.frames.oversample} min={1} max={8}
                  onChange={(v) => set("frames", { oversample: v })} />
              </div>

              <h3>{web ? ts.poses : ts.posesColmap}</h3>
              {!web && (
              <Field
                label={ts.features}
                hint={ts.featuresHint[s.poses.features]}
              >
                <Select
                  value={s.poses.features}
                  onChange={(features) => set("poses", { features })}
                  options={[
                    { value: "sift", ...ts.featureOptions.sift },
                    { value: "siftPlus", ...ts.featureOptions.siftPlus },
                    { value: "aliked", ...ts.featureOptions.aliked },
                  ]}
                />
              </Field>
              )}
              <div className="grid2">
                <NumField label={ts.matchWindow} value={s.poses.matchWindow} min={3} max={40}
                  onChange={(v) => set("poses", { matchWindow: v })} />
                {!web && (
                <Field label={ts.camera}>
                  <Select
                    value={s.poses.cameraModel}
                    onChange={(cameraModel) => set("poses", { cameraModel })}
                    options={[
                      { value: "SIMPLE_PINHOLE", label: ts.simplePinhole },
                      { value: "PINHOLE", label: ts.pinhole },
                    ]}
                  />
                </Field>
                )}
              </div>
              <Toggle label={ts.closedLoop} hint={ts.closedLoopHint} checked={s.poses.closedLoop}
                onChange={(v) => set("poses", { closedLoop: v })} />
              {!web && (
                <Toggle label={ts.flatOrbit} hint={ts.flatOrbitHint}
                  checked={s.poses.flatOrbit ?? false}
                  onChange={(v) => set("poses", { flatOrbit: v })} />
              )}

              <h3>{ts.training}</h3>
              <div className="grid2">
                <NumField label={ts.steps} value={s.train.totalSteps} min={1000} max={100000} step={1000}
                  onChange={(v) => set("train", { totalSteps: v })} />
                <NumField label={ts.shDegree} hint="0–3" value={s.train.shDegree} min={0} max={3}
                  onChange={(v) => set("train", { shDegree: v })} />
                <NumField label={ts.maxSplats} value={s.train.maxSplats / 1e6} min={0.1} max={10} step={0.1} suffix="M"
                  onChange={(v) => set("train", { maxSplats: Math.round(v * 1e6) })} />
                <NumField label={ts.maxResolution} value={s.train.maxResolution} min={480} max={3840} step={160} suffix="px"
                  onChange={(v) => set("train", { maxResolution: v })} />
                <NumField label={ts.previewEvery} value={s.train.exportEvery} min={500} max={20000} step={500} suffix={ts.stepsSuffix}
                  onChange={(v) => set("train", { exportEvery: v })} />
              </div>
              <Toggle
                label={ts.ssim}
                hint={ts.ssimHint}
                checked={s.train.ssimWeight > 0}
                onChange={(v) => set("train", { ssimWeight: v ? 0.2 : 0 })}
              />
              <Toggle
                label={ts.smooth}
                hint={ts.smoothHint}
                checked={s.train.scaleLossWeight != null}
                onChange={(v) => set("train", { scaleLossWeight: v ? 1e-6 : undefined })}
              />
              <Toggle
                label={ts.psnr}
                hint={ts.psnrHint}
                checked={s.train.evalSplitEvery > 0}
                onChange={(v) => set("train", { evalSplitEvery: v ? 16 : 0 })}
              />
            </fieldset>
          </details>
        </section>

        {p.exportPanel}
      </div>

      <footer className="run">
        <Field label={ts.runFrom}>
          <Segmented
            value={p.fromStage}
            onChange={p.onFromStage}
            disabled={locked}
            options={[
              { value: "frames", label: t.stages.frames },
              {
                value: "masks",
                label: t.stages.masks,
                disabled: !p.available.masks || !p.masks.enabled,
                title: p.masks.enabled ? ts.needsFrames : ts.enableMasks,
              },
              { value: "poses", label: t.stages.poses, disabled: !p.available.poses, title: ts.needsFrames },
              { value: "train", label: t.stages.train, disabled: !p.available.train, title: ts.needsPoses },
            ]}
          />
        </Field>
        {p.running ? (
          <button className="btn-primary danger" onClick={p.onCancel}>
            {t.common.cancel}
          </button>
        ) : (
          <button className="btn-primary" onClick={p.onRun} disabled={!p.canRun}>
            {ts.run}
          </button>
        )}
      </footer>
    </aside>
  );
}
