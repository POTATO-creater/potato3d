import { useRef, type ReactNode } from "react";
import { fileUrl, type ImageSet, type MaskOptions, type MaskSource, type MaskSourceOf, type PoseCheckMode, type PoseEstimator, type Settings, type Stage, type SubjectPrompt, type VideoInfo } from "../lib/api";
import { promptInTrim } from "../lib/prompt";
import { PRESETS, REDUCE_HAZE_WEIGHT, type PresetId } from "../lib/presets";
import { Field, NumField, Segmented, Toggle } from "./fields";
import { VideoPlayer } from "./VideoPlayer";
import { ImageSequence } from "./ImageSequence";
import { MasksPanel } from "./MasksPanel";
import type { PreviewSource } from "./MaskPreview";
import { basename } from "../lib/platform";
import { fmtBytes } from "../lib/format";
import type { PackageStatus } from "../lib/batch";
import { useI18n } from "../i18n";

export type { PackageStatus };

interface Props {
  /** Item selector, above the preview (with more than one input open). */
  batch?: ReactNode;
  /** Something is open: picking files adds to the batch. */
  hasItems: boolean;
  /** Identifies the shown item (resets the player when it changes). */
  sourceKey: string | null;
  videoPath: string | null;
  video: VideoInfo | null;
  /** Already-extracted frames, instead of a video. */
  imageSet: ImageSet | null;
  trim: { start: number; end: number | null };
  onTrim: (t: { start: number; end: number | null }) => void;
  /** Videos, images or packages. */
  onPickFiles: () => void;
  /** Folders of frames. */
  onPickFolder: () => void;
  projectDir: string | null;
  onPickProject: () => void;
  /** Save the project as .gorbit; missing when there is nothing to save yet. */
  onSavePackage?: () => void;
  onOpenPackage: () => void;
  /** Progress or result of the last package save/open. */
  packageStatus: PackageStatus | null;
  preset: PresetId | "custom";
  onPreset: (p: PresetId) => void;
  /** Camera estimator (stored apart from the quality presets). */
  estimator: PoseEstimator;
  onEstimator: (e: PoseEstimator) => void;
  /** Pose check (stored apart from the quality presets). */
  poseCheck: PoseCheckMode;
  onPoseCheck: (m: PoseCheckMode) => void;
  settings: Settings;
  onSettings: (s: Settings) => void;
  masks: MaskOptions;
  onMasks: (m: MaskOptions) => void;
  /** Copies this masking to the other items; missing without a batch. */
  onMasksToAll?: () => void;
  /** The attached or generated mask of a project changed. */
  onMaskSource: (projectDir: string, src: MaskSource | null) => void;
  fromStage: Stage;
  onFromStage: (s: Stage) => void;
  available: Record<Stage, boolean>;
  /** The shown item is being processed. */
  running: boolean;
  /** Some item is being processed (quality is shared, so it locks). */
  anyRunning: boolean;
  canRun: boolean;
  onRun: () => void;
  onCancel: () => void;
  /** "Process all", with more than one item open. */
  queue?: { pending: number; running: boolean; canRunAll: boolean; onRunAll: () => void; onSkip: () => void };
  /** Export panel (only shown with a trained splat). */
  exportPanel?: ReactNode;
}

const fmtTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

export function Sidebar(p: Props) {
  const { t } = useI18n();
  const ts = t.sidebar;
  const videoEl = useRef<HTMLVideoElement>(null);
  const s = p.settings;
  const set = <K extends keyof Settings>(k: K, patch: Partial<Settings[K]>) =>
    p.onSettings({ ...s, [k]: { ...s[k], ...patch } });
  const locked = p.running;
  const settingsLocked = p.anyRunning;
  // SAM 2.1 marking (on the video or images); invalid outside the trim range.
  const sam2 = p.masks.enabled && p.masks.method === "sam2";
  const prompt = sam2 ? (p.masks.prompt ?? null) : null;
  const promptOutside = !!prompt && !promptInTrim(prompt, p.trim);
  const marking = sam2
    ? { prompt, onPrompt: locked ? undefined : (pr: SubjectPrompt | null) => p.onMasks({ ...p.masks, prompt: pr }) }
    : {};
  // Mask preview: the marked frame, else the trim start (or the first image).
  const markedImage = prompt?.image ? p.imageSet?.paths.find((x) => basename(x) === prompt.image) : undefined;
  const previewSource: PreviewSource | null = p.videoPath
    ? { kind: "video", src: fileUrl(p.videoPath), time: !promptOutside && prompt?.time != null ? prompt.time : p.trim.start }
    : p.imageSet
      ? { kind: "image", src: fileUrl(markedImage ?? p.imageSet.paths[0]), name: basename(markedImage ?? p.imageSet.paths[0]) }
      : null;
  // Mask video/frames ("external"): matched to this source.
  const maskSource: MaskSourceOf | null =
    p.videoPath && p.video
      ? { kind: "video", path: p.videoPath, fps: p.video.fps, trim: p.trim }
      : p.imageSet
        ? { kind: "images", paths: p.imageSet.paths }
        : null;
  const sourceSize: [number, number] | null = p.video ? [p.video.width, p.video.height] : p.imageSet ? [p.imageSet.width, p.imageSet.height] : null;
  const pack = p.packageStatus;
  const packBusy = pack?.kind === "saving" || pack?.kind === "opening";
  const end = p.trim.end ?? p.video?.duration ?? 0;
  const span = Math.max(0, end - p.trim.start);

  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark" aria-hidden />
        Gausify Orbit
      </div>

      <div className="sidebar-scroll">
        <section className="panel">
          <header className="panel-head">
            <h2>{p.imageSet ? ts.images : p.videoPath ? ts.video : ts.source}</h2>
            <span className="panel-actions">
              <button className="link" onClick={p.onPickFiles} title={p.hasItems ? ts.addTitle : undefined}>
                {p.hasItems ? ts.add : ts.chooseVideo}
              </button>
              <button className="link" onClick={p.onPickFolder} title={ts.chooseFolderTitle}>
                {ts.chooseFolder}
              </button>
            </span>
          </header>

          {p.batch}

          {p.videoPath ? (
            <>
              <VideoPlayer
                key={p.sourceKey}
                videoRef={videoEl}
                src={fileUrl(p.videoPath)}
                duration={p.video?.duration ?? 0}
                fps={p.video?.fps ?? 30}
                trim={p.trim}
                onTrim={p.onTrim}
                locked={locked}
                {...marking}
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
              <ImageSequence key={p.sourceKey} paths={p.imageSet.paths} {...marking} />
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
          ) : p.hasItems ? null : (
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
              <button
                className="link"
                onClick={p.onPickProject}
                disabled={locked || !(p.videoPath || p.imageSet)}
                title={ts.importTitle}
              >
                {ts.importProject}
              </button>
            </span>
          </header>
          <div className="path" title={p.projectDir ?? ""}>
            {p.projectDir ?? ts.pickVideoFirst}
          </div>
          <div className="package-row">
            <button className="btn-sm" onClick={p.onSavePackage} disabled={!p.onSavePackage || packBusy} title={ts.savePackageTitle}>
              {ts.savePackage}
            </button>
            <button className="btn-sm ghost" onClick={p.onOpenPackage} title={ts.openPackageTitle}>
              {ts.openPackage}
            </button>
          </div>
          {pack && (pack.kind === "saving" || pack.kind === "opening") && (
            <div className="export-progress">
              <div className="bar">
                <div className="bar-fill" style={{ width: `${Math.round(pack.fraction * 100)}%` }} />
              </div>
              <span className="help">{pack.kind === "saving" ? ts.packSaving : ts.packOpening}</span>
            </div>
          )}
          {pack?.kind === "saved" && <p className="help">{ts.packSaved(pack.name, fmtBytes(pack.bytes))}</p>}
          {pack?.kind === "error" && <p className="help export-error">{ts.packError(pack.message)}</p>}
        </section>

        <MasksPanel
          masks={p.masks}
          onMasks={p.onMasks}
          onApplyAll={p.onMasksToAll}
          locked={locked}
          previewPaused={p.anyRunning}
          preview={previewSource}
          promptOutside={promptOutside}
          projectDir={p.projectDir}
          source={maskSource}
          sourceSize={sourceSize}
          onMaskSource={p.onMaskSource}
        />

        <section className="panel">
          <header className="panel-head">
            <h2>{ts.quality}</h2>
            {p.queue && <span className="muted small">{ts.batch.sharedQuality}</span>}
            {p.preset === "custom" && <span className="badge">{ts.custom}</span>}
          </header>
          <div className="presets">
            {PRESETS.map((pr) => (
              <button
                key={pr.id}
                className={`preset ${p.preset === pr.id ? "active" : ""}`}
                disabled={settingsLocked}
                onClick={() => p.onPreset(pr.id)}
              >
                <span className="preset-label">{t.presets[pr.id].label}</span>
                <span className="preset-hint">{t.presets[pr.id].hint}</span>
              </button>
            ))}
          </div>

          <details className="advanced">
            <summary>{ts.advanced}</summary>
            <fieldset disabled={settingsLocked}>
              <h3>{ts.frames}</h3>
              <div className="grid2">
                <NumField label={ts.frames} value={s.frames.targetFrames} min={20} max={600} step={10}
                  onChange={(v) => set("frames", { targetFrames: v })} />
                <NumField label={ts.maxSide} value={s.frames.maxSide} min={480} max={3840} step={160} suffix="px"
                  onChange={(v) => set("frames", { maxSide: v })} />
                <NumField label={ts.candidates} hint={ts.candidatesHint} value={s.frames.oversample} min={1} max={8}
                  onChange={(v) => set("frames", { oversample: v })} />
              </div>

              <h3>{ts.poses}</h3>
              <Field label={ts.estimator} hint={p.estimator === "colmap" ? ts.estimatorColmapHint : ts.estimatorAkazeHint}>
                <Segmented
                  value={p.estimator}
                  options={[
                    { value: "akaze", label: ts.estimatorAkaze },
                    { value: "colmap", label: ts.estimatorColmap },
                  ]}
                  onChange={p.onEstimator}
                />
              </Field>
              <div className="grid2">
                <NumField label={ts.matchWindow} value={s.poses.matchWindow} min={3} max={40}
                  onChange={(v) => set("poses", { matchWindow: v })} />
              </div>
              <Toggle label={ts.closedLoop} hint={ts.closedLoopHint} checked={s.poses.closedLoop}
                onChange={(v) => set("poses", { closedLoop: v })} />
              <Toggle label={ts.flatOrbit} hint={ts.flatOrbitHint}
                checked={s.poses.flatOrbit ?? false}
                onChange={(v) => set("poses", { flatOrbit: v })} />
              <Toggle label={ts.fillMissing} hint={ts.fillMissingHint}
                checked={s.poses.fillMissing ?? false}
                onChange={(v) => set("poses", { fillMissing: v })} />
              <Field
                label={ts.poseCheck}
                hint={
                  !p.masks.enabled
                    ? ts.poseCheckNeedsMasks
                    : p.poseCheck === "off"
                      ? ts.poseCheckOffHint
                      : p.poseCheck === "always"
                        ? ts.poseCheckAlwaysHint
                        : ts.poseCheckAutoHint
                }
              >
                <Segmented
                  value={p.poseCheck}
                  disabled={!p.masks.enabled}
                  options={[
                    { value: "off", label: ts.poseCheckOff },
                    { value: "auto", label: ts.poseCheckAuto },
                    { value: "always", label: ts.poseCheckAlways },
                  ]}
                  onChange={p.onPoseCheck}
                />
              </Field>

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
                label={ts.reduceHaze}
                hint={ts.reduceHazeHint}
                checked={(s.train.depthDistortionWeight ?? 0) > 0}
                onChange={(v) => set("train", { depthDistortionWeight: v ? REDUCE_HAZE_WEIGHT : undefined })}
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
        {p.queue?.running ? (
          <div className="run-row">
            <button className="btn-primary danger" onClick={p.onCancel} title={ts.batch.cancelAllTitle}>
              {ts.batch.cancelAll}
            </button>
            <button className="btn-secondary" onClick={p.queue.onSkip} title={ts.batch.skipTitle}>
              {ts.batch.skip}
            </button>
          </div>
        ) : p.anyRunning ? (
          <button className="btn-primary danger" onClick={p.onCancel}>
            {t.common.cancel}
          </button>
        ) : p.queue ? (
          <div className="run-row">
            <button className="btn-primary" onClick={p.queue.onRunAll} disabled={!p.queue.canRunAll} title={ts.batch.runAllTitle}>
              {ts.batch.runAll(p.queue.pending)}
            </button>
            <button className="btn-secondary" onClick={p.onRun} disabled={!p.canRun} title={ts.batch.runThisTitle}>
              {ts.batch.runThis}
            </button>
          </div>
        ) : (
          <button className="btn-primary" onClick={p.onRun} disabled={!p.canRun}>
            {ts.run}
          </button>
        )}
      </footer>
    </aside>
  );
}
