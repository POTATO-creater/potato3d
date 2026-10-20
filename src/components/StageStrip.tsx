import type { MaskOptions, PoseCheckStats, PoseEstimator, Stage } from "../lib/api";
import { stageToolLabels } from "../lib/platform";
import type { JobState, StageState } from "../lib/job";
import { fmtCount, fmtDuration as fmtSecs } from "../lib/format";
import { useI18n } from "../i18n";

function StatusDot({ status }: { status: StageState["status"] }) {
  return <span className={`dot dot-${status}`} aria-label={status} />;
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const w = 96, h = 26;
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const pts = values
    .map((v, i) => `${(i / (values.length - 1)) * w},${h - 2 - ((v - min) / span) * (h - 4)}`)
    .join(" ");
  return (
    <svg className="spark" width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden>
      <polyline points={pts} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

/** One bar per frame of the pose check: green under the threshold, then
 *  amber, red from twice it. */
function ConsistencyStrip({ values, threshold }: { values: number[]; threshold: number }) {
  if (values.length < 2) return null;
  const w = 96, h = 14, bw = w / values.length;
  const tone = (v: number) => (v < threshold ? "var(--good)" : v < 2 * threshold ? "var(--warn)" : "var(--bad)");
  return (
    <svg className="spark" width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden>
      {values.map((v, i) => {
        const bh = Math.max(2, Math.min(1, v / (3 * threshold)) * h);
        return <rect key={i} x={i * bw} y={h - bh} width={Math.max(bw - 0.5, 0.5)} height={bh} fill={tone(v)} />;
      })}
    </svg>
  );
}

const pct = (x: number) => `${(100 * x).toFixed(1)}%`;

function Consistency({ check }: { check: PoseCheckStats }) {
  const { t } = useI18n();
  const tl = t.stageStrip;
  const final = check.after ?? check.score;
  const tone = final < check.threshold ? "good" : check.rounds ? "warn" : "bad";
  const title = tl.consistencyTitle(pct(check.threshold)) + (check.rounds ? ` ${tl.consistencyCorrected(check.rounds)}` : "");
  return (
    <>
      <div className="metric" title={title}>
        <span className="metric-label">{tl.consistency}</span>
        <span className={`metric-value ${tone}`}>{check.after != null ? `${pct(check.score)} → ${pct(check.after)}` : pct(check.score)}</span>
      </div>
      <ConsistencyStrip values={check.afterFrames ?? check.frames} threshold={check.threshold} />
    </>
  );
}

function Metric({ label, value, tone }: { label: string; value: string; tone?: "good" | "warn" | "bad" }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className={`metric-value ${tone ?? ""}`}>{value}</span>
    </div>
  );
}

/** ▶ runs just this stage; on the running stage it becomes ■ (cancels the job). */
function StageButton({
  running,
  blocked,
  busy,
  onRun,
  onCancel,
}: {
  running: boolean;
  /** Why it can't run; null = it can. */
  blocked: string | null;
  /** Some job is running. */
  busy: boolean;
  onRun: () => void;
  onCancel: () => void;
}) {
  const { t } = useI18n();
  const tl = t.stageStrip;
  if (running)
    return (
      <button className="stage-run stop" onClick={onCancel} aria-label={tl.stopStage} title={tl.stopStage}>
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
          <rect x="2.5" y="2.5" width="7" height="7" rx="1.2" fill="currentColor" />
        </svg>
      </button>
    );
  return (
    <button
      className="stage-run"
      onClick={onRun}
      disabled={busy || !!blocked}
      aria-label={tl.runStage}
      title={blocked && !busy ? `${tl.runStage}: ${blocked}` : tl.runStage}
    >
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden>
        <path d="M3.5 2.2v7.6L9.8 6z" fill="currentColor" strokeLinejoin="round" stroke="currentColor" strokeWidth="0.8" />
      </svg>
    </button>
  );
}

export function StageStrip({
  job,
  totalSteps,
  masks: maskOpts,
  images,
  estimator,
  blocked,
  busy,
  onRunStage,
  onCancel,
}: {
  job: JobState;
  totalSteps: number;
  masks: MaskOptions;
  /** Frames come from images, not a video. */
  images?: boolean;
  /** Camera estimator chosen in the settings (the poses stage's tool label). */
  estimator?: PoseEstimator;
  /** Why each stage can't run on its own right now; null = it can. */
  blocked: Record<Stage, string | null>;
  /** Something else holds the pipeline (another batch item, a retouch). */
  busy?: boolean;
  onRunStage: (stage: Stage) => void;
  onCancel: () => void;
}) {
  const { t, tm } = useI18n();
  const tl = t.stageStrip;
  const { stages, frames, masks, poses, train, evals } = job;
  const labels = stageToolLabels(t, maskOpts, images, estimator);

  const rejected = frames.filter((f) => f.rejected).length;
  const medianSharp = (() => {
    if (!frames.length) return null;
    const v = frames.map((f) => f.sharpness).sort((a, b) => a - b);
    return v[Math.floor(v.length / 2)];
  })();

  const regRatio = poses ? poses.registered / Math.max(poses.total, 1) : null;
  const psnr = evals.length ? evals[evals.length - 1].psnr : null;
  const eta =
    train?.rate && stages.train.status === "running"
      ? Math.max(0, (totalSteps - train.iter) / train.rate)
      : null;

  return (
    <div className="stages">
      {(["frames", "masks", "poses", "train"] as Stage[]).map((st) => {
        const s = stages[st];
        return (
          <div key={st} className={`stage stage-${s.status}`}>
            <div className="stage-head">
              <StatusDot status={s.status} />
              <span className="stage-title">{t.stageTitles[st]}</span>
              <span className="stage-tool">{labels[st]}</span>
              {s.seconds != null && <span className="stage-time mono">{fmtSecs(s.seconds)}</span>}
              <StageButton
                running={s.status === "running"}
                blocked={blocked[st]}
                busy={job.running || !!busy}
                onRun={() => onRunStage(st)}
                onCancel={onCancel}
              />
            </div>
            <div className="bar">
              <div
                className={`bar-fill ${s.status === "running" && s.fraction === 0 ? "indeterminate" : ""}`}
                style={{ width: `${Math.round((s.status === "running" && s.fraction === 0 ? 1 : s.fraction) * 100)}%` }}
              />
            </div>
            <div className="stage-msg">{tm(s.message) || " "}</div>

            <div className="metrics">
              {st === "frames" && frames.length > 0 && (
                <>
                  <Metric label={tl.selected} value={String(frames.length - rejected)} />
                  {rejected > 0 && <Metric label={tl.rejected} value={String(rejected)} tone="warn" />}
                  {medianSharp != null && <Metric label={tl.medianSharpness} value={medianSharp.toFixed(0)} />}
                </>
              )}
              {st === "masks" && masks && (
                <>
                  <Metric label={tl.cuts} value={String(masks.count)} />
                  <Metric label={tl.coverage} value={`${Math.round(masks.meanCoverage * 100)}%`} />
                  {masks.noSubject > 0 && <Metric label={tl.noSubject} value={String(masks.noSubject)} tone="warn" />}
                  {masks.keyColor && (
                    <div className="metric">
                      <span className="metric-label">{tl.background}</span>
                      <span className="metric-value key-swatch">
                        <i style={{ background: `rgb(${masks.keyColor.join(",")})` }} />
                        {"#" + masks.keyColor.map((v) => v.toString(16).padStart(2, "0")).join("")}
                      </span>
                    </div>
                  )}
                </>
              )}
              {st === "poses" && poses && (
                <>
                  <Metric
                    label={tl.cameras}
                    value={`${poses.registered}/${poses.total}`}
                    tone={regRatio! >= 0.95 ? "good" : regRatio! >= 0.75 ? "warn" : "bad"}
                  />
                  {!!poses.inferred && (
                    <div className="metric" title={tl.inferredTitle}>
                      <span className="metric-label">{tl.inferred}</span>
                      <span className="metric-value">{poses.inferred}</span>
                    </div>
                  )}
                  {!!poses.refined && (
                    <div className="metric" title={tl.refinedTitle}>
                      <span className="metric-label">{tl.refined}</span>
                      <span className="metric-value good">{poses.refined}</span>
                    </div>
                  )}
                  <Metric label={tl.points} value={fmtCount(poses.points)} />
                  {poses.reprojectionError != null && (
                    <Metric
                      label={tl.reprojError}
                      value={`${poses.reprojectionError.toFixed(2)} px`}
                      tone={poses.reprojectionError < 1 ? "good" : poses.reprojectionError < 1.6 ? "warn" : "bad"}
                    />
                  )}
                  {poses.check && <Consistency check={poses.check} />}
                </>
              )}
              {st === "train" && (train || psnr != null) && (
                <>
                  {train && <Metric label={tl.splats} value={fmtCount(train.splats)} />}
                  {train?.rate != null && s.status === "running" && (
                    <Metric label={tl.speed} value={tl.stepsPerSec(train.rate.toFixed(0))} />
                  )}
                  {eta != null && <Metric label={tl.remaining} value={`~${fmtSecs(eta)}`} />}
                  {psnr != null && (
                    <div className="metric">
                      <span className="metric-label" title={tl.psnrTitle}>
                        PSNR
                      </span>
                      <span className={`metric-value ${psnr >= 30 ? "good" : psnr >= 25 ? "warn" : "bad"}`}>
                        {psnr.toFixed(1)} dB
                      </span>
                    </div>
                  )}
                  <Sparkline values={evals.map((e) => e.psnr)} />
                </>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
