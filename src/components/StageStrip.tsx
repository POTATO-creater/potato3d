import type { Stage, Tools } from "../lib/api";
import { stageToolLabels } from "../lib/platform";
import type { JobState, StageState } from "../lib/job";
import { fmtCount, fmtDuration as fmtSecs } from "../lib/format";

const TITLES: Record<Stage, string> = { frames: "Frames", poses: "Poses de câmera", train: "Treino do splat" };


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

function Metric({ label, value, tone }: { label: string; value: string; tone?: "good" | "warn" | "bad" }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className={`metric-value ${tone ?? ""}`}>{value}</span>
    </div>
  );
}

export function StageStrip({ job, totalSteps, tools }: { job: JobState; totalSteps: number; tools: Tools | null }) {
  const { stages, frames, poses, train, evals } = job;
  const labels = stageToolLabels(tools);

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
      {(["frames", "poses", "train"] as Stage[]).map((st) => {
        const s = stages[st];
        return (
          <div key={st} className={`stage stage-${s.status}`}>
            <div className="stage-head">
              <StatusDot status={s.status} />
              <span className="stage-title">{TITLES[st]}</span>
              <span className="stage-tool">{labels[st]}</span>
              {s.seconds != null && <span className="stage-time mono">{fmtSecs(s.seconds)}</span>}
            </div>
            <div className="bar">
              <div
                className={`bar-fill ${s.status === "running" && s.fraction === 0 ? "indeterminate" : ""}`}
                style={{ width: `${Math.round((s.status === "running" && s.fraction === 0 ? 1 : s.fraction) * 100)}%` }}
              />
            </div>
            <div className="stage-msg">{s.message || " "}</div>

            <div className="metrics">
              {st === "frames" && frames.length > 0 && (
                <>
                  <Metric label="Selecionados" value={String(frames.length)} />
                  {medianSharp != null && <Metric label="Nitidez mediana" value={medianSharp.toFixed(0)} />}
                </>
              )}
              {st === "poses" && poses && (
                <>
                  <Metric
                    label="Câmeras"
                    value={`${poses.registered}/${poses.total}`}
                    tone={regRatio! >= 0.95 ? "good" : regRatio! >= 0.75 ? "warn" : "bad"}
                  />
                  <Metric label="Pontos" value={fmtCount(poses.points)} />
                  {poses.reprojectionError != null && (
                    <Metric
                      label="Erro reproj."
                      value={`${poses.reprojectionError.toFixed(2)} px`}
                      tone={poses.reprojectionError < 1 ? "good" : poses.reprojectionError < 1.6 ? "warn" : "bad"}
                    />
                  )}
                </>
              )}
              {st === "train" && (train || psnr != null) && (
                <>
                  {train && <Metric label="Splats" value={fmtCount(train.splats)} />}
                  {train?.rate != null && s.status === "running" && (
                    <Metric label="Velocidade" value={`${train.rate.toFixed(0)} passos/s`} />
                  )}
                  {eta != null && <Metric label="Restante" value={`~${fmtSecs(eta)}`} />}
                  {psnr != null && (
                    <div className="metric">
                      <span className="metric-label" title="Medido em frames que não entram no treino. >30 dB é ótimo; <25 dB indica vídeo inconsistente.">
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
