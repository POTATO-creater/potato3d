import { useRef, type ReactNode } from "react";
import { fileUrl, type MaskOptions, type Settings, type Stage, type VideoInfo } from "../lib/api";
import { PRESETS, type PresetId } from "../lib/presets";
import { Field, NumField, Segmented, Toggle } from "./fields";
import { VideoPlayer } from "./VideoPlayer";
import { basename, fileManager, type Os } from "../lib/platform";

interface Props {
  os: Os | undefined;
  videoPath: string | null;
  video: VideoInfo | null;
  trim: { start: number; end: number | null };
  onTrim: (t: { start: number; end: number | null }) => void;
  onPickVideo: () => void;
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
  const videoEl = useRef<HTMLVideoElement>(null);
  const s = p.settings;
  const set = <K extends keyof Settings>(k: K, patch: Partial<Settings[K]>) =>
    p.onSettings({ ...s, [k]: { ...s[k], ...patch } });
  const locked = p.running;
  const end = p.trim.end ?? p.video?.duration ?? 0;
  const span = Math.max(0, end - p.trim.start);

  return (
    <aside className="sidebar">
      <div className="brand" data-tauri-drag-region>
        <span className="brand-mark" aria-hidden />
        SplatForge
      </div>

      <div className="sidebar-scroll">
        <section className="panel">
          <header className="panel-head">
            <h2>Vídeo</h2>
            <button className="link" onClick={p.onPickVideo} disabled={locked}>
              {p.videoPath ? "Trocar…" : "Escolher…"}
            </button>
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
                  <span>{p.video.codec}</span>
                </div>
              )}
              <div className="trim">
                <div className="trim-row">
                  <span className="trim-label">Trecho</span>
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
                    Início aqui
                  </button>
                  <button
                    className="btn-sm"
                    disabled={locked}
                    onClick={() => {
                      const t = videoEl.current?.currentTime ?? 0;
                      if (t > p.trim.start) p.onTrim({ ...p.trim, end: t });
                    }}
                  >
                    Fim aqui
                  </button>
                  <button
                    className="btn-sm ghost"
                    disabled={locked || (p.trim.start === 0 && p.trim.end == null)}
                    onClick={() => p.onTrim({ start: 0, end: null })}
                  >
                    Vídeo todo
                  </button>
                </div>
                <p className="help">
                  Arraste as alças laranja na linha do tempo ou use os botões. Use só o trecho em que o objeto continua
                  consistente: vídeo de IA costuma "derreter" no fim da órbita.
                </p>
              </div>
            </>
          ) : (
            <button className="dropzone" onClick={p.onPickVideo}>
              <strong>Arraste um vídeo orbital aqui</strong>
              <span>ou clique para escolher (.mp4, .mov, .webm)</span>
            </button>
          )}
        </section>

        <section className="panel">
          <header className="panel-head">
            <h2>Projeto</h2>
            <span className="panel-actions">
              <button className="link" onClick={p.onRevealProject} disabled={!p.projectDir}>
                {fileManager(p.os)}
              </button>
              <button className="link" onClick={p.onPickProject} disabled={locked || !p.videoPath}>
                Alterar…
              </button>
            </span>
          </header>
          <div className="path" title={p.projectDir ?? ""}>
            {p.projectDir ?? "Escolha um vídeo primeiro"}
          </div>
        </section>

        <section className="panel">
          <header className="panel-head">
            <h2>Recorte do sujeito</h2>
          </header>
          <Toggle
            label="Isolar o sujeito (remover o fundo)"
            hint="o splat sai só com a pessoa ou objeto, sem paredes e riscos em volta"
            checked={p.masks.enabled}
            disabled={locked}
            onChange={(v) => p.onMasks({ ...p.masks, enabled: v })}
          />
          {p.masks.enabled && (
            <Toggle
              label="Usar o recorte também nas poses"
              hint="ajuda se o fundo muda entre frames; atrapalha em sujeitos com pouca textura"
              checked={p.masks.forPoses}
              disabled={locked}
              onChange={(v) => p.onMasks({ ...p.masks, forPoses: v })}
            />
          )}
        </section>

        <section className="panel">
          <header className="panel-head">
            <h2>Qualidade</h2>
            {p.preset === "custom" && <span className="badge">personalizado</span>}
          </header>
          <div className="presets">
            {PRESETS.map((pr) => (
              <button
                key={pr.id}
                className={`preset ${p.preset === pr.id ? "active" : ""}`}
                disabled={locked}
                onClick={() => p.onPreset(pr.id)}
              >
                <span className="preset-label">{pr.label}</span>
                <span className="preset-hint">{pr.hint}</span>
              </button>
            ))}
          </div>

          <details className="advanced">
            <summary>Parâmetros avançados</summary>
            <fieldset disabled={locked}>
              <h3>Frames</h3>
              <div className="grid2">
                <NumField label="Frames" value={s.frames.targetFrames} min={20} max={600} step={10}
                  onChange={(v) => set("frames", { targetFrames: v })} />
                <NumField label="Lado máx." value={s.frames.maxSide} min={480} max={3840} step={160} suffix="px"
                  onChange={(v) => set("frames", { maxSide: v })} />
                <NumField label="Candidatos/frame" hint="escolhe o mais nítido" value={s.frames.oversample} min={1} max={8}
                  onChange={(v) => set("frames", { oversample: v })} />
              </div>

              <h3>Poses (COLMAP)</h3>
              <Field
                label="Features"
                hint={
                  s.poses.features === "sift"
                    ? "se registrar < 95% das câmeras, repete com o reforçado"
                    : s.poses.features === "siftPlus"
                      ? "acha pontos em cabelo, pele e fundo desfocado (~2× o tempo)"
                      : "registra bem, mas ~10× mais lento na CPU"
                }
              >
                <select
                  className="input"
                  value={s.poses.features}
                  onChange={(e) => set("poses", { features: e.currentTarget.value as Settings["poses"]["features"] })}
                >
                  <option value="sift">SIFT (rápido)</option>
                  <option value="siftPlus">SIFT reforçado</option>
                  <option value="aliked">ALIKED + LightGlue</option>
                </select>
              </Field>
              <div className="grid2">
                <NumField label="Vizinhos casados" value={s.poses.matchWindow} min={3} max={40}
                  onChange={(v) => set("poses", { matchWindow: v })} />
                <Field label="Câmera">
                  <select className="input" value={s.poses.cameraModel}
                    onChange={(e) => set("poses", { cameraModel: e.currentTarget.value as Settings["poses"]["cameraModel"] })}>
                    <option value="SIMPLE_PINHOLE">Pinhole simples</option>
                    <option value="PINHOLE">Pinhole (fx ≠ fy)</option>
                  </select>
                </Field>
              </div>
              <Toggle label="Órbita fechada (360°)" hint="casa o fim com o começo" checked={s.poses.closedLoop}
                onChange={(v) => set("poses", { closedLoop: v })} />

              <h3>Treino (Brush)</h3>
              <div className="grid2">
                <NumField label="Passos" value={s.train.totalSteps} min={1000} max={100000} step={1000}
                  onChange={(v) => set("train", { totalSteps: v })} />
                <NumField label="Grau SH" hint="0–3" value={s.train.shDegree} min={0} max={3}
                  onChange={(v) => set("train", { shDegree: v })} />
                <NumField label="Máx. splats" value={s.train.maxSplats / 1e6} min={0.1} max={10} step={0.1} suffix="M"
                  onChange={(v) => set("train", { maxSplats: Math.round(v * 1e6) })} />
                <NumField label="Resolução máx." value={s.train.maxResolution} min={480} max={3840} step={160} suffix="px"
                  onChange={(v) => set("train", { maxResolution: v })} />
                <NumField label="Prévia a cada" value={s.train.exportEvery} min={500} max={20000} step={500} suffix="passos"
                  onChange={(v) => set("train", { exportEvery: v })} />
              </div>
              <Toggle
                label="Perda SSIM"
                hint="detalhes um pouco mais finos, mas ~3× mais lento no Mac"
                checked={s.train.ssimWeight > 0}
                onChange={(v) => set("train", { ssimWeight: v ? 0.2 : 0 })}
              />
              <Toggle
                label="Suavizar áreas não filmadas"
                hint="menos riscos e cores estranhas no topo; −1,3 dB de detalhe"
                checked={s.train.scaleLossWeight != null}
                onChange={(v) => set("train", { scaleLossWeight: v ? 1e-6 : undefined })}
              />
              <Toggle
                label="Medir qualidade (PSNR)"
                hint="separa 1 a cada 16 frames para avaliação"
                checked={s.train.evalSplitEvery > 0}
                onChange={(v) => set("train", { evalSplitEvery: v ? 16 : 0 })}
              />
            </fieldset>
          </details>
        </section>

        {p.exportPanel}
      </div>

      <footer className="run">
        <Field label="Executar a partir de">
          <Segmented
            value={p.fromStage}
            onChange={p.onFromStage}
            disabled={locked}
            options={[
              { value: "frames", label: "Frames" },
              {
                value: "masks",
                label: "Recorte",
                disabled: !p.available.masks || !p.masks.enabled,
                title: p.masks.enabled ? "Requer frames extraídos" : "Ative o recorte do sujeito",
              },
              { value: "poses", label: "Poses", disabled: !p.available.poses, title: "Requer frames extraídos" },
              { value: "train", label: "Treino", disabled: !p.available.train, title: "Requer poses (e recortes, se ativados)" },
            ]}
          />
        </Field>
        {p.running ? (
          <button className="btn-primary danger" onClick={p.onCancel}>
            Cancelar
          </button>
        ) : (
          <button className="btn-primary" onClick={p.onRun} disabled={!p.canRun}>
            Gerar splat
          </button>
        )}
      </footer>
    </aside>
  );
}
