import { useEffect, useRef, useState } from "react";
import { NumField, Segmented, Slider, Toggle } from "../../src/components/fields";
import { fmtDuration } from "../../src/lib/format";
import { presetById, type PresetId } from "../../src/lib/presets";
import { probeVideo, type VideoInfo } from "./pipeline/frames";
import { DEFAULT_MASKS, type MaskMethod, type MaskSettings, runPipeline } from "./pipeline/project";
import { MODELS } from "./pipeline/segment";
import { DEFAULT_SFM } from "./sfm";

interface Props {
  preset: PresetId;
  /** Treino em andamento, GPU indisponível etc. */
  disabled: boolean;
  log: (s: string) => void;
  onBusy: (busy: boolean) => void;
  onDone: (train: FileSystemDirectoryHandle) => void;
  onError: (e: string) => void;
}

const METHODS: { value: MaskMethod; label: string; title: string }[] = [
  { value: "rmbg", label: "IA (RMBG-1.4)", title: "RMBG-1.4 (BRIA, não comercial), 176 MB" },
  { value: "chroma", label: "Cor sólida", title: "Chroma key: tela verde ou azul" },
];

export function VideoPanel({ preset, disabled, log, onBusy, onDone, onError }: Props) {
  const [video, setVideo] = useState<File | null>(null);
  const [info, setInfo] = useState<VideoInfo | null>(null);
  const [frames, setFrames] = useState(presetById(preset).settings.frames);
  const [trim, setTrim] = useState({ start: 0, end: 0 });
  const [masks, setMasks] = useState<MaskSettings>(DEFAULT_MASKS);
  const [rmbgOk, setRmbgOk] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const urlLoaded = useRef(false);

  useEffect(() => setFrames(presetById(preset).settings.frames), [preset]);

  async function open(file: File) {
    setVideo(file);
    setInfo(null);
    try {
      const i = await probeVideo(file);
      setInfo(i);
      setTrim({ start: 0, end: 0 });
      log(`Vídeo ${file.name}: ${i.width}×${i.height}, ${i.duration.toFixed(1)} s, ${i.fps.toFixed(2)} fps`);
    } catch (e) {
      onError(String(e));
      setVideo(null);
    }
  }

  // `?video=<url>`: abre um vídeo servido por HTTP (testes)
  useEffect(() => {
    const url = new URLSearchParams(location.search).get("video");
    if (!url || urlLoaded.current) return;
    urlLoaded.current = true;
    fetch(url)
      .then((r) => r.blob())
      .then((b) => open(new File([b], url.split("/").pop() || "video.mp4", { type: b.type })))
      .catch((e) => onError(String(e)));
  }, []);

  async function process() {
    if (!video || !info) return;
    setStatus("Iniciando…");
    onBusy(true);
    const t0 = performance.now();
    try {
      const r = await runPipeline(
        video,
        info,
        { ...frames, trimStart: trim.start, trimEnd: trim.end > trim.start ? trim.end : null },
        masks,
        { ...DEFAULT_SFM, window: presetById(preset).settings.poses.matchWindow },
        log,
        (f, m) => setStatus(`${Math.round(f * 100)}% · ${m}`),
      );
      const t = r.timings, s = (k: string) => (t[k] ? `${k} ${(t[k] / 1000).toFixed(1)} s` : null);
      log(
        `Vídeo processado em ${((performance.now() - t0) / 1000).toFixed(1)} s: ` +
          [s("frames"), s("masks"), s("poses"), s("alpha")].filter(Boolean).join(", "),
      );
      // Só em dev: deixa testes automatizados lerem o projeto gerado.
      if (import.meta.env.DEV)
        (window as unknown as { __splatforgeProject?: unknown }).__splatforgeProject = {
          project: r.project,
          timings: r.timings,
          masks: r.masks,
          frames: r.frames,
          stats: r.sfm.model.stats,
        };
      onDone(r.train);
    } catch (e) {
      onError(String(e));
      log(`Erro no processamento: ${e}`);
    } finally {
      setStatus(null);
      onBusy(false);
    }
  }

  const busy = !!status;
  const needsLicense = masks.enabled && masks.method === "rmbg" && !rmbgOk;
  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2>Vídeo</h2>
        </div>
        <button className="btn-sm wide" onClick={() => input.current?.click()} disabled={busy || disabled}>
          {video ? "Trocar vídeo…" : "Escolher vídeo…"}
        </button>
        <input
          ref={input}
          type="file"
          accept="video/*"
          hidden
          onChange={(e) => {
            const f = e.currentTarget.files?.[0];
            if (f) open(f);
            e.currentTarget.value = "";
          }}
        />
        {video && (
          <div className="web-kv">
            <span>Arquivo</span>
            <span className="path">{video.name}</span>
            <span>Vídeo</span>
            <span className="mono">
              {info ? `${info.width}×${info.height} · ${fmtDuration(info.duration)} · ${info.fps.toFixed(0)} fps` : "…"}
            </span>
          </div>
        )}
        {video && info && (
          <div className="web-fields">
            <NumField label="Frames" value={frames.targetFrames} min={20} max={600} step={10} disabled={busy}
              onChange={(v) => setFrames({ ...frames, targetFrames: v })} />
            <NumField label="Lado máx." value={frames.maxSide} min={480} max={3840} step={160} suffix="px" disabled={busy}
              onChange={(v) => setFrames({ ...frames, maxSide: v })} />
            <NumField label="Início" hint="segundos" value={trim.start} min={0} max={info.duration} step={0.5} suffix="s" disabled={busy}
              onChange={(v) => setTrim({ ...trim, start: v })} />
            <NumField label="Fim" hint="0 = até o fim" value={trim.end} min={0} max={info.duration} step={0.5} suffix="s" disabled={busy}
              onChange={(v) => setTrim({ ...trim, end: v })} />
            <NumField label="Candidatos" hint="por frame; fica o mais nítido" value={frames.oversample} min={1} max={8} disabled={busy}
              onChange={(v) => setFrames({ ...frames, oversample: v })} />
          </div>
        )}
      </section>

      {video && info && (
        <section className="panel">
          <div className="panel-head">
            <h2>Recorte</h2>
          </div>
          <Toggle label="Isolar o sujeito" hint="O splat sai sem o fundo" checked={masks.enabled} disabled={busy}
            onChange={(v) => setMasks({ ...masks, enabled: v })} />
          {masks.enabled && (
            <>
              <div className="field">
                <Segmented value={masks.method} disabled={busy} options={METHODS} onChange={(m) => setMasks({ ...masks, method: m })} />
              </div>
              {masks.method === "rmbg" && (
                <>
                  <p className="help">
                    A licença do RMBG-1.4 ({MODELS.rmbg.license}) não permite uso comercial. São 176 MB baixados uma vez
                    e guardados no navegador; o recorte roda na GPU.
                  </p>
                  <Toggle label="Aceito a licença não comercial" checked={rmbgOk} disabled={busy} onChange={setRmbgOk} />
                </>
              )}
              {masks.method === "chroma" && (
                <>
                  <Slider label="Tolerância" value={masks.chroma.tolerance} min={0} max={60}
                    onChange={(v) => setMasks({ ...masks, chroma: { ...masks.chroma, tolerance: v } })} />
                  <Slider label="Suavidade" value={masks.chroma.softness} min={0} max={40}
                    onChange={(v) => setMasks({ ...masks, chroma: { ...masks.chroma, softness: v } })} />
                </>
              )}
              <Toggle label="Tirar o reflexo da tela (despill)" hint="Fundo verde ou azul" checked={masks.despill} disabled={busy}
                onChange={(v) => setMasks({ ...masks, despill: v })} />
              <Toggle label="Usar o recorte também nas poses" hint="Ajuda quando o fundo muda entre frames" checked={masks.forPoses}
                disabled={busy} onChange={(v) => setMasks({ ...masks, forPoses: v })} />
            </>
          )}
          <button className="btn-sm wide web-process" onClick={process} disabled={busy || disabled || needsLicense}>
            {status ?? "Processar vídeo (frames, recorte e poses)"}
          </button>
        </section>
      )}
    </>
  );
}
