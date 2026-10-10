import { useEffect, useRef, useState } from "react";
import { Viewer } from "../../src/components/Viewer";
import { NumField, Segmented } from "../../src/components/fields";
import { fmtBytes, fmtCount, fmtDuration } from "../../src/lib/format";
import { PRESETS, presetById, type PresetId } from "../../src/lib/presets";
import type { TrainOptions } from "../../src/lib/api";
import { brushRuntime, TrainRun, type GpuInfo, type TrainUpdate } from "./brush";
import { datasetFromFiles, datasetFromUrl } from "./opfs";
import { inspectDataset, type DatasetInfo } from "./scene";

type Phase = "idle" | "running" | "paused" | "done" | "error";

interface Progress {
  status: string;
  iter: number;
  elapsedMs: number;
  splats: number;
  views: { train: number; eval: number } | null;
  eval: { iter: number; psnr: number; ssim: number } | null;
  /** Passos/s numa janela das últimas atualizações. */
  rate: number | null;
}

const EMPTY: Progress = { status: "", iter: 0, elapsedMs: 0, splats: 0, views: null, eval: null, rate: null };

const PREVIEW_OPTIONS = [
  { value: "0", label: "Manual" },
  { value: "10", label: "10 s" },
  { value: "30", label: "30 s" },
];

const hasDirPicker = typeof window !== "undefined" && "showDirectoryPicker" in window;
const hasWebGpu = typeof navigator !== "undefined" && "gpu" in navigator;

export default function App() {
  const [gpu, setGpu] = useState<GpuInfo | null>(null);
  const [gpuError, setGpuError] = useState<string | null>(null);
  const [dir, setDir] = useState<FileSystemDirectoryHandle | null>(null);
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [preset, setPreset] = useState<PresetId>("preview");
  const [train, setTrain] = useState<TrainOptions>(presetById("preview").settings.train);
  const [previewEvery, setPreviewEvery] = useState("10");
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState<Progress>(EMPTY);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [splat, setSplat] = useState<{ path: string; url: string; iter: number | null; version: number } | null>(null);
  const [result, setResult] = useState<Uint8Array | null>(null);
  const [importing, setImporting] = useState<string | null>(null);
  const run = useRef<TrainRun | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const samples = useRef<{ iter: number; t: number }[]>([]);

  const addLog = (s: string) => setLog((l) => [...l.slice(-200), `${new Date().toLocaleTimeString("pt-BR")}  ${s}`]);

  // Sobe o WebGPU + WASM logo de cara: mostra a GPU e falha cedo.
  useEffect(() => {
    if (!hasWebGpu) return;
    brushRuntime()
      .then(({ gpu }) => setGpu(gpu))
      .catch((e) => setGpuError(String(e)));
  }, []);

  useEffect(() => {
    if (run.current) run.current.previewEveryMs = Number(previewEvery) * 1000;
  }, [previewEvery]);

  // Blob da prévia anterior: solto um pouco depois, quando o Viewer já trocou.
  useEffect(() => {
    if (!splat) return;
    const url = splat.url;
    return () => void setTimeout(() => URL.revokeObjectURL(url), 30_000);
  }, [splat?.url]);

  // `?dataset=<url>`: baixa um dataset COLMAP servido por HTTP para o OPFS.
  // O ref evita a segunda importação do StrictMode (dev), que disputaria a
  // mesma pasta.
  const urlImported = useRef(false);
  useEffect(() => {
    const url = new URLSearchParams(location.search).get("dataset");
    if (!url || urlImported.current) return;
    urlImported.current = true;
    loadDir(() => datasetFromUrl(url, progressText("Baixando")));
  }, []);

  const progressText = (verb: string) => (done: number, total: number) => setImporting(`${verb} ${done}/${total} arquivos…`);

  async function loadDir(get: () => Promise<FileSystemDirectoryHandle>) {
    setError(null);
    try {
      setImporting("Abrindo…");
      const handle = await get();
      setDir(handle);
      setDataset(null);
      setDataset(await inspectDataset(handle));
    } catch (e) {
      if ((e as DOMException).name !== "AbortError") setError(String(e));
    } finally {
      setImporting(null);
    }
  }

  function pickFolder() {
    if (!hasDirPicker) return fileInput.current?.click();
    loadDir(() => window.showDirectoryPicker({ id: "splatforge-dataset", mode: "read" }));
  }

  function onFilesPicked(list: FileList | null) {
    const files = [...(list ?? [])];
    if (files.length) loadDir(() => datasetFromFiles(files, progressText("Copiando")));
  }

  function showPly(ply: Uint8Array, iter: number | null) {
    const url = URL.createObjectURL(new Blob([ply as BlobPart]));
    setSplat((s) => ({ path: "preview.ply", url, iter, version: (s?.version ?? 0) + 1 }));
  }

  function onUpdate(u: TrainUpdate) {
    switch (u.kind) {
      case "status":
        setProgress((p) => ({ ...p, status: u.text }));
        break;
      case "dataset":
        setProgress((p) => ({ ...p, views: { train: u.train, eval: u.eval } }));
        addLog(`Dataset: ${u.train} vistas de treino, ${u.eval} de avaliação`);
        break;
      case "splats":
        setProgress((p) => ({ ...p, splats: u.count }));
        break;
      case "step": {
        const now = performance.now();
        const s = samples.current;
        s.push({ iter: u.iter, t: now });
        if (s.length > 40) s.shift();
        const rate = s.length > 1 ? ((s[s.length - 1].iter - s[0].iter) * 1000) / (now - s[0].t) : null;
        setProgress((p) => ({ ...p, status: "Treinando", iter: u.iter, elapsedMs: u.elapsedMs ?? p.elapsedMs, rate }));
        break;
      }
      case "eval":
        setProgress((p) => ({ ...p, eval: u }));
        addLog(`Avaliação no passo ${u.iter}: PSNR ${u.psnr.toFixed(2)} dB, SSIM ${u.ssim.toFixed(3)}`);
        break;
      case "warning":
        addLog(`Aviso: ${u.text}`);
        break;
      case "preview":
        showPly(u.ply, u.iter);
        break;
      case "done":
        setPhase("done");
        setProgress((p) => ({ ...p, status: "Pronto" }));
        if (u.ply) {
          // Só em dev: deixa testes automatizados lerem o .ply sem passar pelo download.
          if (import.meta.env.DEV) (window as unknown as { __splatforgePly?: Uint8Array }).__splatforgePly = u.ply;
          setResult(u.ply);
          showPly(u.ply, null);
          addLog(`Treino concluído: ${fmtBytes(u.ply.byteLength)} de .ply`);
        }
        break;
      case "error":
        setPhase("error");
        setError(u.text);
        addLog(`Erro: ${u.text}`);
        break;
    }
  }

  async function start() {
    if (!dir) return;
    setError(null);
    setResult(null);
    setProgress({ ...EMPTY, status: "Iniciando…" });
    samples.current = [];
    setPhase("running");
    addLog(`Treinando "${dir.name}": ${train.totalSteps} passos, SH ${train.shDegree}, até ${train.maxResolution} px`);
    const r = new TrainRun(onUpdate, Number(previewEvery) * 1000);
    run.current = r;
    try {
      await r.start(dir, train, addLog);
    } catch (e) {
      onUpdate({ kind: "error", text: String(e) });
    } finally {
      if (run.current === r) run.current = null;
      setPhase((p) => (p === "running" || p === "paused" ? "idle" : p));
    }
  }

  function togglePause() {
    const r = run.current;
    if (!r) return;
    r.setPaused(!r.isPaused);
    setPhase(r.isPaused ? "paused" : "running");
    setProgress((p) => ({ ...p, status: r.isPaused ? "Pausado" : "Treinando" }));
    samples.current = [];
  }

  function cancel() {
    run.current?.cancel();
    run.current = null;
    setPhase("idle");
    setProgress((p) => ({ ...p, status: "Cancelado" }));
    addLog("Treino cancelado");
  }

  function download() {
    if (!result) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([result as BlobPart], { type: "application/octet-stream" }));
    a.download = `${dir?.name ?? "splat"}.ply`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
  }

  const busy = phase === "running" || phase === "paused";
  const pct = train.totalSteps ? Math.min(100, (progress.iter / train.totalSteps) * 100) : 0;
  const unsupported = hasWebGpu ? null : "Este navegador não tem WebGPU.";

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" />
          SplatForge
          <span className="web-badge">web · protótipo</span>
        </div>
        <div className="sidebar-scroll">
          <section className="panel">
            <div className="panel-head">
              <h2>Dataset</h2>
            </div>
            <button className="btn-sm wide" onClick={pickFolder} disabled={busy || !!importing}>
              {importing ?? (dir ? "Trocar pasta…" : "Escolher pasta…")}
            </button>
            {/* Sem showDirectoryPicker (Safari, Firefox): a pasta é copiada para o OPFS. */}
            <input
              ref={fileInput}
              type="file"
              hidden
              {...{ webkitdirectory: "" }}
              onChange={(e) => {
                onFilesPicked(e.currentTarget.files);
                e.currentTarget.value = "";
              }}
            />
            {dir && (
              <div className="web-kv">
                <span>Pasta</span>
                <span className="path">{dir.name}</span>
                <span>Imagens</span>
                <span className="mono">{dataset ? dataset.images : "…"}</span>
                <span>COLMAP</span>
                <span className={dataset && !dataset.model ? "bad" : "mono"}>
                  {dataset ? (dataset.model ?? "não encontrado") : "…"}
                </span>
                {dataset?.frame && (
                  <>
                    <span>Câmeras</span>
                    <span className="mono">{dataset.frame.cameras.length}</span>
                  </>
                )}
              </div>
            )}
            <p className="help">
              A pasta <code>dataset</code> (ou <code>dataset_alpha</code>, com recorte) de um projeto do SplatForge, ou
              qualquer export COLMAP com <code>images/</code> e <code>sparse/0</code>. Nada sai do computador.
            </p>
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>Treino</h2>
            </div>
            <Segmented
              value={preset}
              disabled={busy}
              options={PRESETS.map((p) => ({ value: p.id, label: p.label, title: p.hint }))}
              onChange={(id) => {
                setPreset(id);
                setTrain(presetById(id).settings.train);
              }}
            />
            <div className="web-fields">
              <NumField label="Passos" value={train.totalSteps} min={100} step={500} disabled={busy}
                onChange={(v) => setTrain({ ...train, totalSteps: v })} />
              <NumField label="Resolução máx." value={train.maxResolution} min={256} step={64} suffix="px" disabled={busy}
                onChange={(v) => setTrain({ ...train, maxResolution: v })} />
              <NumField label="Grau SH" value={train.shDegree} min={0} max={3} disabled={busy}
                onChange={(v) => setTrain({ ...train, shDegree: v })} />
              <NumField label="Peso SSIM" value={train.ssimWeight} min={0} max={1} step={0.05} disabled={busy}
                onChange={(v) => setTrain({ ...train, ssimWeight: v })} />
            </div>
            <div className="field">
              <span className="field-label">Prévia no visualizador</span>
              <Segmented value={previewEvery} options={PREVIEW_OPTIONS} onChange={setPreviewEvery} />
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>Progresso</h2>
              {progress.status && <span className="muted small">{progress.status}</span>}
            </div>
            <div className="bar">
              <div className="bar-fill" style={{ width: `${pct}%` }} />
            </div>
            <div className="web-kv">
              <span>Passo</span>
              <span className="mono">
                {progress.iter.toLocaleString("pt-BR")} / {train.totalSteps.toLocaleString("pt-BR")}
              </span>
              <span>Splats</span>
              <span className="mono">{progress.splats ? fmtCount(progress.splats) : "—"}</span>
              <span>Velocidade</span>
              <span className="mono">{progress.rate ? `${progress.rate.toFixed(1)} passos/s` : "—"}</span>
              <span>Tempo</span>
              <span className="mono">{progress.elapsedMs ? fmtDuration(progress.elapsedMs / 1000) : "—"}</span>
              <span>Vistas</span>
              <span className="mono">{progress.views ? `${progress.views.train} + ${progress.views.eval} aval.` : "—"}</span>
              <span>PSNR</span>
              <span className="mono">
                {progress.eval ? `${progress.eval.psnr.toFixed(2)} dB · SSIM ${progress.eval.ssim.toFixed(3)}` : "—"}
              </span>
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <h2>GPU</h2>
            </div>
            {gpu ? (
              <div className="web-kv">
                <span>Adaptador</span>
                <span className="mono">{[gpu.vendor, gpu.architecture].filter(Boolean).join(" · ") || "—"}</span>
                <span>Subgroups</span>
                <span className={gpu.subgroups ? "good" : "bad"}>{gpu.subgroups ? "sim" : "não"}</span>
                <span>Buffer máx.</span>
                <span className="mono">{fmtBytes(gpu.maxBufferSize)}</span>
                <span>Storage máx.</span>
                <span className="mono">{fmtBytes(gpu.maxStorageBinding)}</span>
              </div>
            ) : (
              <p className={`help ${gpuError || unsupported ? "bad" : ""}`}>{gpuError ?? unsupported ?? "Iniciando WebGPU…"}</p>
            )}
          </section>
        </div>

        <div className="run">
          {busy ? (
            <div className="web-run-row">
              <button className="btn-primary" onClick={togglePause}>
                {phase === "paused" ? "Retomar" : "Pausar"}
              </button>
              <button className="btn-sm" onClick={() => run.current?.requestPreview()} disabled={phase !== "running"}>
                Prévia agora
              </button>
              <button className="btn-sm ghost" onClick={cancel}>
                Cancelar
              </button>
            </div>
          ) : (
            <button
              className="btn-primary"
              onClick={start}
              disabled={!dir || !dataset?.model || !gpu || !!unsupported}
            >
              Treinar no navegador
            </button>
          )}
        </div>
      </aside>

      <main className="main">
        <div className="topbar">
          <span className="muted small">Brush · WebGPU{gpu?.vendor ? ` · ${gpu.vendor}` : ""}</span>
          <span style={{ marginLeft: "auto" }} />
          {result && (
            <button className="btn-sm" onClick={download}>
              Baixar .ply ({fmtBytes(result.byteLength)})
            </button>
          )}
        </div>
        {(error || unsupported) && (
          <div className="banner error">
            <pre>{error ?? `${unsupported} Use o Chrome ou o Edge no desktop.`}</pre>
          </div>
        )}
        <div className="content">
          <div className="pane">
            <Viewer
              splat={splat}
              frame={dataset?.frame ?? null}
              placeholder={
                <div className="placeholder">
                  <h1>Treino de Gaussian Splatting no navegador</h1>
                  <ol>
                    <li>
                      <strong>Escolha a pasta</strong> de um dataset COLMAP (imagens + poses).
                    </li>
                    <li>
                      <strong>Treine</strong>: o Brush roda em WebAssembly e usa a GPU via WebGPU.
                    </li>
                    <li>
                      <strong>Acompanhe</strong> as prévias aqui e baixe o <code>.ply</code> no fim.
                    </li>
                  </ol>
                </div>
              }
            />
          </div>
        </div>
        <pre className="web-log">{log.join("\n") || "—"}</pre>
      </main>
    </div>
  );
}
