// Treino do Brush no navegador (bindings `brush-js`, gerados por `pnpm web:wasm`).

import init, { BrushApp, BrushMessageKind, type BrushMessage, type Training } from "../pkg/brush_js.js";
import type { TrainOptions } from "../../src/lib/api";

export interface GpuInfo {
  vendor: string;
  architecture: string;
  description: string;
  subgroups: boolean;
  maxBufferSize: number;
  maxStorageBinding: number;
}

let runtime: Promise<{ app: BrushApp; gpu: GpuInfo }> | null = null;

/** Sobe o WASM e um GPUDevice com os limites máximos do adaptador (o Brush
 *  precisa de `subgroups` e de buffers grandes). Só uma vez por página. */
export function brushRuntime() {
  runtime ??= (async () => {
    if (!("gpu" in navigator)) throw new Error("Este navegador não tem WebGPU.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("Nenhum adaptador WebGPU disponível.");
    // `mappable-primary-buffers` aparece em alguns adaptadores do Chrome mas
    // é recusado no requestDevice.
    const features = [...adapter.features].filter((f) => f !== "mappable-primary-buffers") as GPUFeatureName[];
    const requiredLimits: Record<string, number> = {};
    for (const k in adapter.limits) {
      const v = (adapter.limits as unknown as Record<string, number>)[k];
      if (typeof v === "number") requiredLimits[k] = v;
    }
    const device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits });
    if (device.features.has("subgroups")) enableSubgroupsInShaders(device);
    await init();
    const app = new BrushApp();
    app.initExisting(adapter, device, device.queue);
    const info = adapter.info;
    return {
      app,
      gpu: {
        vendor: info.vendor,
        architecture: info.architecture,
        description: info.description,
        subgroups: adapter.features.has("subgroups"),
        maxBufferSize: device.limits.maxBufferSize,
        maxStorageBinding: device.limits.maxStorageBufferBindingSize,
      },
    };
  })();
  runtime.catch(() => (runtime = null));
  return runtime;
}

const SUBGROUP_USE = /\bsubgroup[A-Z]\w*\s*\(|@builtin\(\s*subgroup_/;

/** O writer WGSL do naga (wgpu 30) usa as funções de subgroup sem escrever
 *  `enable subgroups;`, que o Chrome exige desde que a extensão saiu do modo
 *  experimental: sem isso os kernels de ordenação/scan do Brush não compilam.
 *  O wgpu chama `device.createShaderModule` pelo objeto JS, então basta
 *  sobrescrever o método nesta instância. */
function enableSubgroupsInShaders(device: GPUDevice) {
  const create = device.createShaderModule.bind(device);
  device.createShaderModule = (desc) =>
    create(
      SUBGROUP_USE.test(desc.code) && !/^\s*enable\s+subgroups\s*;/m.test(desc.code)
        ? { ...desc, code: `enable subgroups;\n${desc.code}` }
        : desc,
    );
}

export type TrainUpdate =
  | { kind: "status"; text: string }
  | { kind: "dataset"; train: number; eval: number }
  | { kind: "step"; iter: number; elapsedMs: number | null }
  | { kind: "splats"; count: number }
  | { kind: "eval"; iter: number; psnr: number; ssim: number }
  | { kind: "warning"; text: string }
  | { kind: "preview"; iter: number; ply: Uint8Array }
  | { kind: "done"; ply: Uint8Array | null }
  | { kind: "error"; text: string };

/** Campos do `TrainStreamConfig` (JSON achatado, kebab-case) que o
 *  SplatForge controla; o resto fica no padrão do Brush. */
function applyOptions(cfg: Record<string, unknown>, o: TrainOptions, log: (s: string) => void) {
  const set: Record<string, unknown> = {
    "total-train-iters": o.totalSteps,
    "sh-degree": Math.min(o.shDegree, 3),
    "max-splats": o.maxSplats,
    "max-resolution": o.maxResolution,
    "ssim-weight": Math.max(o.ssimWeight, 0),
    "eval-split-every": o.evalSplitEvery || null,
    "eval-every": o.evalSplitEvery ? o.exportEvery : 0,
  };
  if (o.scaleLossWeight != null) set["scale-loss-weight"] = o.scaleLossWeight;
  for (const [k, v] of Object.entries(set)) {
    if (k in cfg) cfg[k] = v;
    else log(`Opção "${k}" não existe nesta versão do Brush; ignorada.`);
  }
  return cfg;
}

// Passos por ida ao WASM: mais amortiza a fronteira JS↔WASM; menos deixa
// pausa e prévia mais responsivas.
const STEPS_PER_BATCH = 10;

export class TrainRun {
  private training: Training | null = null;
  private paused = false;
  private resume: (() => void) | null = null;
  private previewWanted = false;
  private lastPreview = 0;

  constructor(
    private emit: (u: TrainUpdate) => void,
    /** Intervalo das prévias automáticas, em ms (0 desliga). */
    public previewEveryMs: number,
  ) {}

  async start(dir: FileSystemDirectoryHandle, opts: TrainOptions, log: (s: string) => void) {
    const { app } = await brushRuntime();
    this.emit({ kind: "status", text: "Carregando dataset…" });
    const t = app.startTrainingFromDirectory(dir, async (initial: Record<string, unknown>) =>
      applyOptions({ ...initial }, opts, log),
    );
    this.training = t;
    this.lastPreview = performance.now();
    try {
      while (true) {
        while (this.paused) await new Promise<void>((r) => (this.resume = r));
        if (this.training !== t) return; // cancelado
        const msgs = await t.trainSteps(STEPS_PER_BATCH);
        if (this.training !== t) return;
        if (msgs.length === 0) break;
        for (const m of msgs) this.handle(m);
        const due = this.previewEveryMs > 0 && performance.now() - this.lastPreview > this.previewEveryMs;
        if (this.previewWanted || due) await this.preview(t);
      }
      this.emit({ kind: "status", text: "Exportando .ply…" });
      this.emit({ kind: "done", ply: (await t.exportPly()) ?? null });
    } catch (e) {
      if (this.training === t) this.emit({ kind: "error", text: String(e) });
    } finally {
      if (this.training === t) this.training = null;
      t.free();
    }
  }

  private lastIter = 0;

  private handle(m: BrushMessage) {
    switch (m.kind) {
      case BrushMessageKind.StartLoading:
        this.emit({ kind: "status", text: `Carregando ${m.name ?? "dataset"}…` });
        break;
      case BrushMessageKind.DatasetLoaded:
        this.emit({ kind: "dataset", train: m.trainViews ?? 0, eval: m.evalViews ?? 0 });
        break;
      case BrushMessageKind.SplatsUpdated:
      case BrushMessageKind.RefineStep:
        if (m.numSplats != null) this.emit({ kind: "splats", count: m.numSplats });
        break;
      case BrushMessageKind.TrainStep:
        this.lastIter = m.iter ?? this.lastIter;
        this.emit({ kind: "step", iter: this.lastIter, elapsedMs: m.elapsedMs ?? null });
        break;
      case BrushMessageKind.EvalResult:
        this.emit({ kind: "eval", iter: m.iter ?? 0, psnr: m.psnr ?? 0, ssim: m.ssim ?? 0 });
        break;
      case BrushMessageKind.Warning:
        this.emit({ kind: "warning", text: m.text ?? "" });
        break;
    }
    m.free();
  }

  /** Lê os splats da GPU como .ply. O treino fica parado enquanto isso: o
   *  stream só anda quando é consumido. */
  private async preview(t: Training) {
    this.previewWanted = false;
    const ply = await t.exportPly();
    this.lastPreview = performance.now();
    if (ply && this.training === t) this.emit({ kind: "preview", iter: this.lastIter, ply });
  }

  requestPreview() {
    this.previewWanted = true;
  }

  setPaused(p: boolean) {
    this.paused = p;
    if (!p) {
      this.resume?.();
      this.resume = null;
    }
  }

  get isPaused() {
    return this.paused;
  }

  /** O laço sai ao fim do lote em andamento e libera o treino (soltar o
   *  stream cancela o trabalho pendente do Burn). */
  cancel() {
    this.training = null;
    this.setPaused(false);
  }
}
