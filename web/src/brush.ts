// Treino do Brush no navegador (bindings `brush-js`, gerados por `pnpm web:wasm`).

import init, { BrushApp, BrushMessageKind, type BrushMessage } from "../pkg/brush_js.js";
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

// Passos por ida ao WASM: mais amortiza a fronteira JS↔WASM; menos deixa o
// cancelamento e a barra de progresso mais responsivos.
const STEPS_PER_BATCH = 10;

export interface TrainEvents {
  status(text: string): void;
  dataset(train: number, evalViews: number): void;
  /** Uma vez por lote de passos. */
  step(iter: number, splats: number): void;
  eval(iter: number, psnr: number, ssim: number): void;
  /** Prévia a cada `exportEvery` passos (o treino espera a leitura da GPU). */
  snapshot(iter: number, ply: Uint8Array): void;
  log(line: string): void;
}

/** Treina o dataset da pasta e devolve o .ply final. Soltar o stream (no
 *  cancelamento) descarta o trabalho pendente do Burn. */
export async function train(dir: FileSystemDirectoryHandle, opts: TrainOptions, ev: TrainEvents, signal: AbortSignal): Promise<Uint8Array> {
  const { app } = await brushRuntime();
  const t = app.startTrainingFromDirectory(dir, async (initial: Record<string, unknown>) =>
    applyOptions({ ...initial }, opts, ev.log),
  );
  let iter = 0, splats = 0, lastSnapshot = 0;
  const handle = (m: BrushMessage) => {
    switch (m.kind) {
      case BrushMessageKind.StartLoading:
        ev.status(`Carregando ${m.name ?? "dataset"} na GPU`);
        break;
      case BrushMessageKind.DatasetLoaded:
        ev.dataset(m.trainViews ?? 0, m.evalViews ?? 0);
        break;
      case BrushMessageKind.SplatsUpdated:
      case BrushMessageKind.RefineStep:
        if (m.numSplats != null) splats = m.numSplats;
        break;
      case BrushMessageKind.TrainStep:
        iter = m.iter ?? iter;
        break;
      case BrushMessageKind.EvalResult:
        ev.eval(m.iter ?? iter, m.psnr ?? 0, m.ssim ?? 0);
        break;
      case BrushMessageKind.Warning:
        ev.log(`aviso: ${m.text ?? ""}`);
        break;
    }
    m.free();
  };
  try {
    for (;;) {
      if (signal.aborted) throw new DOMException("cancelado", "AbortError");
      const msgs = await t.trainSteps(STEPS_PER_BATCH);
      if (msgs.length === 0) break;
      const before = iter;
      for (const m of msgs) handle(m);
      if (iter > before) ev.step(iter, splats);
      if (opts.exportEvery > 0 && iter - lastSnapshot >= opts.exportEvery && iter < opts.totalSteps) {
        lastSnapshot = iter;
        const ply = await t.exportPly();
        if (ply) ev.snapshot(iter, ply);
      }
    }
    if (signal.aborted) throw new DOMException("cancelado", "AbortError");
    const ply = await t.exportPly();
    if (!ply) throw new Error("o Brush não devolveu o splat final");
    return ply;
  } finally {
    t.free();
  }
}
