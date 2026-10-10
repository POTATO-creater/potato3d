// Frames → recorte → poses → treino no navegador, com os mesmos eventos,
// regras de retomada e project.json de src-tauri/src/pipeline.rs.

import type { JobConfig, JobEvent, Msg, ProjectState, Stage } from "../../../src/lib/api";
import { train } from "../brush";
import { extractFrames, probeVideo } from "../pipeline/frames";
import { buildAlpha, makeMasks } from "../pipeline/masks";
import { DEFAULT_SFM, runSfm } from "../sfm";
import { renameImages, writeModel } from "../sfm/colmap";
import { dirAt, exists, publish, publishDir, readFile, remove, stem, unpublishUnder, video, writeFile } from "./files";
import { IMAGE_EXT, layout, pruneProjects, readState, saveState } from "./project";

const ORDER: Stage[] = ["frames", "masks", "poses", "train"];
const before = (a: Stage, b: Stage) => ORDER.indexOf(a) <= ORDER.indexOf(b);

const cancelled = () => new DOMException("cancelado", "AbortError");

/** Eventos de uma etapa, com o progresso limitado a ~15 por segundo. */
function reporter(emit: (e: JobEvent) => void, stage: Stage, signal: AbortSignal) {
  emit({ type: "stageStarted", stage });
  let last = 0;
  return {
    /** Também é o ponto de cancelamento dos laços de cada etapa. */
    progress(fraction: number, message: Msg) {
      if (signal.aborted) throw cancelled();
      const now = performance.now();
      if (fraction < 1 && now - last < 66) return;
      last = now;
      emit({ type: "progress", stage, fraction, message });
    },
    log: (line: string) => emit({ type: "log", stage, line }),
  };
}

export async function runJob(cfg: JobConfig, emit: (e: JobEvent) => void, signal: AbortSignal) {
  const t0 = performance.now();
  const at = { stage: null as Stage | null };
  try {
    const ply = await run(cfg, emit, signal, at);
    emit({ type: "finished", ply, seconds: (performance.now() - t0) / 1000 });
  } catch (e) {
    if (signal.aborted) emit({ type: "cancelled" });
    else {
      console.error(e);
      emit({ type: "failed", stage: at.stage, error: e instanceof Error ? e.message : String(e) });
    }
  }
}

async function run(cfg: JobConfig, emit: (e: JobEvent) => void, signal: AbortSignal, at: { stage: Stage | null }) {
  const l = layout(cfg.projectDir);
  const from = cfg.fromStage;
  let state: ProjectState = await readState(l.root);
  const secs = (t: number) => (performance.now() - t) / 1000;

  // ---------- 1. Frames ----------
  if (before(from, "frames")) {
    at.stage = "frames";
    const t = performance.now();
    const v = video(cfg.videoPath);
    const rep = reporter(emit, "frames", signal);
    await pruneProjects(l.root);
    rep.progress(0, { key: "frames.reading" });
    const info = (v.info ??= await probeVideo(v.file));
    rep.log(`${info.width}x${info.height} @ ${info.fps.toFixed(2)} fps, ${info.duration.toFixed(1)}s, ${info.codec}`);
    // Etapas seguintes ficam inválidas.
    for (const p of [l.images, `${l.dataset}/sparse`, l.masks, l.alpha, l.training, l.finalPly, l.exports]) await remove(p);
    const picked = await extractFrames(v.file, info, cfg.frames, await dirAt(l.images, true), (f) =>
      rep.progress(f, { key: "frames.decodingSharpness" }),
    );
    await publishDir(l.images, IMAGE_EXT);
    rep.progress(1, { key: "frames.selected", args: { n: picked.length } });
    const frames = picked.map((f) => ({ path: `${l.images}/${f.name}`, time: f.time, sharpness: f.sharpness }));
    state = { ...state, videoPath: cfg.videoPath, frames, masks: null, masksDir: null, poses: null, finalPly: null, lastPsnr: null, lastExport: null };
    await saveState(l.root, state);
    emit({ type: "framesReady", frames });
    emit({ type: "stageDone", stage: "frames", seconds: secs(t) });
  } else if (!state.frames.length || !(await exists(l.images))) {
    throw new Error("não há frames extraídos neste projeto; rode a partir da etapa de frames");
  }

  // ---------- 2. Recorte (opcional) ----------
  const m = cfg.masks;
  if (before(from, "masks")) {
    if (m.enabled) {
      at.stage = "masks";
      const t = performance.now();
      const rep = reporter(emit, "masks", signal);
      rep.progress(0, { key: "masks.detecting" });
      await remove(l.masks);
      const stats = await makeMasks(await dirAt(l.images), await dirAt(l.masks, true), m, rep.log, rep.progress);
      await publishDir(l.masks, /\.png$/i);
      rep.progress(1, m.method === "chroma" ? { key: "masks.doneChroma", args: { n: stats.count } } : { key: "masks.done", args: { n: stats.count, backend: stats.backend } });
      state = { ...state, masks: stats, masksDir: l.masks };
      await saveState(l.root, state);
      emit({ type: "masksReady", stats, dir: l.masks });
      emit({ type: "stageDone", stage: "masks", seconds: secs(t) });
    } else {
      emit({ type: "stageSkipped", stage: "masks", reason: { key: "masks.disabled" } });
    }
  }
  if (m.enabled && !(state.masks && (await exists(l.masks))))
    throw new Error("não há recortes neste projeto; rode a partir da etapa de recorte");

  // ---------- 3. Poses ----------
  // Recortar de novo sem usar as máscaras nas poses não muda as poses.
  const hasModel = () => exists(`${l.model}/images.bin`);
  const reuse = from === "masks" && !(m.enabled && m.forPoses) && state.poses && (await hasModel());
  if (reuse) {
    emit({ type: "stageSkipped", stage: "poses", reason: { key: "poses.reused" } });
  } else if (before(from, "poses")) {
    at.stage = "poses";
    const t = performance.now();
    const rep = reporter(emit, "poses", signal);
    await remove(`${l.dataset}/sparse`);
    const sfm = await runSfm(
      await dirAt(l.dataset),
      {
        ...DEFAULT_SFM,
        window: cfg.poses.matchWindow,
        closedLoop: cfg.poses.closedLoop,
        masksDir: m.enabled && m.forPoses ? await dirAt(l.masks) : undefined,
        signal,
      },
      rep.log,
      rep.progress,
    );
    await writeModel(await dirAt(l.dataset), sfm.files);
    const s = sfm.model.stats;
    const stats = {
      registered: Number(s.registered),
      total: Number(s.total),
      points: Number(s.points),
      reprojectionError: Number(s.meanError),
      mapper: "SfM incremental (navegador)",
    };
    if (stats.registered < 3) throw new Error(`só ${stats.registered} câmeras registradas; o vídeo não reconstrói`);
    state = { ...state, poses: stats, finalPly: null, lastPsnr: null, lastExport: null };
    await saveState(l.root, state);
    emit({ type: "posesReady", stats });
    emit({ type: "stageDone", stage: "poses", seconds: secs(t) });
  } else if (!(await hasModel())) {
    throw new Error("não há poses calculadas neste projeto; rode a partir da etapa de poses");
  }

  // ---------- 4. Treino ----------
  at.stage = "train";
  const t = performance.now();
  const rep = reporter(emit, "train", signal);
  const total = Math.max(cfg.train.totalSteps, 1);
  let dataset = l.dataset;
  await remove(l.alpha);
  if (m.enabled) {
    rep.progress(0, { key: "train.alphaDataset" });
    dataset = l.alpha;
    await buildAlpha(await dirAt(l.images), await dirAt(l.masks), await dirAt(`${l.alpha}/images`, true), m, rep.log, (_, msg) =>
      rep.progress(0, msg),
    );
    // O Brush acha as imagens pelo nome gravado no images.bin.
    for (const f of ["cameras.bin", "points3D.bin"]) await writeFile(`${l.alpha}/sparse/0/${f}`, await readFile(`${l.model}/${f}`));
    const images = await (await readFile(`${l.model}/images.bin`)).arrayBuffer();
    await writeFile(`${l.alpha}/sparse/0/images.bin`, renameImages(images, (n) => `${stem(n)}.png`));
  }
  rep.progress(0, { key: "train.loading" });
  await remove(l.training);
  let lastPsnr: number | null = null;
  let snapshot: string | null = null;
  const ply = await train(
    await dirAt(dataset),
    cfg.train,
    {
      status: (msg) => rep.progress(0, msg),
      dataset: (n, e) => rep.log(`dataset: ${n} vistas de treino${e ? `, ${e} de avaliação` : ""}`),
      step: (iter, splats) => {
        rep.progress(iter / total, { key: "train.step", args: { iter, total, splats } });
        emit({ type: "train", event: { kind: "step", iter, splats } });
      },
      eval: (iter, psnr, ssim) => {
        lastPsnr = psnr;
        rep.log(`avaliação no passo ${iter}: PSNR ${psnr.toFixed(2)} dB, SSIM ${ssim.toFixed(3)}`);
        emit({ type: "train", event: { kind: "eval", iter, psnr, ssim } });
      },
      snapshot: (iter, data) => {
        // Prévias ficam só na memória; a anterior é solta pelo publish.
        const path = `${l.training}/export_${iter}.ply`;
        publish(path, new Blob([data as BlobPart]));
        if (snapshot) unpublishUnder(snapshot);
        snapshot = path;
        emit({ type: "train", event: { kind: "snapshot", iter, path } });
      },
      log: rep.log,
    },
    signal,
  );
  publish(l.finalPly, await writeFile(l.finalPly, ply as BufferSource));
  rep.progress(1, { key: "train.done" });
  state = { ...state, finalPly: l.finalPly, lastPsnr, lastExport: null };
  await saveState(l.root, state);
  emit({ type: "stageDone", stage: "train", seconds: secs(t) });
  return l.finalPly;
}
