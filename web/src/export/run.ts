// Optimized export of the trained splat, like src-tauri/src/export/mod.rs:
// opacity pruning, SH band truncation, merge-based decimation and output as
// .ply or compressed .sog.

import type { ExportOptions, ExportStats, Msg } from "../../../src/lib/api";
import { decimate } from "./decimate.ts";
import { writeSog, type WebpEncoder } from "./sog.ts";
import { readPly, restLen, select, truncateSh, writePly } from "./splats.ts";

export type Progress = (p: { fraction: number; message: Msg }) => void;

/** Output file name; it varies with the settings so it doesn't overwrite
 *  earlier exports made with other parameters. */
export function outputName(o: ExportOptions) {
  let name = "splat";
  if (o.decimate > 0) name += `_dec${Math.round(o.decimate * 100)}`;
  if (o.minOpacity > 0) name += `_op${Math.round(o.minOpacity * 1000)}`;
  if (o.shDegree != null) name += `_sh${o.shDegree}`;
  // Don't overwrite the training's splat.ply.
  if (o.format === "ply" && name === "splat") name += "_export";
  return `${name}.${o.format}`;
}

export function runExport(
  input: Uint8Array,
  o: ExportOptions,
  webp: WebpEncoder,
  progress: Progress,
): { name: string; bytes: Uint8Array; stats: Omit<ExportStats, "path" | "seconds"> } {
  const report = (fraction: number, message: Msg) => progress({ fraction, message });
  report(0, { key: "export.readingPly" });
  let s = readPly(input);
  const inputCount = s.count;

  // Pruning: drop low opacities and any row with NaN/∞.
  const p = Math.min(Math.max(o.minOpacity, 1e-6), 0.999);
  const minLogit = o.minOpacity > 0 ? Math.log(p / (1 - p)) : -Infinity;
  const rl = restLen(s);
  const keep: number[] = [];
  const finite = (a: Float32Array, from: number, to: number) => {
    for (let k = from; k < to; k++) if (!Number.isFinite(a[k])) return false;
    return true;
  };
  for (let i = 0; i < s.count; i++) {
    if (!(s.opacity[i] >= minLogit)) continue;
    if (!finite(s.pos, i * 3, i * 3 + 3) || !finite(s.scale, i * 3, i * 3 + 3)) continue;
    if (s.rot[i * 4] === 0 && s.rot[i * 4 + 1] === 0 && s.rot[i * 4 + 2] === 0 && s.rot[i * 4 + 3] === 0) continue;
    if (!finite(s.rest, i * rl, (i + 1) * rl)) continue;
    keep.push(i);
  }
  if (keep.length < s.count) s = select(s, keep);
  const prunedCount = s.count;
  if (!prunedCount) throw new Error("opacity pruning removed every Gaussian");
  if (o.shDegree != null) s = truncateSh(s, o.shDegree);

  // Weight of each phase in the progress bar.
  const [decSpan, writeStart] = o.decimate > 0 ? [0.55, 0.6] : [0, 0.1];
  if (o.decimate > 0) {
    report(0.05, { key: "export.merging" });
    s = decimate(s, o.decimate, (f) => report(0.05 + f * decSpan, { key: "export.merging" })) ?? s;
  }
  const span = 1 - writeStart;
  let bytes: Uint8Array;
  if (o.format === "ply") {
    report(writeStart, { key: "export.writingPly" });
    bytes = writePly(s);
  } else {
    bytes = writeSog(s, webp, "Gausify Orbit web", (f, m) => report(writeStart + f * span, m));
  }
  report(1, { key: "export.done" });
  return {
    name: outputName(o),
    bytes,
    stats: { inputCount, prunedCount, outputCount: s.count, inputBytes: input.length, outputBytes: bytes.length, shDegree: s.shDegree },
  };
}
