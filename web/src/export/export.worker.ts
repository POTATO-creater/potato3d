// Export off the UI thread: k-means, decimation and WebP over millions of
// Gaussians take seconds.

import type { ExportOptions, Msg } from "../../../src/lib/api";
import { runExport } from "./run.ts";
import { webpEncoder } from "./webp/index.ts";

export type ExportRequest = { input: Blob; options: ExportOptions };
export type ExportReply =
  | { kind: "progress"; fraction: number; message: Msg }
  | { kind: "done"; name: string; bytes: Uint8Array; stats: ReturnType<typeof runExport>["stats"] }
  | { kind: "error"; message: string };

const post = (m: ExportReply, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);

self.onmessage = async (ev: MessageEvent<ExportRequest>) => {
  try {
    const webp = await webpEncoder();
    const input = new Uint8Array(await ev.data.input.arrayBuffer());
    const r = runExport(input, ev.data.options, webp, (p) => post({ kind: "progress", ...p }));
    post({ kind: "done", ...r }, [r.bytes.buffer]);
  } catch (e) {
    post({ kind: "error", message: e instanceof Error ? e.message : String(e) });
  }
};
