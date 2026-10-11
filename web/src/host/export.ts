// Exportação no web: o mesmo pipeline do desktop (web/src/export/, port de
// src-tauri/src/export) num worker.

import type { ExportOptions, ExportStats, Msg, SplatInfo } from "../../../src/lib/api";
import type { ExportReply, ExportRequest } from "../export/export.worker";
import { plyInfo } from "../export/splats";

export async function splatInfo(blob: Blob): Promise<SplatInfo> {
  return { ...plyInfo(new Uint8Array(await blob.slice(0, 65536).arrayBuffer())), bytes: blob.size };
}

type Exported = { name: string; bytes: Uint8Array; stats: Omit<ExportStats, "path" | "seconds"> };

export function exportSplat(
  input: Blob,
  options: ExportOptions,
  progress: (p: { fraction: number; message: Msg }) => void,
): Promise<Exported> {
  const w = new Worker(new URL("../export/export.worker.ts", import.meta.url), { type: "module" });
  return new Promise<Exported>((resolve, reject) => {
    w.onmessage = (ev: MessageEvent<ExportReply>) => {
      const m = ev.data;
      if (m.kind === "progress") return progress(m);
      if (m.kind === "done") resolve(m);
      else reject(new Error(m.message));
    };
    w.onerror = (e) => {
      // Sem mensagem é o script do worker que não carregou: em geral uma aba
      // aberta antes de um deploy, pedindo um arquivo que não existe mais.
      reject(new Error(e.message ? `worker de exportação: ${e.message}` : "não foi possível carregar o worker de exportação; recarregue a página"));
    };
    w.postMessage({ input, options } satisfies ExportRequest);
  }).finally(() => w.terminate());
}
