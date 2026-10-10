// Roda a reconstrução (síncrona e pesada) fora da thread da interface.

import type { PairMatches } from "./matcher";
import { type ImageFeatures, reconstruct, type SfmModel } from "./reconstruct";

export type SfmWorkerMessage =
  | { kind: "log"; text: string }
  | { kind: "progress"; frac: number; text: string }
  | { kind: "done"; model: SfmModel }
  | { kind: "error"; text: string };

self.onmessage = (ev: MessageEvent<{ imgs: ImageFeatures[]; matches: PairMatches[] }>) => {
  const post = (m: SfmWorkerMessage) => (self as unknown as Worker).postMessage(m);
  try {
    const model = reconstruct(
      ev.data.imgs,
      ev.data.matches,
      (text) => post({ kind: "log", text }),
      (frac, text) => post({ kind: "progress", frac, text }),
    );
    post({ kind: "done", model });
  } catch (e) {
    post({ kind: "error", text: String(e) });
  }
};
