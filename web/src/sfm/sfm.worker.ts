// Runs the (synchronous, heavy) reconstruction off the UI thread.

import type { Msg } from "../../../src/lib/api";
import type { PairMatches } from "./matcher";
import { type ImageFeatures, reconstruct, type SfmModel } from "./reconstruct";

export type SfmWorkerMessage =
  | { kind: "log"; text: string }
  | { kind: "progress"; frac: number; msg: Msg }
  | { kind: "done"; model: SfmModel }
  | { kind: "error"; text: string };

self.onmessage = (ev: MessageEvent<{ imgs: ImageFeatures[]; matches: PairMatches[]; closedLoop: boolean }>) => {
  const post = (m: SfmWorkerMessage) => (self as unknown as Worker).postMessage(m);
  try {
    const model = reconstruct(
      ev.data.imgs,
      ev.data.matches,
      (text) => post({ kind: "log", text }),
      (frac, msg) => post({ kind: "progress", frac, msg }),
      ev.data.closedLoop,
    );
    post({ kind: "done", model });
  } catch (e) {
    post({ kind: "error", text: String(e) });
  }
};
