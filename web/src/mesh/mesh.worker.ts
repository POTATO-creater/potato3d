// Worker: TSDF volume → GLB (surface, cleanup, simplification), off the main thread.

import type { Msg } from "../../../src/lib/api";
import { type BuildOptions, type BuildStats, buildGlb } from "./process.ts";
import type { Volume } from "./tsdf.ts";

export interface MeshRequest {
  volume: Volume;
  options: BuildOptions;
}

export type MeshReply =
  | { kind: "progress"; fraction: number; message: Msg }
  | { kind: "done"; glb: Uint8Array; stats: BuildStats }
  | { kind: "error"; message: string };

self.onmessage = async (ev: MessageEvent<MeshRequest>) => {
  const post = (m: MeshReply, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);
  try {
    const { glb, stats } = await buildGlb(ev.data.volume, ev.data.options, (fraction, message) =>
      post({ kind: "progress", fraction, message }),
    );
    post({ kind: "done", glb, stats }, [glb.buffer]);
  } catch (e) {
    post({ kind: "error", message: e instanceof Error ? e.message : String(e) });
  }
};
