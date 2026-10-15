// Worker: TSDF volume → mesh (surface, cleanup, simplification, smoothing,
// UV atlas), off the main thread.

import type { Msg } from "../../../src/lib/api";
import { type BuildOptions, type BuildStats, type BuiltMesh, buildMesh } from "./process.ts";
import type { Volume } from "./tsdf.ts";

export interface MeshRequest {
  volume: Volume;
  options: BuildOptions;
}

export type MeshReply =
  | { kind: "progress"; fraction: number; message: Msg }
  | { kind: "done"; mesh: BuiltMesh; stats: BuildStats }
  | { kind: "error"; message: string };

self.onmessage = async (ev: MessageEvent<MeshRequest>) => {
  const post = (m: MeshReply, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);
  try {
    const { mesh, stats } = await buildMesh(ev.data.volume, ev.data.options, (fraction, message) =>
      post({ kind: "progress", fraction, message }),
    );
    const arrays = [mesh.positions, mesh.normals, mesh.colors, mesh.indices, ...(mesh.uvs ? [mesh.uvs] : [])];
    post({ kind: "done", mesh, stats }, arrays.map((a) => a.buffer));
  } catch (e) {
    post({ kind: "error", message: e instanceof Error ? e.message : String(e) });
  }
};
